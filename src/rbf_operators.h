#ifndef RBF_OPERATORS_H
#define RBF_OPERATORS_H

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <cusolverdx.hpp>

#include "cuda_arch.h"
#include "rbf_rbffd.h"

namespace rbf_operators {

namespace cs = cusolverdx;

template<typename T, cs::function F, int NT, int NRHS, unsigned int ARCH>
using Solver = decltype(cs::Size<NT,NT,NRHS>()
                        + cs::Function<F>()
                        + cs::Type<cs::type::real>()
                        + cs::Precision<T>()
                        + cs::Arrangement<cs::col_major,cs::col_major>()
                        + cs::LeadingDimension<NT>()
                        + cs::SM<ARCH>()
                        + cs::Block());

// Solve via LU factorization with partial pivoting
template<typename T, int NT, int NRHS, unsigned int ARCH>
using GESV = Solver<T, cs::function::gesv_partial_pivot, NT, NRHS, ARCH>;

// General Least Square (using QR)
template<typename T, int NT, int NRHS, unsigned int ARCH>
using GELS = Solver<T, cs::function::gels, NT, NRHS, ARCH>;


// Test function assembling a single kernel
template<class GESV_solver>
__device__ void solve_system_AB(
        typename GESV_solver::a_data_type* A,
        int * ipiv,
        typename GESV_solver::b_data_type* B,
        typename GESV_solver::status_type* info) {

    static_assert(GESV_solver::is_block_execution,"assumes block configuration");
    GESV_solver().execute(A, ipiv, B, info);
}


// Static (compile-time) configuration for RBF-FD assembly kernels
//
// Template parameters:
//   N     stencil size, fixed across all stencils
//   NRHS  number of right-hand sides (operators) per stencil
//   P     polynomial augmentation degree; npoly = (P+1)*(P+2)/2
//   Q     PHS exponent, phi(r) = r^Q
//   ARCH  target architecture, cuSolverDx SM<CC> code (750, 800, ..., 1210)
//   T     real type, float or double
//
// Launch contract (see assemble_interp_weights):
//   gridDim.x             = nstencils
//   blockdim              = block_dim
//   dynamic shared memory = shared_memory_size()
//
//   If needs_dynamic_smem_opt_in is true, the caller
//   must first call cudaFuncSetAttribute(kernel,
//     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size()).
//
// Shared-memory layout, in order:
//   xs[N], ys[N], As[nt*nt], Bs[nt*NRHS], ipiv[nt]
//   shared_memory_size() and slice() must agree field for field.
//
template<int N_, int NRHS_, int P_, int Q_ = 3,
         unsigned int ARCH_ = 800,
         typename T_ = double>
struct interp_config {

    using value_type = T_;
    static constexpr int N = N_, NRHS = NRHS_, P = P_, Q = Q_;
    static constexpr unsigned int ARCH = ARCH_;

    static constexpr int npoly = PolyBasis<P>::np;
    static constexpr int nt = N + npoly;

    // --- Solver ---

    static_assert(cuda_arch::is_cusolverdx_sm(ARCH),
                  "ARCH is not a code accepted by cuSolverDx's SM<CC>");
    static_assert(std::is_same_v<value_type, float> ||
                  std::is_same_v<value_type, double>,
                  "Solver expects Type<type::real>()");

    using solver = Solver<value_type, cs::function::gesv_partial_pivot,
                          nt, NRHS, ARCH>;

    static_assert(solver::is_block_execution,
                  "interp kernels assume one cooperative solve per thread block");
    static_assert(solver::batches_per_block == 1,
                  "interp kernels map one stencil per block");
    static_assert(solver::block_dim.y == 1 && solver::block_dim.z == 1,
                  "interp fill loops assume a 1-D block");

    // FIXME: allow using GELS here. This requires two changes:
    //        workspace for tau (length nt) and the execute method
    //        does not need info

    // --- Architecture ---

    static constexpr auto& arch = cuda_arch::limits_v<solver::sm>;


    // --- Shared memory layout ---

    static constexpr std::size_t align_up(std::size_t n, std::size_t a) {
        return ((n + a - 1) / a) * a;
    }

    static constexpr std::size_t shared_memory_size() {
        constexpr std::size_t av = alignof(value_type);
        constexpr std::size_t ai = alignof(int);
        std::size_t total = 0;
        total = align_up(total, av) + sizeof(value_type) * N;         // xs
        total = align_up(total, av) + sizeof(value_type) * N;         // ys
        total = align_up(total, av) + sizeof(value_type) * nt * nt;   // As
        total = align_up(total, av) + sizeof(value_type) * nt * NRHS; // Bs
        total = align_up(total, ai) + sizeof(int) * nt;               // ipiv
        return total;
    }

    // Returns pointers to arrays xs, ys, As, Bs, ipiv.
    __device__ static auto slice(cs::byte* p) {
        return cs::shared_memory::slice<
                value_type, value_type, value_type, value_type, int>(
            p,
            alignof(value_type), N,         // xs
            alignof(value_type), N,         // ys
            alignof(value_type), nt * nt,   // As
            alignof(value_type), nt * NRHS, // Bs
            alignof(int), nt);              // ipiv
    }

    // Hard limit: the whole footprint must fit one block on this ARCH.
    static_assert(arch.fits(shared_memory_size()),
                  "shared memory footprint exceeds the per-block limit for ARCH");

    // Cross-check against cuSolverDx's own accounting for As + Bs.
    static_assert(sizeof(value_type) * (nt * nt + nt * NRHS) >= solver::shared_memory_size,
                  "manual A/B slice is smaller than Solver::shared_memory_size");

    // --- Launch-side parameters ---

    static constexpr auto     block_dim = solver::block_dim;
    static constexpr unsigned nthreads  = solver::max_threads_per_block;

    // True when the footprint exceeds the 48 KB static limit. The caller must
    // then launch with dynamic shared memory and, once per kernel, call
    //
    //   cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                        interp_config<...>::shared_memory_size());
    static constexpr bool needs_dynamic_smem_opt_in =
        arch.needs_dynamic_smem_opt_in(shared_memory_size());

};

// Assembles interpolation operators for semi-Lagrangian streaming.
//
// One thread block per stencil: gathers the stencil nodes into the
// stencil-local frame centred on point s, fills the augmented collocation
// matrix and the NRHS right-hand sides, solves with Config::solver, and
// scatters the first N rows of each solution into the CSR values arrays.
// The trailing npoly polynomial coefficients are discarded.
//
// Template parameters
//   Config      an interp_config<...>; supplies N, NRHS, P, Q, the solver,
//               the shared-memory layout, and the launch parameters
//
// Arguments
//    nstencils  number of stencils (= number of points)
//    x, y       point coordinates, length nstencils
//    ja         adjacency graph (column indices), length nnz = nstencils*N,
//               the N neighbour indices of stencil s are contiguous at ja[s*N].
//               This is CSR with a fixed row length, so the row pointer
//               is implicit, ia[s] = s*N, and is not passed.
//    A          NRHS pointers, each to a CSR values array of length
//               nnz = nstencils*N with the same layout as ja: A[op][s*N + k]
//               is the weight of neighbor ja[s*N + k] for operator op
//    xc, yc     interpolation centres in stencil-local frame, length NRHS
//    info       per-stencil solver status, length nstencils; 0 on success
//
// Launch:
//    <<<nstencils, Config::block_dim, Config::shared_memory_size()>>>
//
template<class Config, typename T = typename Config::value_type>
__global__ void assemble_interp_weights(
        const int  nstencils,
        const T* __restrict__ x,
        const T* __restrict__ y,
        const int* __restrict__ ja,
        T* const* __restrict__ A,
        const T* __restrict__ xc,
        const T* __restrict__ yc,
        int* __restrict__ info) {

    CUSOLVERDX_SKIP_IF_NOT_APPLICABLE_SM(typename Config::solver);

    constexpr auto N = Config::N; // stencil size
    constexpr auto P = Config::P; // polynomial degree
    constexpr auto Q = Config::Q; // PHS exponent, phi(r) = r^Q

    constexpr auto nt = Config::nt;
    constexpr auto NRHS = Config::NRHS;

    using solver = typename Config::solver;

    const int s = blockIdx.x;
    if (s >= nstencils) return;

    // Allocate this dynamically
    extern __shared__ __align__(16) cs::byte shared_mem[];

    // Slice shared memory into pointers
    auto [xs, ys, As, Bs, ipiv] = Config::slice(shared_mem);

    const int tid = threadIdx.x;
    const int nthreads = Config::nthreads;

    // Gather and shift to the stencil-local frame; the
    // stencil is centred on point s.
    const T x0 = x[s], y0 = y[s];
    for (int k = tid; k < N; k += nthreads) {
        xs[k] = x[ja[s*N + k]] - x0;
        ys[k] = y[ja[s*N + k]] - y0;
    }
    __syncthreads();

    fill_matrix<N, P, Q>(As, xs, ys, tid, nthreads);
    fill_rhs<N, P, NRHS, Q>(Bs, xs, ys, xc, yc, tid, nthreads);
    __syncthreads();

    solver().execute(As, ipiv, Bs, &info[s]);
    __syncthreads();

    // Scatter into the CSR storage; the first N rows are the interpolation
    // weights, the trailing polynomial coefficients are discarded.
    for (int col = 0; col < NRHS; col++) {
        T* __restrict__ Ag = A[col];
        for (int k = tid; k < N; k += nthreads) {
            Ag[s*N + k] = Bs[col*nt + k];
        }
    }
}

} // namespace rbf_operators

#endif // RBF_OPERATORS_H