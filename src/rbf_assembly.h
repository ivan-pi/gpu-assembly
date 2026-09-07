// rbf_assembly.h -- host RBF-FD assembly
//
// The assembly API, and a reference host implementation of it. An
// assembler is a callable
//
//     weights = assemble(xc, yc, ja, functionals)
//
//     xc, yc        stencil centres, length n (the nodes themselves for
//                   arrival-point stencils, the departure points for
//                   departure-point stencils)
//     ja            n*k column indices, row-major (rbf::periodic_knn,
//                   NodeSet::stencils)
//     functionals   what to evaluate, each with a point in the local
//                   frame of the stencil centre
//     weights       one n*k array per functional, laid out like ja
//
// The cloud the indices refer to is bound at construction, together with
// the periodic box (if any). This is the same contract the cuSolverDx
// kernel assemble_interp_weights fulfils on the device, so a GPU
// assembler is a second class with the same call, and the streaming
// scheme builders in lbm_schemes.h take either.
//
// HostAssembler fills every stencil's system with the shared code in
// rbf_rbffd.h (the same fill the CUDA kernel runs) and solves it with a
// small dense LU. It is a reference: an optimised CPU assembly (LAPACK,
// blocked, reusing the factorisation across functionals) replaces the
// solve, not the interface.

#ifndef RBF_ASSEMBLY_H
#define RBF_ASSEMBLY_H

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "rbf_rbffd.h"
#include "rbf_periodic.h"

namespace rbf {

using rbf_operators::functional;

template<typename T>
struct Functional {
    functional op = functional::value;
    T x = 0, y = 0;   // evaluation point, local frame
};

namespace detail {

// Dense LU with partial pivoting, column-major A (n x n, lda = n),
// B (n x nrhs, ldb = n) overwritten with the solution. Returns 0 or the
// 1-based index of the first zero pivot, as LAPACK's dgesv would.
template<typename T>
int lu_solve(int n, T* A, int* piv, T* B, int nrhs) {
    for (int c = 0; c < n; ++c) {
        int p = c;
        T best = std::abs(A[c + c * n]);
        for (int r = c + 1; r < n; ++r) {
            const T v = std::abs(A[r + c * n]);
            if (v > best) { best = v; p = r; }
        }
        piv[c] = p;
        if (best == T(0)) return c + 1;
        if (p != c) {
            for (int j = 0; j < n; ++j) std::swap(A[c + j * n], A[p + j * n]);
            for (int j = 0; j < nrhs; ++j) std::swap(B[c + j * n], B[p + j * n]);
        }
        const T inv = T(1) / A[c + c * n];
        for (int r = c + 1; r < n; ++r) A[r + c * n] *= inv;
        for (int j = c + 1; j < n; ++j) {
            const T a = A[c + j * n];
            if (a == T(0)) continue;
            for (int r = c + 1; r < n; ++r) A[r + j * n] -= A[r + c * n] * a;
        }
    }
    for (int j = 0; j < nrhs; ++j) {
        T* b = B + j * n;
        for (int c = 0; c < n; ++c) {          // forward, unit lower
            const T bc = b[c];
            if (bc == T(0)) continue;
            for (int r = c + 1; r < n; ++r) b[r] -= A[r + c * n] * bc;
        }
        for (int c = n - 1; c >= 0; --c) {     // backward, upper
            b[c] /= A[c + c * n];
            const T bc = b[c];
            for (int r = 0; r < c; ++r) b[r] -= A[r + c * n] * bc;
        }
    }
    return 0;
}

} // namespace detail

// N stencil size, P polynomial degree, Q PHS exponent: the compile-time
// configuration mirrors interp_config on the device, so a stencil size
// is a template instantiation on both sides.
template<int N, int P, int Q = 3, typename T = double, typename I = std::int32_t>
class HostAssembler {
public:
    using value_type = T;
    using index_type = I;
    static constexpr int stencil_size = N;
    static constexpr int poly_degree = P;
    static constexpr int phs_exponent = Q;
    static constexpr int nt = N + rbf_operators::npoly(P);

    HostAssembler(std::span<const T> x, std::span<const T> y,
                  const PeriodicBox<T>* box = nullptr)
        : x_(x), y_(y), box_(box) {
        if (x_.size() != y_.size()) throw std::invalid_argument("HostAssembler: x/y size mismatch");
    }

    std::vector<std::vector<T>>
    operator()(std::span<const T> xc, std::span<const T> yc,
               std::span<const I> ja, std::span<const Functional<T>> L) const
    {
        const std::size_t n = xc.size();
        const int nrhs = static_cast<int>(L.size());
        if (yc.size() != n || ja.size() != n * N)
            throw std::invalid_argument("HostAssembler: centres/ja sizes do not match N");

        std::vector<functional> ops(nrhs);
        std::vector<T> lx(nrhs), ly(nrhs);
        for (int r = 0; r < nrhs; ++r) { ops[r] = L[r].op; lx[r] = L[r].x; ly[r] = L[r].y; }

        std::vector<std::vector<T>> out(nrhs, std::vector<T>(n * N));
        std::size_t singular = 0;

        #pragma omp parallel reduction(+:singular)
        {
            std::vector<T> A(std::size_t(nt) * nt), B(std::size_t(nt) * nrhs);
            std::vector<int> piv(nt);
            T xs[N], ys[N];

            #pragma omp for schedule(static)
            for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(n); ++s) {
                const T x0 = xc[s], y0 = yc[s];
                for (int j = 0; j < N; ++j) {
                    const I c = ja[s * N + j];
                    T dx = x_[c] - x0, dy = y_[c] - y0;
                    if (box_) { dx = box_->wrap_dx(dx); dy = box_->wrap_dy(dy); }
                    xs[j] = dx; ys[j] = dy;
                }
                rbf_operators::fill_matrix<N, P, Q>(A.data(), xs, ys, 0, 1);
                rbf_operators::fill_rhs_functionals<N, P, Q>(
                    B.data(), xs, ys, ops.data(), lx.data(), ly.data(), nrhs, 0, 1);
                if (detail::lu_solve(nt, A.data(), piv.data(), B.data(), nrhs)) {
                    ++singular;
                    continue;
                }
                for (int r = 0; r < nrhs; ++r)
                    for (int j = 0; j < N; ++j)
                        out[r][s * N + j] = B[std::size_t(r) * nt + j];
            }
        }
        if (singular)
            throw std::runtime_error("HostAssembler: " + std::to_string(singular) + " singular stencil(s)");
        return out;
    }

private:
    std::span<const T> x_, y_;
    const PeriodicBox<T>* box_;
};

} // namespace rbf

#endif // RBF_ASSEMBLY_H
