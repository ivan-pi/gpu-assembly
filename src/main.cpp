
#include <nanoflann.hpp>
#include "rbf_operators.h"

namespace poisson {}

int main() {
    constexpr int N = 21;           // stencil size
    constexpr int P = 2;            // augmentation degree
    constexpr int Q = 3;            // PHS exponent, phi = r^Q (odd)
    constexpr unsigned ARCH = 900;  // cusolverDx SM<> code of the target GPU

    using T = double;

    using Config = rbf_operators::laplace_config<N, P, Q, ARCH, T>;
    auto laplace_weight_kernel = &rbf_operators::assemble_laplace_weights<Config>;

    constexpr int nt = N + operators::npoly(P);

    const int nnz = nstencils * N;

    using T = double using solver = operators::GESV<T, nt, NRHS, ARCH>

        // 1) Read coordinates from file

        // --- 1. nodes and stencils ---

        const NodeSet nodes(nodefile);
    const int npoints = nodes.npoints();

    // Adjacency graph (host)
    const std::vector<int> ja_h = nodes.stencils(N);

    // --- 2. batched assembly over all nodes ---
    {
        const dim3 block = Config::block_dim;
        const unsigned smem = Config::shared_memory_size();

        if (Config::needs_dynamic_smem_opt_in) {
            CHECK_CUDA(cudaFuncSetAttribute(reinterpret_cast<const void*>(assembly_kernel),
                                            cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }

        laplace_weight_kernel<<<npoints, block, smem>>>(npoints, ptr(x_d), ptr(y_d), ptr(A.ja),
                                                        ptr(A_tab), ptr(info_d));
        CHECK_CUDA(cudaGetLastError());

        thrust::host_vector<int> info = info_d;
        const int bad_stencils =
            std::count_if(info.begin(), info.end(), [](int i) { return i != 0; });
        if (bad_stencils) {
            std::fprintf(stderr, "error: %d singular stencils\n", bad_stencils);
            return EXIT_FAILURE;
        }
    }

    // --- 3. operator

    auto mv_op = [&](const T* x, T* y) {
        // Apply bulk operator
        A.spmv(x, y);

        // Fix boundary values
        const int* bi = ptr(bnd_d);
        parallel_for(nbnd, [=] __device__(int k) { y[bi[k]] = x[bi[k]]; });
    }

    // --- 5. solve ---
}
