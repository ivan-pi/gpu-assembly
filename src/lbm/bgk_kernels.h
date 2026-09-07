// bgk_kernels.h -- D2Q9 BGK collision kernel (CUDA C++), interface
//
// Twin of bgk_kernel_split in bgk_kernels.cuf; see the header there for
// the lattice numbering, the equilibrium and the storage convention.
// The kernel is defined in bgk_kernels.cu and instantiated there for
// float and double, so a launch only needs this header.

#ifndef BGK_KERNELS_H
#define BGK_KERNELS_H

#include <cuda_runtime.h>   // __global__ for host compilers

// D2Q9 lattice constants in the working precision T.
template <typename T>
struct d2q9 {
    // weights: rest, straight (axis), diagonal
    static constexpr T w0 = T(4) / T(9);
    static constexpr T ws = T(1) / T(9);
    static constexpr T wd = T(1) / T(36);

    static constexpr T one_third    = T(1) / T(3);
    static constexpr T one_half     = T(1) / T(2);
    static constexpr T three_halves = T(3) / T(2);
};

// One node per thread; pdf[i + n*a] is direction a of node i and indp
// receives the direction-independent part of the equilibrium.
template <typename T>
__global__ void bgk_kernel_split(int n, T omega, T *pdf, T *rho, T *ux, T *uy, T *indp);

#endif // BGK_KERNELS_H
