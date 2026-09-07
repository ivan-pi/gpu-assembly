// d2q9_kernels.h -- D2Q9 lattice Boltzmann kernels (CUDA C++), interface
//
// Twins of the kernels in d2q9_kernels.cuf; see the header there for the
// lattice numbering, the equilibrium and the storage convention. The
// kernels are defined in d2q9_kernels.cu and instantiated there for
// float and double, so a launch only needs this header.
//
// One thread per node throughout; pdf[i + n*a] is direction a of node i.

#ifndef D2Q9_KERNELS_H
#define D2Q9_KERNELS_H

#include <cuda_runtime.h>   // __global__ for host compilers

// D2Q9 lattice constants in the working precision T.
template <typename T>
struct d2q9 {
    // weights: rest, straight (axis), diagonal
    static constexpr T w0 = T(4) / T(9);
    static constexpr T ws = T(1) / T(9);
    static constexpr T wd = T(1) / T(36);

    static constexpr T three_w0 = T(3) * w0;
    static constexpr T three_ws = T(3) * ws;
    static constexpr T three_wd = T(3) * wd;

    static constexpr T one_third    = T(1) / T(3);
    static constexpr T one_half     = T(1) / T(2);
    static constexpr T three_halves = T(3) / T(2);
};

// Equilibrium populations for the given rho, ux, uy. Same arithmetic as
// bgk_kernel_split with omega = 1, so the two agree bit for bit.
template <typename T>
__global__ void feq_kernel(int n, const T *rho, const T *ux, const T *uy, T *pdf);

// Density and velocity from the populations; the same expressions
// bgk_kernel_split uses for its rho, ux, uy outputs.
template <typename T>
__global__ void macros_kernel(int n, const T *pdf, T *rho, T *ux, T *uy);

// BGK collision in place; also returns rho, ux, uy of the pre-collision
// populations and indp, the direction-independent part of the equilibrium.
template <typename T>
__global__ void bgk_kernel_split(int n, T omega, T *pdf, T *rho, T *ux, T *uy, T *indp);

#endif // D2Q9_KERNELS_H
