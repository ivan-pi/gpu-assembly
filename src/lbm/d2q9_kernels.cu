// d2q9_kernels.cu -- D2Q9 lattice Boltzmann kernels (CUDA C++)
//
// Twins of the kernels in d2q9_kernels.cuf; see the header there for the
// lattice numbering, the equilibrium and the storage convention. The two
// files are kept in step line by line. Templated on the real type T and
// instantiated for float and double at the end of the file; wp in
// d2q9_kernels.cuf selects one of the two.

#include "d2q9_kernels.h"

// Equilibrium populations for the given rho, ux, uy; the same arithmetic
// as bgk_kernel_split with omega = 1.
template <typename T>
__global__ void feq_kernel(const int n, const T *rho, const T *ux, const T *uy, T *pdf)
{
    using lattice = d2q9<T>;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const T rho_i = rho[i];
    const T ux_i = ux[i];
    const T uy_i = uy[i];
    const T uxsq = ux_i * ux_i;
    const T uysq = uy_i * uy_i;

    // direction-independent part of the equilibrium
    const T indp_i = lattice::one_third - lattice::one_half * (uxsq + uysq);

    // opposite directions share the even terms
    pdf[i] = lattice::three_w0 * rho_i * indp_i;

    const T vel_trm_13 = indp_i + lattice::three_halves * uxsq;
    pdf[i + n*1] = lattice::three_ws * rho_i * (vel_trm_13 + ux_i);
    pdf[i + n*3] = lattice::three_ws * rho_i * (vel_trm_13 - ux_i);

    const T vel_trm_24 = indp_i + lattice::three_halves * uysq;
    pdf[i + n*2] = lattice::three_ws * rho_i * (vel_trm_24 + uy_i);
    pdf[i + n*4] = lattice::three_ws * rho_i * (vel_trm_24 - uy_i);

    const T velxpy = ux_i + uy_i;
    const T vel_trm_57 = indp_i + lattice::three_halves * velxpy * velxpy;
    pdf[i + n*5] = lattice::three_wd * rho_i * (vel_trm_57 + velxpy);
    pdf[i + n*7] = lattice::three_wd * rho_i * (vel_trm_57 - velxpy);

    const T velxmy = ux_i - uy_i;
    const T vel_trm_68 = indp_i + lattice::three_halves * velxmy * velxmy;
    pdf[i + n*6] = lattice::three_wd * rho_i * (vel_trm_68 - velxmy);
    pdf[i + n*8] = lattice::three_wd * rho_i * (vel_trm_68 + velxmy);
}

// Density and velocity from the populations; the same expressions
// bgk_kernel_split uses for its rho, ux, uy outputs.
template <typename T>
__global__ void macros_kernel(const int n, const T *pdf, T *rho, T *ux, T *uy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // populations of node i
    T f[9];
    #pragma unroll
    for (int a = 0; a < 9; ++a) {
        f[a] = pdf[i + n*a];
    }

    // density
    const T rho_i = (((f[5] + f[7]) + (f[6] + f[8])) + ((f[1] + f[3]) + (f[2] + f[4]))) + f[0];
    rho[i] = rho_i;
    const T invrho = T(1) / rho_i;

    // velocity
    ux[i] = invrho * (((f[5] - f[7]) + (f[8] - f[6])) + (f[1] - f[3]));
    uy[i] = invrho * (((f[5] - f[7]) + (f[6] - f[8])) + (f[2] - f[4]));
}

// BGK collision in place; also returns rho, ux, uy of the pre-collision
// populations and indp, the direction-independent part of the equilibrium.
template <typename T>
__global__ void bgk_kernel_split(const int n, const T omega,
                                 T *pdf, T *rho, T *ux, T *uy, T *indp)
{
    using lattice = d2q9<T>;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const T omegabar = T(1) - omega;
    const T omega_w0 = omega * lattice::three_w0;
    const T omega_ws = omega * lattice::three_ws;
    const T omega_wd = omega * lattice::three_wd;

    // populations of node i
    T f[9];
    #pragma unroll
    for (int a = 0; a < 9; ++a) {
        f[a] = pdf[i + n*a];
    }

    // density
    const T rho_i = (((f[5] + f[7]) + (f[6] + f[8])) + ((f[1] + f[3]) + (f[2] + f[4]))) + f[0];
    rho[i] = rho_i;
    const T invrho = T(1) / rho_i;

    // velocity
    const T ux_i = invrho * (((f[5] - f[7]) + (f[8] - f[6])) + (f[1] - f[3]));
    const T uy_i = invrho * (((f[5] - f[7]) + (f[6] - f[8])) + (f[2] - f[4]));
    ux[i] = ux_i;
    uy[i] = uy_i;
    const T uxsq = ux_i * ux_i;
    const T uysq = uy_i * uy_i;

    // direction-independent part of the equilibrium
    const T indp_i = lattice::one_third - lattice::one_half * (uxsq + uysq);
    indp[i] = indp_i;

    // relax towards equilibrium; opposite directions share the even terms
    pdf[i] = omegabar * f[0] + omega_w0 * rho_i * indp_i;

    const T vel_trm_13 = indp_i + lattice::three_halves * uxsq;
    pdf[i + n*1] = omegabar * f[1] + omega_ws * rho_i * (vel_trm_13 + ux_i);
    pdf[i + n*3] = omegabar * f[3] + omega_ws * rho_i * (vel_trm_13 - ux_i);

    const T vel_trm_24 = indp_i + lattice::three_halves * uysq;
    pdf[i + n*2] = omegabar * f[2] + omega_ws * rho_i * (vel_trm_24 + uy_i);
    pdf[i + n*4] = omegabar * f[4] + omega_ws * rho_i * (vel_trm_24 - uy_i);

    const T velxpy = ux_i + uy_i;
    const T vel_trm_57 = indp_i + lattice::three_halves * velxpy * velxpy;
    pdf[i + n*5] = omegabar * f[5] + omega_wd * rho_i * (vel_trm_57 + velxpy);
    pdf[i + n*7] = omegabar * f[7] + omega_wd * rho_i * (vel_trm_57 - velxpy);

    const T velxmy = ux_i - uy_i;
    const T vel_trm_68 = indp_i + lattice::three_halves * velxmy * velxmy;
    pdf[i + n*6] = omegabar * f[6] + omega_wd * rho_i * (vel_trm_68 - velxmy);
    pdf[i + n*8] = omegabar * f[8] + omega_wd * rho_i * (vel_trm_68 + velxmy);
}

template __global__ void feq_kernel<float >(int, const float  *, const float  *, const float  *, float  *);
template __global__ void feq_kernel<double>(int, const double *, const double *, const double *, double *);

template __global__ void macros_kernel<float >(int, const float  *, float  *, float  *, float  *);
template __global__ void macros_kernel<double>(int, const double *, double *, double *, double *);

template __global__ void bgk_kernel_split<float >(int, float,  float  *, float  *, float  *, float  *, float  *);
template __global__ void bgk_kernel_split<double>(int, double, double *, double *, double *, double *, double *);
