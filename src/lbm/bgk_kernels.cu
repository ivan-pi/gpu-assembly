// bgk_kernels.cu -- D2Q9 BGK collision kernel (CUDA C++)
//
// Twin of bgk_kernel_split in bgk_kernels.F90; see the header there for
// the lattice numbering, the equilibrium and the storage convention.
// The two are kept in step line by line.

using real_t = float;   // working precision; matches wp in bgk_kernels.F90

// D2Q9 weights: rest, straight (axis), diagonal
constexpr real_t w0 = real_t(4) / real_t(9);
constexpr real_t ws = real_t(1) / real_t(9);
constexpr real_t wd = real_t(1) / real_t(36);

constexpr real_t one_third    = real_t(1) / real_t(3);
constexpr real_t one_half     = real_t(1) / real_t(2);
constexpr real_t three_halves = real_t(3) / real_t(2);

// One node per thread; pdf[i + n*a] is direction a of node i and indp
// receives the direction-independent part of the equilibrium.
__global__ void bgk_kernel_split(const int n, const real_t omega,
                                 real_t *pdf, real_t *rho, real_t *ux, real_t *uy,
                                 real_t *indp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const real_t omegabar = real_t(1) - omega;
    const real_t omega_w0 = real_t(3) * omega * w0;
    const real_t omega_ws = real_t(3) * omega * ws;
    const real_t omega_wd = real_t(3) * omega * wd;

    // populations of node i
    real_t f[9];
    for (int a = 0; a < 9; ++a) {
        f[a] = pdf[i + n*a];
    }

    // density
    const real_t rho_i = (((f[5] + f[7]) + (f[6] + f[8])) + ((f[1] + f[3]) + (f[2] + f[4]))) + f[0];
    rho[i] = rho_i;
    const real_t invrho = real_t(1) / rho_i;

    // velocity
    const real_t ux_i = invrho * (((f[5] - f[7]) + (f[8] - f[6])) + (f[1] - f[3]));
    const real_t uy_i = invrho * (((f[5] - f[7]) + (f[6] - f[8])) + (f[2] - f[4]));
    ux[i] = ux_i;
    uy[i] = uy_i;
    const real_t uxsq = ux_i * ux_i;
    const real_t uysq = uy_i * uy_i;

    // direction-independent part of the equilibrium
    const real_t indp_i = one_third - one_half * (uxsq + uysq);
    indp[i] = indp_i;

    // relax towards equilibrium; opposite directions share the even terms
    pdf[i] = omegabar * f[0] + omega_w0 * rho_i * indp_i;

    const real_t vel_trm_13 = indp_i + three_halves * uxsq;
    pdf[i + n*1] = omegabar * f[1] + omega_ws * rho_i * (vel_trm_13 + ux_i);
    pdf[i + n*3] = omegabar * f[3] + omega_ws * rho_i * (vel_trm_13 - ux_i);

    const real_t vel_trm_24 = indp_i + three_halves * uysq;
    pdf[i + n*2] = omegabar * f[2] + omega_ws * rho_i * (vel_trm_24 + uy_i);
    pdf[i + n*4] = omegabar * f[4] + omega_ws * rho_i * (vel_trm_24 - uy_i);

    const real_t velxpy = ux_i + uy_i;
    const real_t vel_trm_57 = indp_i + three_halves * velxpy * velxpy;
    pdf[i + n*5] = omegabar * f[5] + omega_wd * rho_i * (vel_trm_57 + velxpy);
    pdf[i + n*7] = omegabar * f[7] + omega_wd * rho_i * (vel_trm_57 - velxpy);

    const real_t velxmy = ux_i - uy_i;
    const real_t vel_trm_68 = indp_i + three_halves * velxmy * velxmy;
    pdf[i + n*6] = omegabar * f[6] + omega_wd * rho_i * (vel_trm_68 - velxmy);
    pdf[i + n*8] = omegabar * f[8] + omega_wd * rho_i * (vel_trm_68 + velxmy);
}
