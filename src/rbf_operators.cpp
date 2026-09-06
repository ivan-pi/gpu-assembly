#include "rbf_operators.h"

constexpr auto arch = 900u; // Hopper

// Solve a single system AX = B using the cuSolverDx library
// Dimensions are fixed at M = N = 15, and K = NRHS = 9
extern "C" __global__ void solve_dp(
    double* A,
    int* ipiv,
    double* B,
    int* info) {

    using solver =
        rbf_operators::GESV<double, /* NT */ 15, /* NRHS */ 9, /* ARCH */ arch>;

    rbf_operators::solve_system_AB<solver>(A,ipiv,B,info);
}
