/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#include "reclu.h"
#include <cmath>

/* Column-oriented forward/back substitution, the way DTRSM's reference
 * implementation walks a column-major triangular matrix. Each element x[i]
 * receives its updates in increasing k (L) / decreasing k (U), each as one
 * fused multiply-add; the optimized kernels keep exactly this order. */
void reclu_solve_naive(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) {
    for (int r = 0; r < nrhs; r++) {
        double *x = B + (size_t)r * ldb;
        for (int k = 0; k < m; k++) {                       /* P */
            const int kp = ipiv[k] - 1;
            if (kp != k) { const double t = x[k]; x[k] = x[kp]; x[kp] = t; }
        }
        for (int k = 0; k < m; k++) {                       /* L y = P b, unit diagonal */
            const double xk = x[k];
            const double *c = LU + (size_t)k * lda;
            for (int i = k + 1; i < m; i++) x[i] = std::fma(-c[i], xk, x[i]);
        }
        for (int k = m - 1; k >= 0; k--) {                  /* U x = y */
            const double *c = LU + (size_t)k * lda;
            x[k] /= c[k];
            const double xk = x[k];
            for (int i = 0; i < k; i++) x[i] = std::fma(-c[i], xk, x[i]);
        }
    }
}
