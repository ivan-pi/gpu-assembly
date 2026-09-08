/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
/* reclu_solve_{opt,asm} vs reclu_solve_naive: bit-identity of X, and backward
 * error ||A X - B|| / (||A|| ||X||) on the original matrix, for LU computed by
 * reclu_opt_rt with padded lda, m = 1..70, nrhs in {1, 3, 4, 8}. */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "reclu.h"



int main() {
    std::mt19937 rng(11); std::uniform_real_distribution<double> U(-1, 1);
    int tests = 0, bit_fail = 0, res_fail = 0; double worst = 0;
    for (int m = 1; m <= 70; m++) for (int nrhs : {1, 3, 4, 5, 8, 9, 12}) for (int rep = 0; rep < 3; rep++) {
        const int lda = (m + 3) & ~3, ldb = m + (rep % 2);          /* ldb == m or m+1 */
        std::vector<double> A((size_t)lda * m, 0.0), LU;
        for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * lda] = U(rng);
        for (int i = 0; i < m; i++) A[i + (size_t)i * lda] += 2.0 * m;   /* well conditioned */
        LU = A; std::vector<int> ipiv(m); int info;
        reclu_lu_opt(m, m, LU.data(), lda, ipiv.data(), &info);
        std::vector<double> B((size_t)ldb * nrhs); for (auto &b : B) b = U(rng);
        std::vector<double> Xn = B, Xo = B, Xa = B;
        reclu_solve_naive(m, nrhs, LU.data(), lda, ipiv.data(), Xn.data(), ldb);
        reclu_solve_opt  (m, nrhs, LU.data(), lda, ipiv.data(), Xo.data(), ldb);
        reclu_solve_asm  (m, nrhs, LU.data(), lda, ipiv.data(), Xa.data(), ldb);
        tests++;
        /* opt and asm must agree bit-for-bit; opt vs naive differs by the
         * reciprocal-vs-divide rounding (<= a few ulp) -> checked by residual
         * below and by a tight relative bound here. */
        bool same = true; double maxdiff = 0, xmax = 0;
        for (int r = 0; r < nrhs; r++) for (int i = 0; i < m; i++) {
            const size_t q = i + (size_t)r * ldb;
            if (std::memcmp(&Xo[q], &Xa[q], 8)) same = false;
            maxdiff = std::max(maxdiff, std::fabs(Xn[q] - Xo[q])); xmax = std::max(xmax, std::fabs(Xn[q]));
        }
        const double maxrel = maxdiff / xmax;   /* norm-wise */
        if (!same) { bit_fail++; if (bit_fail < 4) std::printf("opt/asm bit mismatch m=%d nrhs=%d\n", m, nrhs); }
        if (maxrel > 1e-12) { bit_fail++; if (bit_fail < 4) std::printf("opt vs naive rel diff %.2e m=%d nrhs=%d\n", maxrel, m, nrhs); }
        /* residual with the optimized solution */
        double err = 0, na = 0, nx = 0;
        for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) na = std::max(na, std::fabs(A[i + (size_t)j * lda]));
        for (int r = 0; r < nrhs; r++) {
            for (int i = 0; i < m; i++) nx = std::max(nx, std::fabs(Xo[i + (size_t)r * ldb]));
            for (int i = 0; i < m; i++) {
                double s = 0; for (int j = 0; j < m; j++) s += A[i + (size_t)j * lda] * Xo[j + (size_t)r * ldb];
                err = std::max(err, std::fabs(s - B[i + (size_t)r * ldb]));
            }
        }
        const double rel = err / (na * nx * m);
        worst = std::max(worst, rel);
        if (rel > 1e-14) { res_fail++; if (res_fail < 4) std::printf("residual m=%d nrhs=%d rel=%.2e\n", m, nrhs, rel); }
    }
    std::printf("%d cases: %d bit mismatches, %d residual failures, worst rel residual %.2e -> %s\n",
                tests, bit_fail, res_fail, worst, (bit_fail || res_fail) ? "FAIL" : "PASS");
    return (bit_fail || res_fail) != 0;
}
