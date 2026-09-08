/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
/* Adversarial tests for reclu_solve_{opt,asm} against reclu_solve_naive.
 *
 * Each case builds an LU (via reclu_opt_rt, or via the reference reclu_lu
 * with unpadded lda to exercise the fallback), a right-hand side block, and
 * compares:
 *   - opt vs asm: bit-identical always (NaN sign/payload ignored);
 *   - opt vs naive: for finite outputs, |diff| <= tol * (|x| + ||x||_inf)
 *     with tol scaled by n (reciprocal vs divide differs per pivot step);
 *     for non-finite outputs, the NaN/Inf pattern must match (same element
 *     is NaN in both, or +-Inf with matching sign);
 *   - pivots/ipiv are shared, so the permutation logic is tested by the
 *     "reversal" and "random full-rank" pivot patterns.
 * Categories: identity / reversed / random pivoting, singular LU (zero
 * pivot, exact dgetf2 info > 0), ill-conditioned (Hilbert-like, 1e-12 scale
 * diagonal), huge/tiny scaling (overflow, denormal), NaN and Inf in B,
 * all-zero B, +-0 B, ldb > m, ldb == m, unpadded lda (fallback), nrhs
 * 0..13 (single, 4-lane, 8-lane, remainder paths), m = 1..3 and 63..66. */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "reclu.h"



static int tests = 0, fails = 0;
static std::mt19937 rng(2024);

static bool same_bits_nanaware(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof a) == 0;
}

static void check(const char *what, int m, int nrhs, const std::vector<double> &LU, int lda, const std::vector<int> &ipiv,
                  const std::vector<double> &B, int ldb) {
    std::vector<double> Xn = B, Xo = B, Xa = B;
    reclu_solve_naive(m, nrhs, LU.data(), lda, ipiv.data(), Xn.data(), ldb);
    reclu_solve_opt  (m, nrhs, LU.data(), lda, ipiv.data(), Xo.data(), ldb);
    reclu_solve_asm  (m, nrhs, LU.data(), lda, ipiv.data(), Xa.data(), ldb);
    tests++;
    /* rows >= m and columns >= nrhs of B must be untouched */
    for (int r = 0; r < nrhs; r++) for (int i = m; i < ldb; i++) {
        const size_t q = i + (size_t)r * ldb;
        if (!same_bits_nanaware(B[q], Xo[q]) || !same_bits_nanaware(B[q], Xa[q])) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: pad row %d modified\n", what, m, nrhs, i); return; }
    }
    double xinf = 0;
    for (int r = 0; r < nrhs; r++) for (int i = 0; i < m; i++) if (std::isfinite(Xn[i + (size_t)r * ldb])) xinf = std::max(xinf, std::fabs(Xn[i + (size_t)r * ldb]));
    const double tol = 4.0 * 2.2e-16 * (m + 4);
    for (int r = 0; r < nrhs; r++) for (int i = 0; i < m; i++) {
        const size_t q = i + (size_t)r * ldb;
        if (!same_bits_nanaware(Xo[q], Xa[q])) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: opt/asm differ at (%d,%d): %.17g vs %.17g\n", what, m, nrhs, i, r, Xo[q], Xa[q]); return; }
        const double n_ = Xn[q], o_ = Xo[q];
        if (std::isnan(n_) || std::isnan(o_)) {
            if (!(std::isnan(n_) && std::isnan(o_))) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: NaN pattern differs at (%d,%d): %g vs %g\n", what, m, nrhs, i, r, n_, o_); return; }
            continue;
        }
        if (std::isinf(n_) || std::isinf(o_)) {
            if (n_ != o_) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: Inf pattern differs at (%d,%d): %g vs %g\n", what, m, nrhs, i, r, n_, o_); return; }
            continue;
        }
        /* absolute floor: denormal results have a fixed spacing of 2^-1074, so
         * their relative precision is far worse than eps */
        if (std::fabs(n_ - o_) > tol * (std::fabs(n_) + xinf) + 16.0 * (m + 4) * 4.9406564584124654e-324) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: (%d,%d) naive %.17g opt %.17g\n", what, m, nrhs, i, r, n_, o_); return; }
    }
}

struct Sys { int m, lda; std::vector<double> A, LU; std::vector<int> ipiv; int info; };

/* For ill-conditioned (numerically singular) systems the forward solutions
 * of two rounding-different algorithms legitimately differ by kappa*eps, so
 * compare backward errors instead: r = ||A x - b||_inf / (||A||_inf ||x||_inf + ||b||_inf)
 * of opt must be no worse than 4x that of naive (+ 4 eps m). opt/asm still
 * bit-identical. */
static void check_backward(const char *what, const Sys &s, int nrhs, const std::vector<double> &B, int ldb) {
    const int m = s.m;
    std::vector<double> Xn = B, Xo = B, Xa = B;
    reclu_solve_naive(m, nrhs, s.LU.data(), s.lda, s.ipiv.data(), Xn.data(), ldb);
    reclu_solve_opt  (m, nrhs, s.LU.data(), s.lda, s.ipiv.data(), Xo.data(), ldb);
    reclu_solve_asm  (m, nrhs, s.LU.data(), s.lda, s.ipiv.data(), Xa.data(), ldb);
    tests++;
    for (size_t q = 0; q < Xo.size(); q++) if (!same_bits_nanaware(Xo[q], Xa[q])) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: opt/asm differ\n", what, m, nrhs); return; }
    double anorm = 0; for (int i = 0; i < m; i++) { double r = 0; for (int j = 0; j < m; j++) r += std::fabs(s.A[i + (size_t)j * s.lda]); anorm = std::max(anorm, r); }
    auto resid = [&](const std::vector<double> &X) {
        double worst = 0;
        for (int r = 0; r < nrhs; r++) {
            double err = 0, xn = 0, bn = 0;
            for (int i = 0; i < m; i++) { xn = std::max(xn, std::fabs(X[i + (size_t)r * ldb])); bn = std::max(bn, std::fabs(B[i + (size_t)r * ldb])); }
            for (int i = 0; i < m; i++) { double a = 0; for (int j = 0; j < m; j++) a += s.A[i + (size_t)j * s.lda] * X[j + (size_t)r * ldb]; err = std::max(err, std::fabs(a - B[i + (size_t)r * ldb])); }
            worst = std::max(worst, err / (anorm * xn + bn));
        }
        return worst;
    };
    const double rn = resid(Xn), ro = resid(Xo);
    if (!(ro <= 4.0 * rn + 4.0 * 2.2e-16 * m)) { fails++; std::printf("FAIL %-28s m=%d nrhs=%d: backward error naive %.2e opt %.2e\n", what, m, nrhs, rn, ro); }
}

/* factor with reclu_opt_rt (padded) or the reference (unpadded lda = m) */
static Sys factor(const std::vector<double> &Am, int m, bool padded) {
    Sys s; s.m = m; s.lda = padded ? ((m + 3) & ~3) : m;
    s.A.assign((size_t)s.lda * m, 0.0);
    for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) s.A[i + (size_t)j * s.lda] = Am[i + (size_t)j * m];
    s.LU = s.A; s.ipiv.resize(m);
    if (padded) reclu_lu_opt(m, m, s.LU.data(), s.lda, s.ipiv.data(), &s.info);
    else        reclu_lu(m, m, s.LU.data(), s.lda, s.ipiv.data(), &s.info);
    return s;
}

static std::vector<double> rand_mat(int m, double diag_add) {
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> A((size_t)m * m); for (auto &a : A) a = U(rng);
    for (int i = 0; i < m; i++) A[i + (size_t)i * m] += diag_add;
    return A;
}
static std::vector<double> rand_B(int m, int nrhs, int ldb) {
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> B((size_t)ldb * nrhs); for (auto &b : B) b = U(rng);
    return B;
}

int main() {
    const int sizes[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 100};
    const int nrhss[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13};

    for (int m : sizes) for (int nrhs : nrhss) for (int ldb_extra : {0, 1, 5}) {
        const int ldb = m + ldb_extra;
        /* 1. random well-conditioned, padded LU */
        { Sys s = factor(rand_mat(m, 2.0 * m), m, true); check("random wellcond", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 2. same, unpadded lda from the reference LU -> fallback path */
        { Sys s = factor(rand_mat(m, 2.0 * m), m, false); check("unpadded lda (fallback)", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 3. identity pivoting: strongly diagonally dominant */
        { auto A = rand_mat(m, 0.0); for (int i = 0; i < m; i++) A[i + (size_t)i * m] = 100.0 * m; Sys s = factor(A, m, true);
          check("identity pivots", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 4. reversed pivoting: anti-diagonal dominant */
        { auto A = rand_mat(m, 0.0); for (int i = 0; i < m; i++) A[(m - 1 - i) + (size_t)i * m] = 100.0 * m; Sys s = factor(A, m, true);
          check("reversed pivots", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 5. random dense pivoting (no dominance) */
        { Sys s = factor(rand_mat(m, 0.0), m, true); check("random pivots", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 6. singular: one zero column -> zero pivot, info > 0 -> Inf/NaN propagation must match */
        if (m >= 2) { auto A = rand_mat(m, 2.0 * m); const int jz = m / 2; for (int i = 0; i < m; i++) A[i + (size_t)jz * m] = 0.0;
          Sys s = factor(A, m, true); if (s.info <= 0) { std::printf("test bug: singular case has info=%d\n", s.info); }
          check("singular (zero pivot)", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 7. ill-conditioned: Hilbert-like, cond ~ 1e16 for m >= 12 */
        { std::vector<double> A((size_t)m * m); for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * m] = 1.0 / (i + j + 1);
          Sys s = factor(A, m, true); check_backward("Hilbert (ill-cond)", s, nrhs, rand_B(m, nrhs, ldb), ldb); }
        /* 8. tiny pivots: diagonal scaled to 1e-12, others O(1) -> huge multipliers */
        { auto A = rand_mat(m, 0.0); for (int i = 0; i < m; i++) A[i + (size_t)i * m] = 1e-12 * (1 + (i % 3)); Sys s = factor(A, m, true);
          check("tiny diagonal", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb);
          check_backward("tiny diagonal (backward)", s, nrhs, rand_B(m, nrhs, ldb), ldb); }
        /* 7b. Hilbert forward comparison only where kappa*eps << 1 */
        if (m <= 8) { std::vector<double> A((size_t)m * m); for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * m] = 1.0 / (i + j + 1);
          Sys s = factor(A, m, true); check("Hilbert small (forward)", m, nrhs, s.LU, s.lda, s.ipiv, rand_B(m, nrhs, ldb), ldb); }
        /* 9. overflow: huge B with a moderately ill-conditioned A -> some Inf in x */
        { Sys s = factor(rand_mat(m, 0.0), m, true); auto B = rand_B(m, nrhs, ldb); for (auto &b : B) b *= 1e307;
          check("huge B (overflow)", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 10. denormal B */
        { Sys s = factor(rand_mat(m, 2.0 * m), m, true); auto B = rand_B(m, nrhs, ldb); for (auto &b : B) b *= 1e-310;
          check("denormal B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 11. NaN in B: one entry per rhs; every dependent x must be NaN in both */
        if (nrhs > 0) { Sys s = factor(rand_mat(m, 2.0 * m), m, true); auto B = rand_B(m, nrhs, ldb); for (int r = 0; r < nrhs; r++) B[(r % m) + (size_t)r * ldb] = NAN;
          check("NaN in B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 12. Inf in B */
        if (nrhs > 0) { Sys s = factor(rand_mat(m, 2.0 * m), m, true); auto B = rand_B(m, nrhs, ldb); for (int r = 0; r < nrhs; r++) B[((r * 7) % m) + (size_t)r * ldb] = (r & 1) ? INFINITY : -INFINITY;
          check("Inf in B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 13. all-zero and negative-zero B */
        { Sys s = factor(rand_mat(m, 2.0 * m), m, true); std::vector<double> B((size_t)ldb * nrhs, 0.0); check("zero B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb);
          for (auto &b : B) b = -0.0; check("negative-zero B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 14. B with garbage in the pad rows (rows m..ldb-1) -> must be preserved */
        if (ldb > m) { Sys s = factor(rand_mat(m, 2.0 * m), m, true); auto B = rand_B(m, nrhs, ldb); for (int r = 0; r < nrhs; r++) for (int i = m; i < ldb; i++) B[i + (size_t)r * ldb] = NAN;
          check("NaN in pad rows of B", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
        /* 15. integer matrix: exact ties in pivoting, exact arithmetic paths */
        { std::uniform_int_distribution<int> I(-3, 3); std::vector<double> A((size_t)m * m); for (auto &a : A) a = I(rng); for (int i = 0; i < m; i++) A[i + (size_t)i * m] += 8;
          Sys s = factor(A, m, true); auto B = rand_B(m, nrhs, ldb); for (auto &b : B) b = std::round(b * 4); check("small integers", m, nrhs, s.LU, s.lda, s.ipiv, B, ldb); }
    }
    /* 16. aliasing-free check: lda larger than roundup4(m) (extra pad rows) */
    for (int m : {5, 30, 64}) { const int lda = ((m + 3) & ~3) + 8; std::vector<double> A((size_t)lda * m, 0.0); std::uniform_real_distribution<double> U(-1, 1);
        for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) A[i + (size_t)j * lda] = U(rng) + (i == j ? 2.0 * m : 0.0);
        std::vector<double> LU = A; std::vector<int> ipiv(m); int info; reclu_lu_opt(m, m, LU.data(), lda, ipiv.data(), &info);
        for (int nrhs : {1, 4, 5, 8}) check("lda = roundup4 + 8", m, nrhs, LU, lda, ipiv, rand_B(m, nrhs, m), m); }

    std::printf("%d cases, %d failures -> %s\n", tests, fails, fails ? "FAIL" : "PASS");
    return fails != 0;
}
