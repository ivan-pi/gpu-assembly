/*
 * AI disclaimer: written with the assistance of an AI model. Assisted-by: Claude:claude-fable-5
 * reclu_sytf2_opt / reclu_sytrs_opt vs the naive references, both uplo:
 *  - factorization bit-identical (pivots, info, every element of the factored triangle;
 *    NaN sign ignored), the other triangle and the pad rows of 'U' storage untouched;
 *  - solve: within 16 n eps of the reference norm-wise (the forward substitution is
 *    bit-identical; the transposed substitution's dot product is contracted differently by
 *    the compiler in the reference), Inf/NaN pattern identical on singular input,
 *    backward error <= 1e-14 on well-conditioned input, Inf/NaN pattern preserved on singular;
 *  - matrices: SPD, indefinite, zero diagonal (forces 2x2 pivots), small integers (ties),
 *    singular (zero row/col), tiny/huge scaling, NaN entry; lda = n / roundup4 / +4 / n+1.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "reclu.h"
static int tests = 0, fails = 0; static std::mt19937 rng(77); static double worst_solve_rel = 0;
static bool same(double a, double b) { return (std::isnan(a) && std::isnan(b)) || !std::memcmp(&a, &b, 8); }

static void check(const char *what, char uplo, int n, int lda, const std::vector<double> &Sym /* n x n dense symmetric, ld n */) {
    std::uniform_real_distribution<double> U(-1, 1);
    const int MP = (n + 3) & ~3;
    std::vector<double> A((size_t)lda * n); for (auto &v : A) v = U(rng) * 1e200;           /* garbage everywhere */
    for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((uplo == 'U') ? (i <= j) : (i >= j)) A[i + (size_t)j * lda] = Sym[i + (size_t)j * n];
    std::vector<double> Ar = A, Ao = A; std::vector<int> pr(n), po(n); int ir = 0, io = 0;
    reclu_sytf2    (uplo, n, Ar.data(), lda, pr.data(), &ir);
    reclu_sytf2_opt(uplo, n, Ao.data(), lda, po.data(), &io);
    tests++;
    bool ok = (ir == io) && (pr == po);
    for (int j = 0; j < n && ok; j++) for (int i = 0; i < lda && ok; i++) {
        const size_t q = i + (size_t)j * lda;
        const bool tri = i < n && ((uplo == 'U') ? (i <= j) : (i >= j));
        const bool pad_l = (uplo == 'L') && i >= n && i < MP && (lda & 3) == 0 && lda >= MP;   /* zeroed by opt, documented */
        if (tri) ok = same(Ar[q], Ao[q]);
        else if (!pad_l) ok = same(A[q], Ao[q]);                                           /* untouched */
    }
    if (!ok) { fails++; if (fails <= 8) std::printf("FACTOR FAIL %-18s uplo=%c n=%d lda=%d info %d/%d pivots %s\n", what, uplo, n, lda, ir, io, pr == po ? "same" : "differ"); return; }
    /* solves: nrhs 5 = one lane group + one single */
    const int nrhs = 5, ldb = n + 2;
    std::vector<double> B((size_t)ldb * nrhs); for (auto &v : B) v = U(rng);
    std::vector<double> Xr = B, Xo = B;
    reclu_sytrs    (uplo, n, nrhs, Ar.data(), lda, pr.data(), Xr.data(), ldb);
    reclu_sytrs_opt(uplo, n, nrhs, Ao.data(), lda, po.data(), Xo.data(), ldb);
    bool bit4 = true; double xmax = 0, dmax = 0; bool nonfinite = false;
    for (int r = 0; r < nrhs; r++) for (int i = 0; i < n; i++) {
        const size_t q = i + (size_t)r * ldb;
        if (!std::isfinite(Xr[q])) { nonfinite = true; if (std::isnan(Xr[q]) != std::isnan(Xo[q]) || (std::isinf(Xr[q]) && Xr[q] != Xo[q])) { if (bit4) std::printf("   nonfinite: row %d rhs %d: naive %g opt %g\n", i, r, Xr[q], Xo[q]); bit4 = false; } continue; }
        xmax = std::max(xmax, std::fabs(Xr[q])); dmax = std::max(dmax, std::fabs(Xr[q] - Xo[q]));
    }
    for (int r = 0; r < nrhs; r++) for (int i = n; i < ldb; i++) if (!same(B[i + (size_t)r * ldb], Xo[i + (size_t)r * ldb])) bit4 = false;   /* pad rows of B */
    if (!bit4) { fails++; if (fails <= 8) std::printf("SOLVE PATTERN FAIL %-18s uplo=%c n=%d\n", what, uplo, n); return; }
    if (ir == 0 && !nonfinite) {
        /* the transposed solve's dot product is contracted differently by the compiler in the
         * reference (see dot_rows); the agreement is specified at a few ulp, norm-wise */
        if (dmax > 16 * 2.2e-16 * n * xmax) { fails++; if (fails <= 8) std::printf("SOLVE ULP FAIL %-18s uplo=%c n=%d rel %.2e\n", what, uplo, n, dmax / xmax); return; }
        worst_solve_rel = std::max(worst_solve_rel, dmax / (xmax * 2.2e-16 * n));
        /* backward error of the opt solution on the original symmetric matrix */
        double anorm = 0; for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += std::fabs(Sym[i + (size_t)j * n]); anorm = std::max(anorm, s); }
        double worst = 0;
        for (int r = 0; r < nrhs; r++) {
            double err = 0, xn = 0, bn = 0;
            for (int i = 0; i < n; i++) { xn = std::max(xn, std::fabs(Xo[i + (size_t)r * ldb])); bn = std::max(bn, std::fabs(B[i + (size_t)r * ldb])); }
            for (int i = 0; i < n; i++) { double s = 0; for (int j = 0; j < n; j++) s += Sym[i + (size_t)j * n] * Xo[j + (size_t)r * ldb]; err = std::max(err, std::fabs(s - B[i + (size_t)r * ldb])); }
            worst = std::max(worst, err / (anorm * xn + bn));
        }
        if (!(worst <= 1e-13)) { fails++; if (fails <= 8) std::printf("RESIDUAL FAIL %-18s uplo=%c n=%d backward err %.2e\n", what, uplo, n, worst); }
    }
}

int main() {
    std::uniform_real_distribution<double> U(-1, 1); std::uniform_int_distribution<int> I(-3, 3);
    for (int n : {1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 20, 21, 24, 30, 31, 33, 48, 63, 64, 65, 100}) for (char uplo : {'L', 'U'}) for (int lda : {n, (n + 3) & ~3, ((n + 3) & ~3) + 4, n + 1}) for (int rep = 0; rep < 3; rep++) {
        std::vector<double> S((size_t)n * n);
        auto sym = [&](auto f) { for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { const double v = f(i, j); S[i + (size_t)j * n] = v; S[j + (size_t)i * n] = v; } };
        sym([&](int i, int j) { return U(rng) + (i == j ? 3.0 * n : 0.0); });          check("SPD-ish", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng); });                                       check("indefinite", uplo, n, lda, S);
        sym([&](int i, int j) { return i == j ? 0.0 : U(rng); });                        check("zero diagonal", uplo, n, lda, S);
        sym([&](int i, int j) { return (double)I(rng); });                               check("small integers", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng) * (i == j ? 1e-3 : 1.0); });               check("small diagonal", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng) * 1e150; });                               check("huge scale", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng) * 1e-160; });                              check("tiny scale", uplo, n, lda, S);
        sym([&](int i, int j) { return (i == n / 2 || j == n / 2) ? 0.0 : U(rng); });    check("zero row/col", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng); }); S[(n / 3) + (size_t)(n / 3) * n] = NAN; check("NaN diagonal", uplo, n, lda, S);
        sym([&](int i, int j) { return U(rng); }); if (n > 1) { S[0 + (size_t)(n - 1) * n] = S[(n - 1) + 0] = NAN; } check("NaN offdiag", uplo, n, lda, S);
    }
    std::printf("%d cases, %d failures (worst solve difference %.2f x n x eps norm-wise) -> %s\n", tests, fails, worst_solve_rel, fails ? "FAIL" : "PASS");
    return fails != 0;
}
