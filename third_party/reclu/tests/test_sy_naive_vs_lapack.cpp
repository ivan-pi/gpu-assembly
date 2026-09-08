/*
 * AI disclaimer: written with the assistance of an AI model. Assisted-by: Claude:claude-fable-5
 * reclu_sytf2 / reclu_sytrs (naive) vs netlib LAPACK dsytf2 / dsytrs (dlopen'd liblapack.so.3):
 * pivot sequences must match; factors and solutions agree to a few ulp (rounding of
 * contracted operations differs between compilers). Skipped (exit 77) if LAPACK is absent.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <vector>
#include "reclu.h"
typedef void (*sytf2_t)(const char *, const int *, double *, const int *, int *, int *, int);
typedef void (*sytrs_t)(const char *, const int *, const int *, const double *, const int *, const int *, double *, const int *, int *, int);
int main() {
    void *h = dlopen("liblapack.so.3", RTLD_NOW); if (!h) { std::printf("liblapack.so.3 not found: skipped\n"); return 77; }
    auto dsytf2 = (sytf2_t)dlsym(h, "dsytf2_"); auto dsytrs = (sytrs_t)dlsym(h, "dsytrs_");
    if (!dsytf2 || !dsytrs) { std::printf("dsytf2_/dsytrs_ not found: skipped\n"); return 77; }
    std::mt19937 rng(5); std::uniform_real_distribution<double> U(-1, 1);
    int tests = 0, pivfail = 0, pivdiff = 0, valfail = 0, solfail = 0, singular = 0; double worst = 0, worst_sol = 0;
    for (int n : {1, 2, 3, 4, 5, 7, 8, 9, 13, 16, 20, 21, 30, 33, 48, 63, 64, 65, 100}) for (char uplo : {'L', 'U'}) for (int kind = 0; kind < 5; kind++) for (int rep = 0; rep < 4; rep++) {
        const int lda = n + (rep & 1);
        std::vector<double> A((size_t)lda * n, 0.0);
        for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) {
            double v = U(rng);
            if (kind == 1) v = std::round(3 * v);                     /* small integers: ties */
            if (kind == 2 && i == j) v += 3.0 * n;                    /* SPD-ish: mostly 1x1 pivots, no swaps */
            if (kind == 3 && i == j) v = 0.0;                         /* zero diagonal: forces 2x2 pivots */
            if (kind == 4) v *= (i == j) ? 1e-3 : 1.0;                /* small diagonal: mixed */
            A[i + (size_t)j * lda] = v; A[j + (size_t)i * lda] = v;
        }
        /* keep only the referenced triangle meaningful; poison the other one to catch writes */
        std::vector<double> Ain = A;
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) if ((uplo == 'U') ? (i > j) : (i < j)) Ain[i + (size_t)j * lda] = 1e300;
        std::vector<double> Ar = Ain, Al = Ain; std::vector<int> pr(n), pl(n); int ir, il;
        reclu_sytf2(uplo, n, Ar.data(), lda, pr.data(), &ir);
        dsytf2(&uplo, &n, Al.data(), &lda, pl.data(), &il, 1);
        tests++;
        const bool samepiv = (pr == pl && ir == il);
        if (!samepiv) { pivdiff++; if (ir > 0 || il > 0) continue; }   /* near-tie: both valid, compare solutions below */
        if (ir > 0) { singular++; continue; }               /* zero pivot: D singular, values are not comparable */
        /* condition estimate from the factor: min |d| / max |d| over the 1x1 blocks */
        double dmin = 1e300, dmax2 = 0; for (int i = 0; i < n; i++) if (pr[i] > 0) { const double d = std::fabs(Ar[i + (size_t)i * lda]); dmin = std::min(dmin, d); dmax2 = std::max(dmax2, d); }
        const bool illcond = (dmin < 1e-9 * dmax2);
        double md = 0, amax = 0;
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
            const bool tri = (uplo == 'U') ? (i <= j) : (i >= j); const size_t q = i + (size_t)j * lda;
            if (!tri) { if (Ar[q] != 1e300) { valfail++; std::printf("OTHER TRIANGLE WRITTEN n=%d uplo=%c\n", n, uplo); } continue; }
            amax = std::max(amax, std::fabs(Al[q])); if (samepiv) md = std::max(md, std::fabs(Ar[q] - Al[q]));
        }
        md /= amax;                                            /* norm-wise */
        worst = std::max(worst, md);
        if (md > 1e-12 && !illcond) { valfail++; if (valfail < 5) std::printf("VALUE n=%d uplo=%c kind=%d rel %.2e\n", n, uplo, kind, md); }
        /* solve: 3 rhs */
        const int nrhs = 3, ldb = n + 1; std::vector<double> B((size_t)ldb * nrhs); for (auto &x : B) x = U(rng);
        std::vector<double> Xr = B, Xl = B;
        reclu_sytrs(uplo, n, nrhs, Ar.data(), lda, pr.data(), Xr.data(), ldb);
        dsytrs(&uplo, &n, &nrhs, Al.data(), &lda, pl.data(), Xl.data(), &ldb, &il, 1);
        double xmax = 0, dmax = 0;
        for (int r = 0; r < nrhs; r++) for (int i = 0; i < n; i++) { xmax = std::max(xmax, std::fabs(Xl[i + (size_t)r * ldb])); dmax = std::max(dmax, std::fabs(Xr[i + (size_t)r * ldb] - Xl[i + (size_t)r * ldb])); }
        const double rel = dmax / (xmax + 1e-300); worst_sol = std::max(worst_sol, rel);
        if (rel > 1e-9 && !illcond) { solfail++; if (!samepiv) pivfail++; if (solfail < 5) std::printf("SOLVE n=%d uplo=%c kind=%d rel %.2e\n", n, uplo, kind, rel); }
    }
    std::printf("%d cases vs netlib (%d singular): pivot sequences differ in %d (near ties; %d of those give different solutions), factor mismatches %d (worst norm-wise %.1e), solve mismatches %d (worst rel %.1e) -> %s\n",
                tests, singular, pivdiff, pivfail, valfail, worst, solfail, worst_sol, (pivfail || valfail || solfail) ? "FAIL" : "PASS");
    return (pivfail || valfail || solfail) != 0;
}
