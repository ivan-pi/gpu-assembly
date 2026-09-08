/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
/* reclu_lu_opt vs the reference reclu_lu: bit-identical factors and pivots
 * (two NaNs compare equal regardless of sign) over adversarial inputs:
 * uniform, small-integer (ties, exact zeros), sparse, denormal, NaN entry,
 * zero column (singular, info > 0), symmetric saddle-point; square and
 * rectangular shapes; sizes 1..100 including every awkward one; lda equal to
 * m, roundup4(m), roundup4(m)+4/+8 (in-place paths, garbage in the pad rows)
 * and m+1 (copy path). Rows 0..m-1 must match; pad rows are not compared. */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "reclu.h"

static int tests = 0, fails = 0;
static std::mt19937 rng(12345);

static void check(int m, int n, int lda, const std::vector<double> &Am, const char *what) {
    /* Am is dense m x n (ld m); embed in an lda-stride buffer with garbage pad rows */
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> A((size_t)lda * n); for (auto &a : A) a = U(rng) * 1e30;   /* garbage */
    for (int j = 0; j < n; j++) std::memcpy(&A[(size_t)j * lda], &Am[(size_t)j * m], m * 8);
    std::vector<double> Ar = A, Ao = A;
    const int K = m < n ? m : n;
    std::vector<int> pr(K), po(K); int ir, io;
    reclu_lu    (m, n, Ar.data(), lda, pr.data(), &ir);
    reclu_lu_opt(m, n, Ao.data(), lda, po.data(), &io);
    tests++;
    bool ok = ir == io && pr == po;
    for (int j = 0; j < n && ok; j++) for (int i = 0; i < m && ok; i++) {
        const double a = Ar[i + (size_t)j * lda], b = Ao[i + (size_t)j * lda];
        if (std::isnan(a) && std::isnan(b)) continue;
        ok = std::memcmp(&a, &b, 8) == 0;
    }
    if (!ok) {
        fails++;
        if (fails <= 8) std::printf("FAIL %-24s m=%d n=%d lda=%d info %d/%d pivots %s\n", what, m, n, lda, ir, io, pr == po ? "same" : "differ");
    }
}

static void suite(int m, int n) {
    std::uniform_real_distribution<double> U(-1, 1);
    std::uniform_int_distribution<int> I(-3, 3);
    const int mp = (m + 3) & ~3;
    std::vector<double> A((size_t)m * n);
    for (int rep = 0; rep < 12; rep++) for (int lda : {m, mp, mp + 4, mp + 8, m + 1}) {
        for (auto &a : A) a = U(rng);                       check(m, n, lda, A, "uniform");
        for (auto &a : A) a = I(rng);                       check(m, n, lda, A, "small ints (ties/zeros)");
        for (auto &a : A) a = (I(rng) == 0) ? 0.0 : U(rng); check(m, n, lda, A, "sparse");
        for (auto &a : A) a = U(rng) * 1e-310;              check(m, n, lda, A, "denormal");
        for (auto &a : A) a = U(rng);
        A[(rep % n) * m + (rep % m)] = NAN;                 check(m, n, lda, A, "NaN entry");
        for (auto &a : A) a = U(rng);
        for (int i = 0; i < m; i++) A[i + (rep % n) * m] = 0.0; check(m, n, lda, A, "zero column (singular)");
        if (m == n) {
            for (int j = 0; j < n; j++) for (int i = 0; i <= j; i++) { double v = U(rng); A[i + j * m] = v; A[j + i * m] = v; }
            const int nb = n / 5; for (int j = n - nb; j < n; j++) for (int i = n - nb; i < n; i++) A[i + j * m] = 0.0;
            check(m, n, lda, A, "symmetric saddle-point");
        }
    }
}

int main() {
    for (int m : {1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 16, 17, 19, 20, 21, 24, 30, 31, 32, 33, 48, 63, 64, 65, 100}) suite(m, m);
    suite(10, 6); suite(6, 10); suite(33, 12); suite(12, 33); suite(64, 3); suite(3, 64); suite(25, 40); suite(40, 25);
    std::printf("%d cases, %d failures -> %s\n", tests, fails, fails ? "FAIL" : "ALL BIT-IDENTICAL");
    return fails != 0;
}
