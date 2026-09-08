/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#include "reclu.h"

#if !(defined(__AVX2__) && defined(__FMA__))
void reclu_lu_opt(int m, int n, double *A, int lda, int *ipiv, int *info) { reclu_lu(m, n, A, lda, ipiv, info); }
#else

#include "reclu_simd.h"
#include <cmath>
#include <cstring>
#include <limits>

/* Optimized dgetf2 (right-looking rank-1 LU, partial pivoting), AVX2/FMA.
 *
 * The arithmetic is exactly the reference's: pivot = first row with the
 * largest |a|, scale by the reciprocal (or divide for a tiny pivot), one
 * fused multiply-subtract per trailing element. Results are bit-identical
 * to reclu_lu. What differs is scheduling:
 *
 *  - rows are padded to a multiple of 4 with zeros (never selected as pivot,
 *    scale to +-0, only ever update themselves), so every vector access is
 *    a full unmasked load/store -- masked and scalar tails do not
 *    store-forward to the next step's vector loads and were the dominant
 *    cost at small n;
 *  - the trailing update is an 8-row x 4-column register tile; the 4-row
 *    group containing row k+1 is blended so rows <= k are stored unchanged;
 *  - the pivot search for column k+1 is fused into the update of that
 *    column (per-lane running max, exact LAPACK tie-breaking), and the next
 *    pivot's reciprocal is issued before the remaining tiles so the serial
 *    per-step chain overlaps this step's bulk work. */

namespace {

constexpr double kSfmin = std::numeric_limits<double>::min();

/* Running "first index of the largest |a|" over vectors of 4 rows. Lane l
 * only ever sees rows == first_row + l (mod 4) in increasing order, so a
 * strict '>' per lane plus "larger wins, ties -> smaller index" across lanes
 * reproduces the scalar scan. Seeded with 0 at diag_row: any non-zero |a|
 * beats it, an all-zero column yields diag_row, NaN never wins. */
struct PivotTracker {
    __m256d vmax, vidx, cur;
    void init(int diag_row, int first_row) {
        vmax = _mm256_setzero_pd();
        vidx = _mm256_set1_pd(diag_row);
        cur  = _mm256_setr_pd(first_row, first_row + 1, first_row + 2, first_row + 3);
    }
    void feed(__m256d v) {
        const __m256d a  = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);           /* |v| */
        const __m256d gt = _mm256_cmp_pd(a, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_pd(vmax, a, gt);
        vidx = _mm256_blendv_pd(vidx, cur, gt);
        cur  = _mm256_add_pd(cur, _mm256_set1_pd(4.0));
    }
    /* Deliberately branchy: the predictor usually guesses right and lets the
     * next step start early, which beats a longer branchless chain. */
    int reduce() const {
        alignas(32) double m[4], ix[4];
        _mm256_store_pd(m, vmax); _mm256_store_pd(ix, vidx);
        double best = m[0], bi = ix[0];
        for (int l = 1; l < 4; l++)
            if (m[l] > best || (m[l] == best && ix[l] < bi)) { best = m[l]; bi = ix[l]; }
        return (int)bi;
    }
};

/* Rank-1 update of W columns starting at col[0], rows g0..MP-1, by the pivot
 * column: c[i] -= pivcol[i] * c[k]. The first row group is blended with the
 * live mask (rows <= k keep their U values). If track, the first column is
 * fed to the pivot tracker. W = 4 is the register tile, W = 1 the remainder. */
template <int W, bool TRACK>
inline __attribute__((always_inline)) void
update_cols(double *const col[W], const double *__restrict piv, int k, int g0, int MP, PivotTracker *pt) {
    const __m256d live = reclu_live_mask(k + 1);
    __m256d b[W];
    for (int w = 0; w < W; w++) b[w] = _mm256_broadcast_sd(col[w] + k);
    if (TRACK) pt->init(k + 1, g0);

    auto step = [&](int i, bool blend) {                     /* one 4-row group */
        const __m256d p = _mm256_loadu_pd(piv + i);
        for (int w = 0; w < W; w++) {
            __m256d x = _mm256_loadu_pd(col[w] + i);
            RECLU_PIN(x);
            __m256d y = _mm256_fnmadd_pd(p, b[w], x);
            if (blend) y = _mm256_blendv_pd(x, y, live);
            _mm256_storeu_pd(col[w] + i, y);
            if (TRACK && w == 0) pt->feed(blend ? _mm256_and_pd(y, live) : y);
        }
    };
    step(g0, true);
    int i = g0 + 4;
    #pragma GCC unroll 1
    for (; i + 8 <= MP; i += 8) { step(i, false); step(i + 4, false); }
    if (i < MP) step(i, false);
}

/* Factor the first minmn columns of an MP x n matrix (MP % 4 == 0, rows >= m
 * are zero), leading dimension lda. */
void factor(int MP, int n, int minmn, double *__restrict A, int lda, int *__restrict ipiv, int *__restrict info) {
    auto col = [&](int j) { return A + (size_t)j * lda; };

    /* column 0: plain scan (the reference's loop) */
    int kp_next = 0; double pmax = std::fabs(col(0)[0]);
    for (int i = 1; i < MP; i++) { const double a = std::fabs(col(0)[i]); if (a > pmax) { pmax = a; kp_next = i; } }
    double akk_next = col(0)[kp_next], inv_next = 1.0 / akk_next;

    for (int k = 0; k < minmn; k++) {
        const int kp = kp_next; const double akk = akk_next;
        ipiv[k] = kp + 1;
        double *ck = col(k);
        const int g0 = (k + 1) & ~3;                          /* group holding row k+1 */

        if (akk != 0.0) {
            if (kp != k)
                for (int j = 0; j < n; j++) { double *c = col(j); const double t = c[k]; c[k] = c[kp]; c[kp] = t; }
            if (k + 1 < MP) {
                if (std::fabs(akk) >= kSfmin) {
                    const __m256d inv = _mm256_set1_pd(inv_next), live = reclu_live_mask(k + 1);
                    const __m256d o = _mm256_loadu_pd(ck + g0);
                    _mm256_storeu_pd(ck + g0, _mm256_blendv_pd(o, _mm256_mul_pd(o, inv), live));
                    #pragma GCC unroll 1
                    for (int i = g0 + 4; i < MP; i += 4) _mm256_storeu_pd(ck + i, _mm256_mul_pd(_mm256_loadu_pd(ck + i), inv));
                } else {
                    for (int i = k + 1; i < MP; i++) ck[i] /= akk;
                }
            }
        } else if (*info == 0) {
            *info = k + 1;
        }
        if (k + 1 >= minmn) break;

        /* trailing update; the first tile also finds the next pivot */
        PivotTracker pt;
        int j = k + 1;
        if (j + 4 <= n) { double *c[4] = {col(j), col(j + 1), col(j + 2), col(j + 3)}; update_cols<4, true>(c, ck, k, g0, MP, &pt); j += 4; }
        else            { double *c[1] = {col(j)}; update_cols<1, true>(c, ck, k, g0, MP, &pt); j += 1; }
        {
            const double *c1 = col(k + 1);
            int kpn = pt.reduce();
            if (!(std::fabs(c1[kpn]) > std::fabs(c1[k + 1]))) kpn = k + 1;   /* diagonal wins ties; NaN diag -> k+1 */
            kp_next = kpn; akk_next = c1[kpn]; inv_next = 1.0 / akk_next;    /* divide starts now */
        }
        for (; j + 4 <= n; j += 4) { double *c[4] = {col(j), col(j + 1), col(j + 2), col(j + 3)}; update_cols<4, false>(c, ck, k, g0, MP, nullptr); }
        for (; j < n; j++)         { double *c[1] = {col(j)}; update_cols<1, false>(c, ck, k, g0, MP, nullptr); }
    }
}

/* Own frame for the scratch buffer: a 128 KB array in the entry function
 * would be stack-probed on every call, including the small-size path. */
__attribute__((noinline)) void factor_copy(int m, int n, int minmn, double *A, int lda, int *ipiv, int *info) {
    const int MP = (m + 3) & ~3;
    alignas(32) double buf[RECLU_MAXM * RECLU_MAXM + 4];
    for (int j = 0; j < n; j++) {
        std::memcpy(buf + (size_t)j * MP, A + (size_t)j * lda, m * sizeof(double));
        for (int i = m; i < MP; i++) buf[(size_t)j * MP + i] = 0.0;
    }
    factor(MP, n, minmn, buf, MP, ipiv, info);
    for (int j = 0; j < n; j++) std::memcpy(A + (size_t)j * lda, buf + (size_t)j * MP, m * sizeof(double));
}

} // namespace

void reclu_lu_opt(int m, int n, double *A, int lda, int *ipiv, int *info) {
    *info = 0;
    if (m < 0) { *info = -1; return; }
    if (n < 0) { *info = -2; return; }
    if (lda < (m < 1 ? 1 : m)) { *info = -4; return; }
    if (m == 0 || n == 0) return;
    if (m < RECLU_SMALL || m > RECLU_MAXM) { reclu_lu(m, n, A, lda, ipiv, info); return; }

    const int MP = (m + 3) & ~3, minmn = m < n ? m : n;
    if ((lda & 3) == 0 && lda >= MP) {                        /* in place; pad rows are ours */
        for (int j = 0; j < n; j++) for (int i = m; i < MP; i++) A[(size_t)j * lda + i] = 0.0;
        factor(MP, n, minmn, A, lda, ipiv, info);
    } else {
        factor_copy(m, n, minmn, A, lda, ipiv, info);
    }
}
#endif
