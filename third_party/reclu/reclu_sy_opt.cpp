/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code was reviewed and validated by tests against the reference kernels
 * (bit-identical factorization), but AI-generated material can contain
 * subtle errors; verify before relying on it.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#include "reclu.h"

#if !(defined(__AVX2__) && defined(__FMA__))
void reclu_sytf2_opt(char uplo, int n, double *A, int lda, int *ipiv, int *info) { reclu_sytf2(uplo, n, A, lda, ipiv, info); }
void reclu_sytrs_opt(char uplo, int n, int nrhs, const double *A, int lda, const int *ipiv, double *B, int ldb) { reclu_sytrs(uplo, n, nrhs, A, lda, ipiv, B, ldb); }
#else

#include "reclu_simd.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

/* Optimized Bunch-Kaufman LDL^T (dsytf2) and solve (dsytrs), AVX2/FMA.
 *
 * Same operation order as reclu_sytf2, so the factorization is bit-identical
 * to the reference. The techniques are the ones that paid off for the LU:
 *
 *  - padded, aligned storage (lda % 4 == 0, lda >= roundup4(n)); for 'L'
 *    the pad rows are zeroed (they never win the pivot search and only
 *    update themselves), for 'U' they are only read into dead lanes;
 *  - column streaming in 4-column tiles: the pivot column(s) are loaded once
 *    per row group and reused for 4 columns; the triangle boundary
 *    (rows < j for 'L', rows > j for 'U') is handled by blending the boundary
 *    row groups and storing full width, so the other triangle is never
 *    modified and every store forwards to the next step's loads;
 *  - the column-max search for the next pivot (idamax over the next column)
 *    is fused into the first tile of the update, with first-index semantics;
 *  - 'U' is the mirror image of 'L' with rows [0, j] instead of [j, n): the
 *    same tile code with different row ranges, so both are equally fast.
 *
 * Solve: per-pivot streaming updates for the forward substitution (bit-identical
 * to the reference); the transposed substitution is a dot product per pivot in
 * the reference's 4-way partial-sum order with fused terms. Whether the
 * reference's own terms are fused depends on the compiler (GCC 13 does not
 * contract them when it packs the four chains into a vector), so the solve
 * agrees with the reference to a few ulp rather than bit for bit. */

#ifdef RECLU_SY_PROFILE
#include <x86intrin.h>
unsigned long long reclu_sy_prof[4];   /* search+swap, update, scale, other */
#define PROF_T0 unsigned long long _t0 = __rdtsc()
#define PROF_ADD(i) do { unsigned long long _t1 = __rdtsc(); reclu_sy_prof[i] += _t1 - _t0; _t0 = _t1; } while (0)
#else
#define PROF_T0
#define PROF_ADD(i)
#endif
namespace {
constexpr double ALPHA = 0.6403882032022076;    /* (1 + sqrt(17)) / 8 */


/* first index of the max |a| over rows fed in increasing order, 4 at a time
 * (dead lanes zeroed by the caller) */
struct MaxTracker {
    __m256d vmax, vidx, cur;
    void init(int first_row, int seed_row) { vmax = _mm256_setzero_pd(); vidx = _mm256_set1_pd(seed_row); cur = _mm256_setr_pd(first_row, first_row + 1, first_row + 2, first_row + 3); }
    void feed(__m256d v) {
        const __m256d a = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v), gt = _mm256_cmp_pd(a, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_pd(vmax, a, gt); vidx = _mm256_blendv_pd(vidx, cur, gt); cur = _mm256_add_pd(cur, _mm256_set1_pd(4.0));
    }
    int reduce() const {
        alignas(32) double m[4], ix[4]; _mm256_store_pd(m, vmax); _mm256_store_pd(ix, vidx);
        double best = m[0], bi = ix[0];
        for (int l = 1; l < 4; l++) if (m[l] > best || (m[l] == best && ix[l] < bi)) { best = m[l]; bi = ix[l]; }
        return (int)bi;
    }
};

/* Lane masks: GE[a] = lanes >= a, LT[b] = lanes < b (a, b in 0..4). */
alignas(32) static const long long MASK_GE[5][4] = {{-1,-1,-1,-1},{0,-1,-1,-1},{0,0,-1,-1},{0,0,0,-1},{0,0,0,0}};
alignas(32) static const long long MASK_LT[5][4] = {{0,0,0,0},{-1,0,0,0},{-1,-1,0,0},{-1,-1,-1,0},{-1,-1,-1,-1}};
inline __m256d mask_ge(int a) { return _mm256_load_pd(reinterpret_cast<const double *>(MASK_GE[a])); }
inline __m256d mask_lt(int b) { return _mm256_load_pd(reinterpret_cast<const double *>(MASK_LT[b])); }

/* Update of W (1..4) columns that all start in the same row group g0 (the
 * "staircase" group), by NP pivot columns:
 *   NP = 1:  y = fmadd(p0, b0[w], x)                       (b0 = temp = -r1*a_jk)
 *   NP = 2:  y = fnmadd(p1, b1[w], fnmadd(p0, b0[w], x))   (b0 = wk, b1 = wk+1)
 * 'L': column w lives in rows [g0 + lane0 + w, gB): group g0 is blended with
 *      the constant mask GE[lane0 + w], groups g0+4.. are full.
 * 'U': column w (w = 0 is the highest column) lives in rows [0, g0 + laneT - w]:
 *      groups [0, g0) are full, group g0 is blended with LT[laneT - w + 1].
 * If TRACK, column 0 is fed to the tracker: 'L' rows > its diagonal (mask
 * GE[lane0 + 1] in group g0, full afterwards), 'U' rows < its diagonal
 * (full groups before g0, LT[laneT] in group g0).
 * Every group is loaded and stored full width; the pinned load keeps GCC from
 * folding the load into the FMA (store forwarding needs the plain load). */
template <int W, int NP, bool TRACK>
inline __attribute__((always_inline)) void
tile_L(double *const c[W], int lane0, int g0, int gB, const double *__restrict p0, const double *__restrict p1,
       const __m256d b0[W], const __m256d b1[W], MaxTracker *tr) {
    auto upd = [&](int w, int g, __m256d q0, __m256d q1) {
        __m256d x = _mm256_loadu_pd(c[w] + g); RECLU_PIN(x);
        return NP == 2 ? _mm256_fnmadd_pd(q1, b1[w], _mm256_fnmadd_pd(q0, b0[w], x)) : _mm256_fmadd_pd(q0, b0[w], x);
    };
    {   /* staircase group */
        const __m256d q0 = _mm256_loadu_pd(p0 + g0), q1 = NP == 2 ? _mm256_loadu_pd(p1 + g0) : _mm256_setzero_pd();
        for (int w = 0; w < W; w++) {
            const __m256d x = _mm256_loadu_pd(c[w] + g0), y = _mm256_blendv_pd(x, upd(w, g0, q0, q1), mask_ge(lane0 + w));
            _mm256_storeu_pd(c[w] + g0, y);
            if (TRACK && w == 0) tr->feed(_mm256_and_pd(y, mask_ge(lane0 + 1)));
        }
    }
    #pragma GCC unroll 1
    for (int g = g0 + 4; g < gB; g += 4) {
        const __m256d q0 = _mm256_loadu_pd(p0 + g), q1 = NP == 2 ? _mm256_loadu_pd(p1 + g) : _mm256_setzero_pd();
        for (int w = 0; w < W; w++) {
            const __m256d y = upd(w, g, q0, q1);
            _mm256_storeu_pd(c[w] + g, y);
            if (TRACK && w == 0) tr->feed(y);
        }
    }
}
template <int W, int NP, bool TRACK>
inline __attribute__((always_inline)) void
tile_U(double *const c[W], int laneT, int g0, const double *__restrict p0, const double *__restrict p1,
       const __m256d b0[W], const __m256d b1[W], MaxTracker *tr) {
    auto upd = [&](int w, int g, __m256d q0, __m256d q1) {
        __m256d x = _mm256_loadu_pd(c[w] + g); RECLU_PIN(x);
        return NP == 2 ? _mm256_fnmadd_pd(q1, b1[w], _mm256_fnmadd_pd(q0, b0[w], x)) : _mm256_fmadd_pd(q0, b0[w], x);
    };
    #pragma GCC unroll 1
    for (int g = 0; g < g0; g += 4) {
        const __m256d q0 = _mm256_loadu_pd(p0 + g), q1 = NP == 2 ? _mm256_loadu_pd(p1 + g) : _mm256_setzero_pd();
        for (int w = 0; w < W; w++) {
            const __m256d y = upd(w, g, q0, q1);
            _mm256_storeu_pd(c[w] + g, y);
            if (TRACK && w == 0) tr->feed(y);
        }
    }
    {   /* staircase group */
        const __m256d q0 = _mm256_loadu_pd(p0 + g0), q1 = NP == 2 ? _mm256_loadu_pd(p1 + g0) : _mm256_setzero_pd();
        for (int w = 0; w < W; w++) {
            const __m256d x = _mm256_loadu_pd(c[w] + g0), y = _mm256_blendv_pd(x, upd(w, g0, q0, q1), mask_lt(laneT - w + 1));
            _mm256_storeu_pd(c[w] + g0, y);
            if (TRACK && w == 0) tr->feed(_mm256_and_pd(y, mask_lt(laneT)));
        }
    }
}

/* ================================================================ 'L' ==== */
struct Lower {
    int n, MP, lda; double *A;
    double &a(int i, int j) const { return A[i + (size_t)j * lda]; }
    double *col(int j) const { return A + (size_t)j * lda; }

    void scale(int j, int lo, double s) const {              /* rows [lo, MP) of column j *= s */
        double *c = col(j); const __m256d vs = _mm256_set1_pd(s); int g = lo & ~3;
        { const __m256d x = _mm256_loadu_pd(c + g); _mm256_storeu_pd(c + g, _mm256_blendv_pd(x, _mm256_mul_pd(x, vs), mask_ge(lo & 3))); }
        for (g += 4; g < MP; g += 4) _mm256_storeu_pd(c + g, _mm256_mul_pd(_mm256_loadu_pd(c + g), vs));
    }
    void swap_cols(int j1, int j2, int lo) const {           /* rows [lo, MP) of columns j1, j2 */
        double *c1 = col(j1), *c2 = col(j2); int g = lo & ~3;
        { const __m256d x1 = _mm256_loadu_pd(c1 + g), x2 = _mm256_loadu_pd(c2 + g), m = mask_ge(lo & 3);
          _mm256_storeu_pd(c1 + g, _mm256_blendv_pd(x1, x2, m)); _mm256_storeu_pd(c2 + g, _mm256_blendv_pd(x2, x1, m)); }
        for (g += 4; g < MP; g += 4) { const __m256d x1 = _mm256_loadu_pd(c1 + g), x2 = _mm256_loadu_pd(c2 + g); _mm256_storeu_pd(c1 + g, x2); _mm256_storeu_pd(c2 + g, x1); }
    }
    int colmax_scalar(int j, int lo, double *val) const {   /* first index of max |a| over rows [lo, n) */
        int im = lo; double dm = std::fabs(a(lo, j));
        for (int i = lo + 1; i < n; i++) { const double d = std::fabs(a(i, j)); if (d > dm) { dm = d; im = i; } }
        *val = dm; return im;
    }
    /* Walk columns [j, n) ascending in tiles that never cross a row group:
     * a leading partial tile up to the next group boundary, then aligned
     * tiles of 4, then the remainder. The first tile carries the tracker.
     * Ascending order matters for rank-2: wk/wkp1 are written back into the
     * pivot columns after each tile and later tiles must not read them. */
    template <class Prep, class Run>
    void walk(int j, Prep prep, Run run) const {
        bool first = true;
        while (j < n) {
            const int lane0 = j & 3, W = std::min(4 - lane0, n - j);
            switch (W) {
                case 1: run(std::integral_constant<int, 1>{}, j, lane0, first); break;
                case 2: run(std::integral_constant<int, 2>{}, j, lane0, first); break;
                case 3: run(std::integral_constant<int, 3>{}, j, lane0, first); break;
                default: run(std::integral_constant<int, 4>{}, j, lane0, first); break;
            }
            first = false; j += W;
        }
        (void)prep;
    }
    void rank1(int k, double r1, MaxTracker *tr) const {    /* 1x1 pivot k: columns k+1..n-1; track column k+1 rows >= k+2 */
        const double *p = col(k);
        tr->init((k + 1) & ~3, k + 2);
        walk(k + 1, 0, [&](auto Wc, int j0, int lane0, bool first) {
            constexpr int W = decltype(Wc)::value;
            double *c[W]; __m256d b0[W], b1[W];
            for (int w = 0; w < W; w++) { c[w] = col(j0 + w); b0[w] = _mm256_set1_pd(-r1 * a(j0 + w, k)); }
            if (first) tile_L<W, 1, true >(c, lane0, j0 & ~3, MP, p, nullptr, b0, b1, tr);
            else       tile_L<W, 1, false>(c, lane0, j0 & ~3, MP, p, nullptr, b0, b1, nullptr);
        });
    }
    void rank2(int k, double d11, double d22, double d21, MaxTracker *tr) const {   /* 2x2 pivot k,k+1: columns k+2..n-1; track column k+2 rows >= k+3 */
        const double *p0 = col(k), *p1 = col(k + 1);
        tr->init((k + 2) & ~3, k + 3);
        walk(k + 2, 0, [&](auto Wc, int j0, int lane0, bool first) {
            constexpr int W = decltype(Wc)::value;
            double *c[W]; __m256d b0[W], b1[W]; double wk[W], wkp1[W];
            for (int w = 0; w < W; w++) {
                c[w] = col(j0 + w);
                wk[w] = d21 * (d11 * a(j0 + w, k) - a(j0 + w, k + 1)); wkp1[w] = d21 * (d22 * a(j0 + w, k + 1) - a(j0 + w, k));
                b0[w] = _mm256_set1_pd(wk[w]); b1[w] = _mm256_set1_pd(wkp1[w]);
            }
            if (first) tile_L<W, 2, true >(c, lane0, j0 & ~3, MP, p0, p1, b0, b1, tr);
            else       tile_L<W, 2, false>(c, lane0, j0 & ~3, MP, p0, p1, b0, b1, nullptr);
            for (int w = 0; w < W; w++) { a(j0 + w, k) = wk[w]; a(j0 + w, k + 1) = wkp1[w]; }
        });
    }

    void factor(int *ipiv, int *info) const {
        int k = 0, imax = 0; double colmax = 0.0;
        if (n > 1) imax = colmax_scalar(0, 1, &colmax);
        while (k < n) {
            PROF_T0;
            int kstep = 1, kp;
            const double absakk = std::fabs(a(k, k));
            if (k == n - 1) { imax = k; colmax = 0.0; }
            MaxTracker tr; bool tracked = false;
            if (std::fmax(absakk, colmax) == 0.0 || std::isnan(absakk)) {
                if (*info == 0) *info = k + 1;
                kp = k;
            } else {
                if (absakk >= ALPHA * colmax) {
                    kp = k;
                } else {
                    double rowmax = 0.0;
                    for (int jj = k; jj < imax; jj++) rowmax = std::fmax(rowmax, std::fabs(a(imax, jj)));   /* row imax, cols k..imax-1 */
                    if (imax < n - 1) { double v; colmax_scalar(imax, imax + 1, &v); rowmax = std::fmax(rowmax, v); }
                    if (absakk >= ALPHA * colmax * (colmax / rowmax))   kp = k;
                    else if (std::fabs(a(imax, imax)) >= ALPHA * rowmax) kp = imax;
                    else { kp = imax; kstep = 2; }
                }
                const int kk = k + kstep - 1;
                if (kp != kk) {
                    if (kp < n - 1) swap_cols(kk, kp, kp + 1);
                    for (int i = kk + 1; i < kp; i++) { const double t = a(i, kk); a(i, kk) = a(kp, i); a(kp, i) = t; }
                    double t = a(kk, kk); a(kk, kk) = a(kp, kp); a(kp, kp) = t;
                    if (kstep == 2) { t = a(k + 1, k); a(k + 1, k) = a(kp, k); a(kp, k) = t; }
                }
                PROF_ADD(0);
                if (kstep == 1) {
                    if (k < n - 1) {
                        const double r1 = 1.0 / a(k, k);
                        tracked = (k + 2 < n);
                        rank1(k, r1, &tr); PROF_ADD(1);
                        scale(k, k + 1, r1); PROF_ADD(2);
                    }
                } else if (k < n - 2) {
                    double d21 = a(k + 1, k);
                    const double d11 = a(k + 1, k + 1) / d21, d22 = a(k, k) / d21;
                    const double t = 1.0 / (d11 * d22 - 1.0);
                    d21 = t / d21;
                    tracked = (k + 3 < n);
                    rank2(k, d11, d22, d21, &tr);
                }
            }
            if (kstep == 1) ipiv[k] = kp + 1; else { ipiv[k] = -(kp + 1); ipiv[k + 1] = -(kp + 1); }
            k += kstep; PROF_ADD(3);
            if (k < n - 1) {                                     /* column max for the next step */
#ifdef RECLU_SY_NOTRACK
                tracked = false;
#endif
                if (tracked) { imax = tr.reduce(); if (std::isnan(a(k + 1, k))) imax = k + 1; colmax = std::fabs(a(imax, k)); }   /* idamax: a leading NaN is never beaten */
                else imax = colmax_scalar(k, k + 1, &colmax);
            }
        }
    }
};

/* ================================================================ 'U' ==== */
struct Upper {
    int n, MP, lda; double *A;
    double &a(int i, int j) const { return A[i + (size_t)j * lda]; }
    double *col(int j) const { return A + (size_t)j * lda; }

    void scale(int j, int hi, double s) const {              /* rows [0, hi) of column j *= s */
        double *c = col(j); const __m256d vs = _mm256_set1_pd(s); const int gl = (hi - 1) & ~3; int g = 0;
        for (; g < gl; g += 4) _mm256_storeu_pd(c + g, _mm256_mul_pd(_mm256_loadu_pd(c + g), vs));
        { const __m256d x = _mm256_loadu_pd(c + g); _mm256_storeu_pd(c + g, _mm256_blendv_pd(x, _mm256_mul_pd(x, vs), mask_lt(hi - g))); }
    }
    void swap_cols(int j1, int j2, int hi) const {           /* rows [0, hi) */
        double *c1 = col(j1), *c2 = col(j2); const int gl = (hi - 1) & ~3; int g = 0;
        for (; g < gl; g += 4) { const __m256d x1 = _mm256_loadu_pd(c1 + g), x2 = _mm256_loadu_pd(c2 + g); _mm256_storeu_pd(c1 + g, x2); _mm256_storeu_pd(c2 + g, x1); }
        { const __m256d x1 = _mm256_loadu_pd(c1 + g), x2 = _mm256_loadu_pd(c2 + g), m = mask_lt(hi - g);
          _mm256_storeu_pd(c1 + g, _mm256_blendv_pd(x1, x2, m)); _mm256_storeu_pd(c2 + g, _mm256_blendv_pd(x2, x1, m)); }
    }
    int colmax_scalar(int j, int hi, double *val) const {   /* first index of max |a| over rows [0, hi) */
        int im = 0; double dm = std::fabs(a(0, j));
        for (int i = 1; i < hi; i++) { const double d = std::fabs(a(i, j)); if (d > dm) { dm = d; im = i; } }
        *val = dm; return im;
    }
    /* Walk columns [0, jtop] descending in tiles that never cross a row group;
     * the tile containing jtop comes first (tracker), then aligned tiles.
     * Column w = 0 of a tile is its highest column. */
    template <class Run>
    void walk(int jtop, Run run) const {
        bool first = true;
        while (jtop >= 0) {
            const int laneT = jtop & 3, W = std::min(laneT + 1, jtop + 1);
            switch (W) {
                case 1: run(std::integral_constant<int, 1>{}, jtop, laneT, first); break;
                case 2: run(std::integral_constant<int, 2>{}, jtop, laneT, first); break;
                case 3: run(std::integral_constant<int, 3>{}, jtop, laneT, first); break;
                default: run(std::integral_constant<int, 4>{}, jtop, laneT, first); break;
            }
            first = false; jtop -= W;
        }
    }
    void rank1(int k, double r1, MaxTracker *tr) const {    /* 1x1 pivot k: columns k-1..0; track column k-1 rows < k-1 */
        const double *p = col(k);
        tr->init(0, 0);
        walk(k - 1, [&](auto Wc, int jt, int laneT, bool first) {
            constexpr int W = decltype(Wc)::value;
            double *c[W]; __m256d b0[W], b1[W];
            for (int w = 0; w < W; w++) { c[w] = col(jt - w); b0[w] = _mm256_set1_pd(-r1 * a(jt - w, k)); }
            if (first) tile_U<W, 1, true >(c, laneT, jt & ~3, p, nullptr, b0, b1, tr);
            else       tile_U<W, 1, false>(c, laneT, jt & ~3, p, nullptr, b0, b1, nullptr);
        });
    }
    void rank2(int k, double d11, double d22, double d12, MaxTracker *tr) const {   /* 2x2 pivot k-1,k: columns k-2..0; track column k-2 rows < k-2 */
        const double *p0 = col(k), *p1 = col(k - 1);
        tr->init(0, 0);
        walk(k - 2, [&](auto Wc, int jt, int laneT, bool first) {
            constexpr int W = decltype(Wc)::value;
            double *c[W]; __m256d b0[W], b1[W]; double wk[W], wkm1[W];
            for (int w = 0; w < W; w++) {
                const int j = jt - w; c[w] = col(j);
                wkm1[w] = d12 * (d11 * a(j, k - 1) - a(j, k)); wk[w] = d12 * (d22 * a(j, k) - a(j, k - 1));
                b0[w] = _mm256_set1_pd(wk[w]); b1[w] = _mm256_set1_pd(wkm1[w]);
            }
            if (first) tile_U<W, 2, true >(c, laneT, jt & ~3, p0, p1, b0, b1, tr);
            else       tile_U<W, 2, false>(c, laneT, jt & ~3, p0, p1, b0, b1, nullptr);
            for (int w = 0; w < W; w++) { const int j = jt - w; a(j, k) = wk[w]; a(j, k - 1) = wkm1[w]; }
        });
    }

    void factor(int *ipiv, int *info) const {
        int k = n - 1, imax = 0; double colmax = 0.0;
        if (n > 1) imax = colmax_scalar(n - 1, n - 1, &colmax);
        while (k >= 0) {
            PROF_T0;
            int kstep = 1, kp;
            const double absakk = std::fabs(a(k, k));
            if (k == 0) { imax = 0; colmax = 0.0; }
            MaxTracker tr; bool tracked = false;
            if (std::fmax(absakk, colmax) == 0.0 || std::isnan(absakk)) {
                if (*info == 0) *info = k + 1;
                kp = k;
            } else {
                if (absakk >= ALPHA * colmax) {
                    kp = k;
                } else {
                    double rowmax = 0.0;
                    for (int jj = imax + 1; jj <= k; jj++) rowmax = std::fmax(rowmax, std::fabs(a(imax, jj)));   /* row imax, cols imax+1..k */
                    if (imax > 0) { double v; colmax_scalar(imax, imax, &v); rowmax = std::fmax(rowmax, v); }
                    if (absakk >= ALPHA * colmax * (colmax / rowmax))   kp = k;
                    else if (std::fabs(a(imax, imax)) >= ALPHA * rowmax) kp = imax;
                    else { kp = imax; kstep = 2; }
                }
                const int kk = k - kstep + 1;
                if (kp != kk) {
                    if (kp > 0) swap_cols(kk, kp, kp);
                    for (int i = kp + 1; i < kk; i++) { const double t = a(i, kk); a(i, kk) = a(kp, i); a(kp, i) = t; }
                    double t = a(kk, kk); a(kk, kk) = a(kp, kp); a(kp, kp) = t;
                    if (kstep == 2) { t = a(k - 1, k); a(k - 1, k) = a(kp, k); a(kp, k) = t; }
                }
                PROF_ADD(0);
                if (kstep == 1) {
                    if (k > 0) {
                        const double r1 = 1.0 / a(k, k);
                        tracked = (k - 1 > 0);
                        rank1(k, r1, &tr); PROF_ADD(1);
                        scale(k, k, r1); PROF_ADD(2);
                    }
                } else if (k > 1) {
                    double d12 = a(k - 1, k);
                    const double d22 = a(k - 1, k - 1) / d12, d11 = a(k, k) / d12;
                    const double t = 1.0 / (d11 * d22 - 1.0);
                    d12 = t / d12;
                    tracked = (k - 2 > 0);
                    rank2(k, d11, d22, d12, &tr);
                }
            }
            if (kstep == 1) ipiv[k] = kp + 1; else { ipiv[k] = -(kp + 1); ipiv[k - 1] = -(kp + 1); }
            k -= kstep; PROF_ADD(3);
            if (k > 0) {
#ifdef RECLU_SY_NOTRACK
                tracked = false;
#endif
                if (tracked) { imax = tr.reduce(); if (std::isnan(a(0, k))) imax = 0; colmax = std::fabs(a(imax, k)); }
                else imax = colmax_scalar(k, k, &colmax);
            }
        }
    }
};

__attribute__((noinline)) void factor_copy(bool upper, int n, double *A, int lda, int *ipiv, int *info) {
    const int MP = (n + 3) & ~3;
    alignas(32) double buf[RECLU_MAXM * RECLU_MAXM + 4];
    for (int j = 0; j < n; j++) { std::memcpy(buf + (size_t)j * MP, A + (size_t)j * lda, n * 8); for (int i = n; i < MP; i++) buf[(size_t)j * MP + i] = 0.0; }
    if (upper) Upper{n, MP, MP, buf}.factor(ipiv, info); else Lower{n, MP, MP, buf}.factor(ipiv, info);
    for (int j = 0; j < n; j++) {                             /* copy back the factored triangle only */
        const int i0 = upper ? 0 : j, i1 = upper ? j + 1 : n;
        std::memcpy(A + (size_t)j * lda + i0, buf + (size_t)j * MP + i0, (size_t)(i1 - i0) * 8);
    }
}
} // namespace

void reclu_sytf2_opt(char uplo, int n, double *A, int lda, int *ipiv, int *info) {
    *info = 0;
    const bool upper = (uplo == 'U' || uplo == 'u');
    if (!upper && !(uplo == 'L' || uplo == 'l')) { *info = -1; return; }
    if (n < 0) { *info = -2; return; }
    if (lda < (n < 1 ? 1 : n)) { *info = -4; return; }
    if (n == 0) return;
    if (n < RECLU_SMALL || n > RECLU_MAXM) { reclu_sytf2(uplo, n, A, lda, ipiv, info); return; }
    const int MP = (n + 3) & ~3;
    if ((lda & 3) == 0 && lda >= MP) {
        if (upper) Upper{n, MP, lda, A}.factor(ipiv, info);
        else { for (int j = 0; j < n; j++) for (int i = n; i < MP; i++) A[(size_t)j * lda + i] = 0.0; Lower{n, MP, lda, A}.factor(ipiv, info); }
    } else {
        factor_copy(upper, n, A, lda, ipiv, info);
    }
}

/* ============================================================== solve ==== */
namespace {

/* Single RHS kernels. Row ranges are [lo, MP) for 'L' (first group blended
 * with GE[lo & 3]) and [0, hi) for 'U' (last group blended with LT[hi - g]).
 * The dot products accumulate row i in lane i & 3 of group-aligned vectors
 * (fused) and combine the lanes as (t0 + t1) + (t2 + t3), the reference's
 * summation order; both operands are masked
 * in the boundary group so that a dead lane holding Inf cannot produce
 * 0*Inf = NaN. */
template <int NP>
inline void axpy_L(double *__restrict x, const double *__restrict p0, double b0, const double *__restrict p1, double b1, int lo, int MP) {
    if (lo >= MP) return;
    const __m256d vb0 = _mm256_set1_pd(b0), vb1 = _mm256_set1_pd(b1);
    auto upd = [&](int g, __m256d o) { __m256d y = _mm256_fnmadd_pd(_mm256_loadu_pd(p0 + g), vb0, o); if (NP == 2) y = _mm256_fnmadd_pd(_mm256_loadu_pd(p1 + g), vb1, y); return y; };
    int g = lo & ~3;
    { const __m256d o = _mm256_loadu_pd(x + g); _mm256_storeu_pd(x + g, _mm256_blendv_pd(o, upd(g, o), mask_ge(lo & 3))); }
    for (g += 4; g < MP; g += 4) _mm256_storeu_pd(x + g, upd(g, _mm256_loadu_pd(x + g)));
}
template <int NP>
inline void axpy_U(double *__restrict x, const double *__restrict p0, double b0, const double *__restrict p1, double b1, int hi) {
    if (hi <= 0) return;
    const __m256d vb0 = _mm256_set1_pd(b0), vb1 = _mm256_set1_pd(b1);
    auto upd = [&](int g, __m256d o) { __m256d y = _mm256_fnmadd_pd(_mm256_loadu_pd(p0 + g), vb0, o); if (NP == 2) y = _mm256_fnmadd_pd(_mm256_loadu_pd(p1 + g), vb1, y); return y; };
    const int gl = (hi - 1) & ~3; int g = 0;
    for (; g < gl; g += 4) _mm256_storeu_pd(x + g, upd(g, _mm256_loadu_pd(x + g)));
    { const __m256d o = _mm256_loadu_pd(x + g); _mm256_storeu_pd(x + g, _mm256_blendv_pd(o, upd(g, o), mask_lt(hi - g))); }
}
inline double hsum(__m256d acc) { alignas(32) double t[4]; _mm256_store_pd(t, acc); return (t[0] + t[1]) + (t[2] + t[3]); }
inline double dot_L(const double *__restrict p, const double *__restrict x, int lo, int MP) {
    if (lo >= MP) return 0.0;
    int g = lo & ~3; const __m256d m = mask_ge(lo & 3);
    __m256d acc = _mm256_mul_pd(_mm256_and_pd(_mm256_loadu_pd(p + g), m), _mm256_and_pd(_mm256_loadu_pd(x + g), m));
    for (g += 4; g < MP; g += 4) acc = _mm256_fmadd_pd(_mm256_loadu_pd(p + g), _mm256_loadu_pd(x + g), acc);
    return hsum(acc);
}
inline double dot_U(const double *__restrict p, const double *__restrict x, int hi) {
    if (hi <= 0) return 0.0;
    const int gl = (hi - 1) & ~3; int g = 0; __m256d acc = _mm256_setzero_pd();
    for (; g < gl; g += 4) acc = _mm256_fmadd_pd(_mm256_loadu_pd(p + g), _mm256_loadu_pd(x + g), acc);
    const __m256d m = mask_lt(hi - g);
    acc = _mm256_fmadd_pd(_mm256_and_pd(_mm256_loadu_pd(p + g), m), _mm256_and_pd(_mm256_loadu_pd(x + g), m), acc);
    return hsum(acc);
}

template <bool UPPER>
void solve_1(int n, int MP, const double *A, int lda, const int *ipiv, double *x) {
    auto a = [&](int i, int j) { return A[i + (size_t)j * lda]; };
    auto col = [&](int j) { return A + (size_t)j * lda; };
    auto swp = [&](int r1, int r2) { if (r1 != r2) std::swap(x[r1], x[r2]); };
    auto solve2 = [&](int kA, int kB, double akm1k) {        /* rows kA = k-1 (U) / k (L), kB = k / k+1 */
        const double akm1 = a(kA, kA) / akm1k, ak = a(kB, kB) / akm1k, denom = akm1 * ak - 1.0;
        const double bkm1 = x[kA] / akm1k, bk = x[kB] / akm1k;
        x[kA] = (ak * bkm1 - bk) / denom; x[kB] = (akm1 * bk - bkm1) / denom;
    };
    if (!UPPER) {
        int k = 0;
        while (k < n) {
            if (ipiv[k] > 0) { swp(k, ipiv[k] - 1); axpy_L<1>(x, col(k), x[k], nullptr, 0.0, k + 1, MP); x[k] *= 1.0 / a(k, k); k += 1; }
            else { swp(k + 1, -ipiv[k] - 1); axpy_L<2>(x, col(k), x[k], col(k + 1), x[k + 1], k + 2, MP); solve2(k, k + 1, a(k + 1, k)); k += 2; }
        }
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) { x[k] -= dot_L(col(k), x, k + 1, MP); swp(k, ipiv[k] - 1); k -= 1; }
            else { x[k] -= dot_L(col(k), x, k + 1, MP); x[k - 1] -= dot_L(col(k - 1), x, k + 1, MP); swp(k, -ipiv[k] - 1); k -= 2; }
        }
    } else {
        int k = n - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) { swp(k, ipiv[k] - 1); axpy_U<1>(x, col(k), x[k], nullptr, 0.0, k); x[k] *= 1.0 / a(k, k); k -= 1; }
            else { swp(k - 1, -ipiv[k] - 1); axpy_U<2>(x, col(k), x[k], col(k - 1), x[k - 1], k - 1); solve2(k - 1, k, a(k - 1, k)); k -= 2; }
        }
        k = 0;
        while (k < n) {
            if (ipiv[k] > 0) { x[k] -= dot_U(col(k), x, k); swp(k, ipiv[k] - 1); k += 1; }
            else { x[k] -= dot_U(col(k), x, k); x[k + 1] -= dot_U(col(k + 1), x, k); swp(k, -ipiv[k] - 1); k += 2; }
        }
    }
}

/* 4 RHS in lanes: X row-major, row i = one ymm. Sequential dots -> bit-identical. */
template <bool UPPER>
void solve_4(int n, const double *A, int lda, const int *ipiv, double *B, int ldb) {
    auto a = [&](int i, int j) { return A[i + (size_t)j * lda]; };
    auto col = [&](int j) { return A + (size_t)j * lda; };
    alignas(32) double X[(RECLU_MAXM + 4) * 4];
    auto row = [&](int i) { return X + (size_t)i * 4; };
    for (int i = 0; i < n; i++) for (int r = 0; r < 4; r++) X[(size_t)i * 4 + r] = B[i + (size_t)r * ldb];
    auto swp = [&](int r1, int r2) { if (r1 != r2) { const __m256d t = _mm256_load_pd(row(r1)); _mm256_store_pd(row(r1), _mm256_load_pd(row(r2))); _mm256_store_pd(row(r2), t); } };
    auto axpy = [&](const double *p, int kx, const double *p1, int kx1, int lo, int hi) {   /* X[i] -= a(i,kx) X[kx] (- a(i,kx1) X[kx1]) */
        const __m256d xk = _mm256_load_pd(row(kx)), xk1 = p1 ? _mm256_load_pd(row(kx1)) : _mm256_setzero_pd();
        for (int i = lo; i < hi; i++) {
            __m256d y = _mm256_fnmadd_pd(_mm256_broadcast_sd(p + i), xk, _mm256_load_pd(row(i)));
            if (p1) y = _mm256_fnmadd_pd(_mm256_broadcast_sd(p1 + i), xk1, y);
            _mm256_store_pd(row(i), y);
        }
    };
    /* X[kx] -= sum_i a(i,.) X[i]: four fused chains (row i -> chain i & 3),
     * combined as (t0 + t1) + (t2 + t3). Same order as the reference's
     * dot_rows; whether the reference's terms are fused is up to the compiler
     * (see reclu_sy_naive.cpp), so the solve agrees to a few ulp. */
    auto dotsub = [&](int kx, const double *p, int lo, int hi) {
        __m256d acc0 = _mm256_setzero_pd(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        auto step = [&](__m256d a, int i) { return _mm256_fmadd_pd(_mm256_broadcast_sd(p + i), _mm256_load_pd(row(i)), a); };
        int i = lo;
        for (; i < hi && (i & 3) != 0; i++) {                   /* head: rows up to the next multiple of 4 */
            if ((i & 3) == 1) acc1 = step(acc1, i); else if ((i & 3) == 2) acc2 = step(acc2, i); else acc3 = step(acc3, i);
        }
        for (; i + 4 <= hi; i += 4) { acc0 = step(acc0, i); acc1 = step(acc1, i + 1); acc2 = step(acc2, i + 2); acc3 = step(acc3, i + 3); }
        if (i < hi) { acc0 = step(acc0, i); i++; }
        if (i < hi) { acc1 = step(acc1, i); i++; }
        if (i < hi) { acc2 = step(acc2, i); }
        const __m256d acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
        _mm256_store_pd(row(kx), _mm256_sub_pd(_mm256_load_pd(row(kx)), acc));
    };
    auto scale = [&](int kx, double s) { _mm256_store_pd(row(kx), _mm256_mul_pd(_mm256_load_pd(row(kx)), _mm256_set1_pd(s))); };
    auto solve2 = [&](int kA, int kB, double akm1k) {
        const double akm1 = a(kA, kA) / akm1k, ak = a(kB, kB) / akm1k, denom = akm1 * ak - 1.0;
        const __m256d ia = _mm256_set1_pd(akm1k), vkm1 = _mm256_set1_pd(akm1), vk = _mm256_set1_pd(ak), vd = _mm256_set1_pd(denom);
        const __m256d bkm1 = _mm256_div_pd(_mm256_load_pd(row(kA)), ia), bk = _mm256_div_pd(_mm256_load_pd(row(kB)), ia);
        _mm256_store_pd(row(kA), _mm256_div_pd(_mm256_sub_pd(_mm256_mul_pd(vk, bkm1), bk), vd));
        _mm256_store_pd(row(kB), _mm256_div_pd(_mm256_sub_pd(_mm256_mul_pd(vkm1, bk), bkm1), vd));
    };
    if (!UPPER) {
        int k = 0;
        while (k < n) {
            if (ipiv[k] > 0) { swp(k, ipiv[k] - 1); axpy(col(k), k, nullptr, 0, k + 1, n); scale(k, 1.0 / a(k, k)); k += 1; }
            else { swp(k + 1, -ipiv[k] - 1); axpy(col(k), k, col(k + 1), k + 1, k + 2, n); solve2(k, k + 1, a(k + 1, k)); k += 2; }
        }
        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) { dotsub(k, col(k), k + 1, n); swp(k, ipiv[k] - 1); k -= 1; }
            else { dotsub(k, col(k), k + 1, n); dotsub(k - 1, col(k - 1), k + 1, n); swp(k, -ipiv[k] - 1); k -= 2; }
        }
    } else {
        int k = n - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) { swp(k, ipiv[k] - 1); axpy(col(k), k, nullptr, 0, 0, k); scale(k, 1.0 / a(k, k)); k -= 1; }
            else { swp(k - 1, -ipiv[k] - 1); axpy(col(k), k, col(k - 1), k - 1, 0, k - 1); solve2(k - 1, k, a(k - 1, k)); k -= 2; }
        }
        k = 0;
        while (k < n) {
            if (ipiv[k] > 0) { dotsub(k, col(k), 0, k); swp(k, ipiv[k] - 1); k += 1; }
            else { dotsub(k, col(k), 0, k); dotsub(k + 1, col(k + 1), 0, k); swp(k, -ipiv[k] - 1); k += 2; }
        }
    }
    for (int i = 0; i < n; i++) for (int r = 0; r < 4; r++) B[i + (size_t)r * ldb] = X[(size_t)i * 4 + r];
}
} // namespace

/* Below this size a 1..3 RHS remainder is solved by the reference loops:
 * the substitution is a latency chain through x[k] (about 10 cycles per
 * pivot whichever way it is written), and the reference's simd loops with
 * a reduction dot are as fast as the vector kernel up to n ~ 100 on
 * Skylake-X (see PERFORMANCE.md); the kernel only pulls ahead beyond that. */
#ifndef RECLU_SY_SOLVE1_MIN
#define RECLU_SY_SOLVE1_MIN 112
#endif

void reclu_sytrs_opt(char uplo, int n, int nrhs, const double *A, int lda, const int *ipiv, double *B, int ldb) {
    if (n <= 0 || nrhs <= 0) return;
    const bool upper = (uplo == 'U' || uplo == 'u');
    const int MP = (n + 3) & ~3;
    if ((lda & 3) != 0 || lda < MP || n > RECLU_MAXM || n < RECLU_SMALL) { reclu_sytrs(uplo, n, nrhs, A, lda, ipiv, B, ldb); return; }
    int r = 0;
    for (; r + 4 <= nrhs; r += 4) { if (upper) solve_4<true>(n, A, lda, ipiv, B + (size_t)r * ldb, ldb); else solve_4<false>(n, A, lda, ipiv, B + (size_t)r * ldb, ldb); }
    if (r < nrhs && n < RECLU_SY_SOLVE1_MIN) { reclu_sytrs(uplo, n, nrhs - r, A, lda, ipiv, B + (size_t)r * ldb, ldb); return; }
    alignas(32) double x[RECLU_MAXM + 4];
    for (; r < nrhs; r++) {
        double *b = B + (size_t)r * ldb;
        std::memcpy(x, b, n * 8); for (int i = n; i < MP; i++) x[i] = 0.0;
        if (upper) solve_1<true>(n, MP, A, lda, ipiv, x); else solve_1<false>(n, MP, A, lda, ipiv, x);
        std::memcpy(b, x, n * 8);
    }
}
#endif
