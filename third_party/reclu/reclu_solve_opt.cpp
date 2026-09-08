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
void reclu_solve_opt(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) { reclu_solve_naive(m, nrhs, LU, lda, ipiv, B, ldb); }
void reclu_solve_asm(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) { reclu_solve_naive(m, nrhs, LU, lda, ipiv, B, ldb); }
#else

#include "reclu_simd.h"
#include <cmath>
#include <cstring>

/* Optimized dgetrs(N). The substitution is latency-bound (each x[k] depends on
 * the previous ones), so the tricks target the serial chain:
 *
 *  - diagonal reciprocals are computed up front with vector divides, so the
 *    back-substitution chain carries a 4-cycle multiply instead of a 14-cycle
 *    divide per pivot (this is the only arithmetic difference from the naive
 *    kernel: <= 1 ulp per pivot step);
 *  - pivots are handled in blocks of 4: the <= 4x4 triangle in scalar code,
 *    then one rank-4 streaming update of the remaining rows -- 4x less
 *    traffic on x and one chain segment per block instead of per row;
 *  - with nrhs >= 4, four right-hand sides ride in the 4 lanes of a ymm
 *    (row-major X), so every FMA does 4 solves and the chain is shared;
 *    eight at a time when nrhs >= 8. Leftover columns use the 1-RHS kernel.
 *
 * Per element the FMAs are applied in the same order as in the naive kernel
 * (k increasing for L, decreasing for U). ASM selects the inline-asm bodies
 * for the two streaming loops; otherwise they are intrinsics. */

namespace {

constexpr int KB = 4;                                   /* pivots per block */

inline void apply_pivots(int m, const int *ipiv, double *x) {
    for (int k = 0; k < m; k++) {
        const int kp = ipiv[k] - 1;
        if (kp != k) { const double t = x[k]; x[k] = x[kp]; x[kp] = t; }
    }
}
/* same on row-major X with rows of G vectors */
template <int G> inline void apply_pivots_rows(int m, const int *ipiv, double *X) {
    for (int k = 0; k < m; k++) {
        const int kp = ipiv[k] - 1;
        if (kp != k) for (int g = 0; g < G; g++) {
            double *a = X + (size_t)k * 4 * G + 4 * g, *b = X + (size_t)kp * 4 * G + 4 * g;
            const __m256d t = _mm256_load_pd(a); _mm256_store_pd(a, _mm256_load_pd(b)); _mm256_store_pd(b, t);
        }
    }
}

/* inv[k] = 1 / U(k,k), vectorized (all independent). */
inline void diag_recip(int m, int MP, const double *LU, int lda, double *inv) {
    alignas(32) double d[RECLU_MAXM + 4];
    for (int k = 0; k < m; k++) d[k] = LU[k + (size_t)k * lda];
    for (int k = m; k < MP; k++) d[k] = 1.0;
    for (int k = 0; k < MP; k += 4) _mm256_store_pd(inv + k, _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_load_pd(d + k)));
}

/* One pivot block: the KB columns and the KB broadcast x values, in the
 * order the FMAs must be applied. */
struct Block { const double *c[KB]; __m256d b[KB]; int kk; };

/* ---- 1 RHS: x[i] -= sum_q c[q][i] * b[q], rows i0..i1 (multiples of 4) ---- */
template <bool ASM>
inline __attribute__((always_inline)) void stream_1(double *__restrict x, const Block &blk, int i0, int i1) {
    int i = i0;
    if (ASM && blk.kk == KB) {
        long cnt = (i1 - i0) / 8, off = 0;
        if (cnt) {
            __asm__ volatile(
                "1:\n\t"
                "vmovupd   (%[x],%[o],8), %%ymm4\n\t"
                "vmovupd 32(%[x],%[o],8), %%ymm5\n\t"
                "vfnmadd231pd   (%[c0],%[o],8), %[b0], %%ymm4\n\t"  "vfnmadd231pd 32(%[c0],%[o],8), %[b0], %%ymm5\n\t"
                "vfnmadd231pd   (%[c1],%[o],8), %[b1], %%ymm4\n\t"  "vfnmadd231pd 32(%[c1],%[o],8), %[b1], %%ymm5\n\t"
                "vfnmadd231pd   (%[c2],%[o],8), %[b2], %%ymm4\n\t"  "vfnmadd231pd 32(%[c2],%[o],8), %[b2], %%ymm5\n\t"
                "vfnmadd231pd   (%[c3],%[o],8), %[b3], %%ymm4\n\t"  "vfnmadd231pd 32(%[c3],%[o],8), %[b3], %%ymm5\n\t"
                "vmovupd %%ymm4,   (%[x],%[o],8)\n\t"
                "vmovupd %%ymm5, 32(%[x],%[o],8)\n\t"
                "addq $8, %[o]\n\t"  "decq %[n]\n\t"  "jnz 1b\n\t"
                : [o] "+r"(off), [n] "+r"(cnt)
                : [x] "r"(x + i0), [c0] "r"(blk.c[0] + i0), [c1] "r"(blk.c[1] + i0), [c2] "r"(blk.c[2] + i0), [c3] "r"(blk.c[3] + i0),
                  [b0] "x"(blk.b[0]), [b1] "x"(blk.b[1]), [b2] "x"(blk.b[2]), [b3] "x"(blk.b[3])
                : "ymm4", "ymm5", "memory", "cc");
            i += (int)off;
        }
    } else {
        for (; i + 8 <= i1; i += 8) {
            __m256d a0 = _mm256_loadu_pd(x + i), a1 = _mm256_loadu_pd(x + i + 4);
            RECLU_PIN(a0); RECLU_PIN(a1);
            for (int q = 0; q < blk.kk; q++) {
                a0 = _mm256_fnmadd_pd(_mm256_loadu_pd(blk.c[q] + i), blk.b[q], a0);
                a1 = _mm256_fnmadd_pd(_mm256_loadu_pd(blk.c[q] + i + 4), blk.b[q], a1);
            }
            _mm256_storeu_pd(x + i, a0); _mm256_storeu_pd(x + i + 4, a1);
        }
    }
    if (i < i1) {
        __m256d a0 = _mm256_loadu_pd(x + i); RECLU_PIN(a0);
        for (int q = 0; q < blk.kk; q++) a0 = _mm256_fnmadd_pd(_mm256_loadu_pd(blk.c[q] + i), blk.b[q], a0);
        _mm256_storeu_pd(x + i, a0);
    }
}

/* x: MP doubles (pad rows zero). */
template <bool ASM>
void solve_1(int m, int MP, const double *LU, int lda, const int *ipiv, const double *inv, double *x) {
    auto col = [&](int k) { return LU + (size_t)k * lda; };
    apply_pivots(m, ipiv, x);

    for (int kb = 0; kb < m; kb += KB) {                             /* L y = P b */
        const int kk = (m - kb < KB) ? (m - kb) : KB;
        for (int k = kb; k < kb + kk; k++) {
            const double *c = col(k); const double xk = x[k];      /* hoisted: x[i] stores may alias x[k] */
            for (int i = k + 1; i < kb + kk; i++) x[i] = std::fma(-c[i], xk, x[i]);
        }
        if (kb + KB < MP) {
            Block blk; blk.kk = kk;
            for (int q = 0; q < kk; q++) { blk.c[q] = col(kb + q); blk.b[q] = _mm256_broadcast_sd(x + kb + q); }
            stream_1<ASM>(x, blk, kb + KB, MP);
        }
    }
    for (int kb = (m - 1) & ~(KB - 1); kb >= 0; kb -= KB) {           /* U x = y */
        const int kk = (m - kb < KB) ? (m - kb) : KB;
        for (int k = kb + kk - 1; k >= kb; k--) {
            const double *c = col(k); const double xk = x[k] * inv[k];
            x[k] = xk;
            for (int i = kb; i < k; i++) x[i] = std::fma(-c[i], xk, x[i]);
        }
        if (kb > 0) {
            Block blk; blk.kk = kk;
            for (int q = 0; q < kk; q++) { const int k = kb + kk - 1 - q; blk.c[q] = col(k); blk.b[q] = _mm256_broadcast_sd(x + k); }
            stream_1<ASM>(x, blk, 0, kb);
        }
    }
}

/* ---- 4G RHS in lanes: X is row-major, row i = G ymm ------------------------ */
template <int G> struct LaneBlock { const double *c[KB]; __m256d xk[KB][G]; int kk; };

/* X[i] -= sum_q bcast(c[q][i]) * xk[q], rows i0..i1-1 */
template <int G, bool ASM>
inline __attribute__((always_inline)) void stream_lanes(double *__restrict X, const LaneBlock<G> &blk, int i0, int i1) {
    constexpr int W = 4 * G;
    int i = i0;
    if (ASM && G == 1 && blk.kk == KB) {
        long cnt = (i1 - i0) / 2, off = 0, ri = 0;
        if (cnt) {
            __asm__ volatile(
                "1:\n\t"
                "vmovapd   (%[x],%[o],8), %%ymm4\n\t"      /* row i   (4 rhs) */
                "vmovapd 32(%[x],%[o],8), %%ymm5\n\t"      /* row i+1 */
                "vbroadcastsd  (%[c0],%[r],8), %%ymm6\n\t"  "vbroadcastsd 8(%[c0],%[r],8), %%ymm7\n\t"
                "vfnmadd231pd %%ymm6, %[k0], %%ymm4\n\t"    "vfnmadd231pd %%ymm7, %[k0], %%ymm5\n\t"
                "vbroadcastsd  (%[c1],%[r],8), %%ymm6\n\t"  "vbroadcastsd 8(%[c1],%[r],8), %%ymm7\n\t"
                "vfnmadd231pd %%ymm6, %[k1], %%ymm4\n\t"    "vfnmadd231pd %%ymm7, %[k1], %%ymm5\n\t"
                "vbroadcastsd  (%[c2],%[r],8), %%ymm6\n\t"  "vbroadcastsd 8(%[c2],%[r],8), %%ymm7\n\t"
                "vfnmadd231pd %%ymm6, %[k2], %%ymm4\n\t"    "vfnmadd231pd %%ymm7, %[k2], %%ymm5\n\t"
                "vbroadcastsd  (%[c3],%[r],8), %%ymm6\n\t"  "vbroadcastsd 8(%[c3],%[r],8), %%ymm7\n\t"
                "vfnmadd231pd %%ymm6, %[k3], %%ymm4\n\t"    "vfnmadd231pd %%ymm7, %[k3], %%ymm5\n\t"
                "vmovapd %%ymm4,   (%[x],%[o],8)\n\t"
                "vmovapd %%ymm5, 32(%[x],%[o],8)\n\t"
                "addq $8, %[o]\n\t"  "addq $2, %[r]\n\t"  "decq %[n]\n\t"  "jnz 1b\n\t"
                : [o] "+r"(off), [r] "+r"(ri), [n] "+r"(cnt)
                : [x] "r"(X + (size_t)i0 * 4), [c0] "r"(blk.c[0] + i0), [c1] "r"(blk.c[1] + i0), [c2] "r"(blk.c[2] + i0), [c3] "r"(blk.c[3] + i0),
                  [k0] "x"(blk.xk[0][0]), [k1] "x"(blk.xk[1][0]), [k2] "x"(blk.xk[2][0]), [k3] "x"(blk.xk[3][0])
                : "ymm4", "ymm5", "ymm6", "ymm7", "memory", "cc");
            i += (int)ri;
        }
    }
    /* two rows per iteration (their broadcast loads interleave; ~20% over one) */
    for (; i + 2 <= i1; i += 2) {
        __m256d a[G], d[G];
        for (int g = 0; g < G; g++) { a[g] = _mm256_load_pd(X + (size_t)i * W + 4 * g); d[g] = _mm256_load_pd(X + (size_t)(i + 1) * W + 4 * g); RECLU_PIN(a[g]); RECLU_PIN(d[g]); }
        for (int q = 0; q < blk.kk; q++) {
            const __m256d l0 = _mm256_broadcast_sd(blk.c[q] + i), l1 = _mm256_broadcast_sd(blk.c[q] + i + 1);
            for (int g = 0; g < G; g++) { a[g] = _mm256_fnmadd_pd(l0, blk.xk[q][g], a[g]); d[g] = _mm256_fnmadd_pd(l1, blk.xk[q][g], d[g]); }
        }
        for (int g = 0; g < G; g++) { _mm256_store_pd(X + (size_t)i * W + 4 * g, a[g]); _mm256_store_pd(X + (size_t)(i + 1) * W + 4 * g, d[g]); }
    }
    if (i < i1) {
        __m256d a[G];
        for (int g = 0; g < G; g++) { a[g] = _mm256_load_pd(X + (size_t)i * W + 4 * g); RECLU_PIN(a[g]); }
        for (int q = 0; q < blk.kk; q++) { const __m256d l = _mm256_broadcast_sd(blk.c[q] + i); for (int g = 0; g < G; g++) a[g] = _mm256_fnmadd_pd(l, blk.xk[q][g], a[g]); }
        for (int g = 0; g < G; g++) _mm256_store_pd(X + (size_t)i * W + 4 * g, a[g]);
    }
}

template <int G, bool ASM>
void solve_lanes(int m, const double *LU, int lda, const int *ipiv, const double *inv, double *B, int ldb) {
    constexpr int W = 4 * G;
    auto col = [&](int k) { return LU + (size_t)k * lda; };
    alignas(32) double X[(RECLU_MAXM + 4) * W];
    auto row = [&](int i, int g) { return X + (size_t)i * W + 4 * g; };
    auto axpy_row = [&](int i, int k, __m256d f) {           /* X[i] -= f * X[k] */
        for (int g = 0; g < G; g++) _mm256_store_pd(row(i, g), _mm256_fnmadd_pd(f, _mm256_load_pd(row(k, g)), _mm256_load_pd(row(i, g))));
    };

    for (int i = 0; i < m; i++) for (int r = 0; r < W; r++) X[(size_t)i * W + r] = B[i + (size_t)r * ldb];   /* transpose in */
    apply_pivots_rows<G>(m, ipiv, X);

    for (int kb = 0; kb < m; kb += KB) {                             /* L */
        const int kk = (m - kb < KB) ? (m - kb) : KB;
        for (int k = kb; k < kb + kk; k++)
            for (int i = k + 1; i < kb + kk; i++) axpy_row(i, k, _mm256_broadcast_sd(col(k) + i));
        if (kb + kk < m) {
            LaneBlock<G> blk; blk.kk = kk;
            for (int q = 0; q < kk; q++) { blk.c[q] = col(kb + q); for (int g = 0; g < G; g++) blk.xk[q][g] = _mm256_load_pd(row(kb + q, g)); }
            stream_lanes<G, ASM>(X, blk, kb + kk, m);
        }
    }
    for (int kb = (m - 1) & ~(KB - 1); kb >= 0; kb -= KB) {           /* U */
        const int kk = (m - kb < KB) ? (m - kb) : KB;
        for (int k = kb + kk - 1; k >= kb; k--) {
            const __m256d ik = _mm256_broadcast_sd(inv + k);
            for (int g = 0; g < G; g++) _mm256_store_pd(row(k, g), _mm256_mul_pd(_mm256_load_pd(row(k, g)), ik));
            for (int i = kb; i < k; i++) axpy_row(i, k, _mm256_broadcast_sd(col(k) + i));
        }
        if (kb > 0) {
            LaneBlock<G> blk; blk.kk = kk;
            for (int q = 0; q < kk; q++) { const int k = kb + kk - 1 - q; blk.c[q] = col(k); for (int g = 0; g < G; g++) blk.xk[q][g] = _mm256_load_pd(row(k, g)); }
            stream_lanes<G, ASM>(X, blk, 0, kb);
        }
    }
    for (int i = 0; i < m; i++) for (int r = 0; r < W; r++) B[i + (size_t)r * ldb] = X[(size_t)i * W + r];   /* transpose out */
}

template <bool ASM>
void solve(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) {
    const int MP = (m + 3) & ~3;
    if (m <= 0 || nrhs <= 0) return;
    if ((lda & 3) != 0 || lda < MP || m > RECLU_MAXM) { reclu_solve_naive(m, nrhs, LU, lda, ipiv, B, ldb); return; }

    alignas(32) double inv[RECLU_MAXM + 4], x[RECLU_MAXM + 4];
    diag_recip(m, MP, LU, lda, inv);
    int r = 0;
    for (; r + 8 <= nrhs; r += 8) solve_lanes<2, ASM>(m, LU, lda, ipiv, inv, B + (size_t)r * ldb, ldb);
    for (; r + 4 <= nrhs; r += 4) solve_lanes<1, ASM>(m, LU, lda, ipiv, inv, B + (size_t)r * ldb, ldb);
    for (; r < nrhs; r++) {
        double *b = B + (size_t)r * ldb;
        std::memcpy(x, b, m * sizeof(double));
        for (int i = m; i < MP; i++) x[i] = 0.0;
        solve_1<ASM>(m, MP, LU, lda, ipiv, inv, x);
        std::memcpy(b, x, m * sizeof(double));
    }
}

} // namespace

void reclu_solve_opt(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) { solve<false>(m, nrhs, LU, lda, ipiv, B, ldb); }
void reclu_solve_asm(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb) { solve<true >(m, nrhs, LU, lda, ipiv, B, ldb); }
#endif
