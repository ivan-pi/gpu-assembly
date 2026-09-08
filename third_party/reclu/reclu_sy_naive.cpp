/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code was reviewed and validated by tests against netlib LAPACK, but
 * AI-generated material can contain subtle errors; verify before relying on it.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#include "reclu.h"
#include <cmath>

/* Reference LDL^T with Bunch-Kaufman diagonal pivoting (LAPACK dsytf2) and
 * the matching solve (dsytrs), plain loops, UPLO = 'U' and 'L'.
 *
 * The two fill modes are one implementation parameterized by a trait
 * (LowerFill / UpperFill) that captures what differs between them: the
 * direction of the pivot walk (k = 0, 1, ... for 'L'; k = n-1, n-2, ... for
 * 'U'), the partner row of a 2x2 pivot (k+1 / k-1), and the row/column ranges
 * of the stored triangle. Everything else - the Bunch-Kaufman test, the
 * symmetric interchange, the rank-1 and rank-2 updates, both substitution
 * phases - is written once.
 *
 * The arithmetic follows the reference implementation operation for
 * operation (dsyr: temp = -r1*x_j then a_ij += x_i*temp; dger; dscal by a
 * reciprocal; dgemv('T') as a dot product subtracted afterwards, with an
 * explicit 4-way partial-sum order, see dot_rows), using plain operators only. With the compiler's default floating-point
 * contraction (GCC/Clang with an FMA target) the update loops become fused
 * multiply-adds, which is what makes reclu_sytf2_opt bit-identical to this
 * code; with -ffp-contract=off the two differ by rounding. The exception is
 * the dot product of the transposed solve (see dot_rows).
 *
 * Differences from netlib: dsyr's skip of zero x_j is not replicated (only
 * matters for non-finite input).
 *
 * Storage: column-major; only the UPLO triangle is referenced or written.
 * ipiv is 1-based: ipiv[k] = kp > 0 means a 1x1 pivot (row/col k swapped
 * with kp); ipiv[k] = ipiv[partner] = -kp a 2x2 pivot. info > 0: D(info,info)
 * is exactly zero (or NaN); the factorization is completed but D is singular. */

namespace {

struct MatrixView {
    double *data;
    int ld, rows, cols;
    __attribute__((always_inline)) double &operator()(int i, int j) const { return data[i + j * ld]; }
    __attribute__((always_inline)) double *col(int j) const { return data + j * ld; }
};

struct Range { int lo, hi; };                                /* [lo, hi) */

/* Fill-mode traits. dir is the step of the pivot walk and the offset of the
 * 2x2 partner; the ranges describe column j of the stored triangle. */
struct LowerFill {
    static constexpr int dir = 1;
    static int   first(int)           { return 0; }
    static Range off (int j, int n)   { return {j + 1, n}; }      /* stored part of column j below/above the diagonal */
    static Range diag(int j, int n)   { return {j, n}; }          /* same, including the diagonal */
    static Range between(int k, int imax) { return {k, imax}; }   /* columns of row imax from k toward imax, imax excluded */
};
struct UpperFill {
    static constexpr int dir = -1;
    static int   first(int n)         { return n - 1; }
    static Range off (int j, int)     { return {0, j}; }
    static Range diag(int j, int)     { return {0, j + 1}; }
    static Range between(int k, int imax) { return {imax + 1, k + 1}; }
};

constexpr double ALPHA = 0.6403882032022076;    /* (1 + sqrt(17)) / 8 */

/* index of the first max |x| over a range of elements x[lo*inc .. (hi-1)*inc]
 * (LAPACK idamax: a leading NaN is never beaten) */
inline int idamax(const double *x, Range r, int inc) {
    int im = r.lo; double dm = std::fabs(x[r.lo * inc]);
    for (int i = r.lo + 1; i < r.hi; i++) { const double d = std::fabs(x[i * inc]); if (d > dm) { dm = d; im = i; } }
    return im;
}

/* ------------------------------------------------------------ dsytf2 ---- */

/* Bunch-Kaufman pivot choice for step k. Returns kp and sets kstep; a zero
 * (or NaN) pivot column sets info and `zero`, and the step then does nothing. */
template <class Fill>
inline int choose_pivot(const MatrixView A, int k, int &kstep, bool &zero, int *info) {
    const int n = A.rows;
    kstep = 1; zero = false;
    const double absakk = std::fabs(A(k, k));
    const Range offk = Fill::off(k, n);
    int imax = k; double colmax = 0.0;
    if (offk.lo < offk.hi) { imax = idamax(A.col(k), offk, 1); colmax = std::fabs(A(imax, k)); }
    if (std::fmax(absakk, colmax) == 0.0 || std::isnan(absakk)) {
        if (*info == 0) *info = k + 1;                          /* D(k,k) is exactly zero: no interchange, no update */
        zero = true;
        return k;
    }
    if (absakk >= ALPHA * colmax) return k;                     /* 1x1, no interchange */
    /* largest off-diagonal element in row imax (between k and imax, strided) and in column imax (beyond it) */
    const int jmax = idamax(A.data + imax, Fill::between(k, imax), A.ld);
    double rowmax = std::fabs(A(imax, jmax));
    const Range offi = Fill::off(imax, n);
    if (offi.lo < offi.hi) {
        const int imax2 = idamax(A.col(imax), offi, 1);
        rowmax = std::fmax(rowmax, std::fabs(A(imax2, imax)));
    }
    if (absakk >= ALPHA * colmax * (colmax / rowmax)) return k;
    if (std::fabs(A(imax, imax)) >= ALPHA * rowmax) return imax; /* 1x1 with interchange */
    kstep = 2;                                                  /* 2x2 pivot (k, partner) with interchange */
    return imax;
}

/* symmetric interchange of rows/columns kk and kp within the trailing matrix */
template <class Fill>
inline void interchange(const MatrixView A, int k, int kk, int kp, int kstep) {
    const int n = A.rows;
    const Range r = Fill::off(kp, n);                          /* rows beyond kp: swap the two column segments */
    double *__restrict ckk = A.col(kk), *__restrict ckp = A.col(kp);
    #pragma omp simd
    for (int i = r.lo; i < r.hi; i++) { const double t = ckk[i]; ckk[i] = ckp[i]; ckp[i] = t; }
    const int lo = (kk < kp ? kk : kp) + 1, hi = (kk < kp ? kp : kk);   /* strictly between: column kk <-> row kp */
    for (int i = lo; i < hi; i++) { const double t = A(i, kk); A(i, kk) = A(kp, i); A(kp, i) = t; }
    double t = A(kk, kk); A(kk, kk) = A(kp, kp); A(kp, kp) = t;
    if (kstep == 2) { t = A(k + Fill::dir, k); A(k + Fill::dir, k) = A(kp, k); A(kp, k) = t; }
}

/* 1x1 pivot: A22 -= r1 x x^T on the stored triangle (dsyr), then x *= r1 */
template <class Fill>
inline void rank1_update(const MatrixView A, int k) {
    const int n = A.rows;
    const double r1 = 1.0 / A(k, k);
    const Range offk = Fill::off(k, n);
    const double *__restrict x = A.col(k);
    for (int j = offk.lo; j < offk.hi; j++) {
        const double temp = -r1 * x[j];
        double *__restrict cj = A.col(j);
        const Range r = Fill::diag(j, n);
        #pragma omp simd
        for (int i = r.lo; i < r.hi; i++) cj[i] += x[i] * temp;
    }
    double *__restrict xk = A.col(k);
    #pragma omp simd
    for (int i = offk.lo; i < offk.hi; i++) xk[i] *= r1;
}

/* 2x2 pivot (k, p = k + dir): A22 -= [x_k x_p] D^-1 [x_k x_p]^T. The pivot
 * columns are overwritten with the multipliers w_k, w_p as the columns are
 * consumed, so the walk must move away from the pivot (j = p+1, p+2, ... for
 * 'L', j = p-1, p-2, ... for 'U'): column j reads x_k, x_p only at rows on
 * its own side of j, which have not been written back yet. */
template <class Fill>
inline void rank2_update(const MatrixView A, int k) {
    const int n = A.rows, p = k + Fill::dir;
    const double dpk = A(p, k);
    const double cpp = A(p, p) / dpk, ckk = A(k, k) / dpk;
    const double t = 1.0 / (cpp * ckk - 1.0);
    const double d = t / dpk;
    const Range offp = Fill::off(p, n);
    for (int j = (Fill::dir > 0 ? offp.lo : offp.hi - 1); j >= offp.lo && j < offp.hi; j += Fill::dir) {
        const double wk = d * (cpp * A(j, k) - A(j, p));
        const double wp = d * (ckk * A(j, p) - A(j, k));
        const double *__restrict xk = A.col(k), *__restrict xp = A.col(p);
        double *__restrict cj = A.col(j);
        const Range r = Fill::diag(j, n);
        #pragma omp simd
        for (int i = r.lo; i < r.hi; i++) cj[i] = cj[i] - xk[i] * wk - xp[i] * wp;
        A(j, k) = wk; A(j, p) = wp;
    }
}

template <class Fill>
void sytf2(const MatrixView A, int *ipiv, int *info) {
    const int n = A.rows;
    for (int k = Fill::first(n); k >= 0 && k < n;) {
        int kstep; bool zero_pivot;
        const int kp = choose_pivot<Fill>(A, k, kstep, zero_pivot, info);
        const int kk = k + (kstep - 1) * Fill::dir;             /* row/column to interchange with kp */
        if (!zero_pivot) {
            if (kp != kk) interchange<Fill>(A, k, kk, kp, kstep);
            const Range rest = Fill::off(kk, n);
            if (kstep == 1) { if (rest.lo < rest.hi) rank1_update<Fill>(A, k); }
            else            { if (rest.lo < rest.hi) rank2_update<Fill>(A, k); }
        }
        if (kstep == 1) ipiv[k] = kp + 1;
        else { ipiv[k] = -(kp + 1); ipiv[k + Fill::dir] = -(kp + 1); }
        k += kstep * Fill::dir;
    }
}

/* ------------------------------------------------------------ dsytrs ---- */

inline void swap_rows(const MatrixView B, int r1, int r2) {
    if (r1 == r2) return;
    for (int j = 0; j < B.cols; j++) { const double t = B(r1, j); B(r1, j) = B(r2, j); B(r2, j) = t; }
}

/* B(rows, :) -= A(rows, c) * B(k, :)  (dger with column c of the factor) */
inline void axpy_rows(const MatrixView A, int c, const MatrixView B, Range r, int k) {
    const double *__restrict x = A.col(c);
    for (int j = 0; j < B.cols; j++) {
        const double bk = B(k, j);
        double *__restrict bj = B.col(j);
        #pragma omp simd
        for (int i = r.lo; i < r.hi; i++) bj[i] -= x[i] * bk;
    }
}

/* B(k, :) -= A(rows, c)^T B(rows, :)  (dgemv('T')) */
inline void dot_rows(const MatrixView A, int c, const MatrixView B, Range r, int k) {
    const double *__restrict x = A.col(c);
    for (int j = 0; j < B.cols; j++) {
        const double *__restrict bj = B.col(j);
        double t = 0.0;
        #pragma omp simd reduction(+: t)
        for (int i = r.lo; i < r.hi; i++) t += x[i] * bj[i];
        B(k, j) -= t;
    }
}

template <class Fill>
void sytrs(const MatrixView A, const int *ipiv, const MatrixView B) {
    const int n = A.rows;
    /* P L D y = b: pivots in factorization order */
    for (int k = Fill::first(n); k >= 0 && k < n;) {
        if (ipiv[k] > 0) {
            swap_rows(B, k, ipiv[k] - 1);
            axpy_rows(A, k, B, Fill::off(k, n), k);
            const double r = 1.0 / A(k, k);
            for (int j = 0; j < B.cols; j++) B(k, j) *= r;
            k += Fill::dir;
        } else {
            const int p = k + Fill::dir;                        /* the block is (k, p) */
            swap_rows(B, p, -ipiv[k] - 1);
            axpy_rows(A, k, B, Fill::off(p, n), k);
            axpy_rows(A, p, B, Fill::off(p, n), p);
            const int r1 = k < p ? k : p, r2 = k < p ? p : k;  /* rows of the block in storage order */
            const double akm1k = A(p, k);                       /* the stored off-diagonal element of the block */
            const double akm1 = A(r1, r1) / akm1k, ak = A(r2, r2) / akm1k, denom = akm1 * ak - 1.0;
            for (int j = 0; j < B.cols; j++) {
                const double bkm1 = B(r1, j) / akm1k, bk = B(r2, j) / akm1k;
                B(r1, j) = (ak * bkm1 - bk) / denom;
                B(r2, j) = (akm1 * bk - bkm1) / denom;
            }
            k += 2 * Fill::dir;
        }
    }
    /* L^T x = y, then P^T: pivots in reverse order; a 2x2 block is entered
     * at its row nearer the end of the walk, its partner is k - dir */
    for (int k = (Fill::dir > 0 ? n - 1 : 0); k >= 0 && k < n;) {
        const Range r = Fill::off(k, n);
        if (ipiv[k] > 0) {
            dot_rows(A, k, B, r, k);
            swap_rows(B, k, ipiv[k] - 1);
            k -= Fill::dir;
        } else {
            const int q = k - Fill::dir;
            dot_rows(A, k, B, r, k);
            dot_rows(A, q, B, r, q);
            swap_rows(B, k, -ipiv[k] - 1);
            k -= 2 * Fill::dir;
        }
    }
}
} // namespace

void reclu_sytf2(char uplo, int n, double *A, int lda, int *ipiv, int *info) {
    *info = 0;
    const bool upper = (uplo == 'U' || uplo == 'u');
    if (!upper && !(uplo == 'L' || uplo == 'l')) { *info = -1; return; }
    if (n < 0) { *info = -2; return; }
    if (lda < (n < 1 ? 1 : n)) { *info = -4; return; }
    if (n == 0) return;
    const MatrixView Av{A, lda, n, n};
    if (upper) sytf2<UpperFill>(Av, ipiv, info); else sytf2<LowerFill>(Av, ipiv, info);
}

void reclu_sytrs(char uplo, int n, int nrhs, const double *A, int lda, const int *ipiv, double *B, int ldb) {
    if (n <= 0 || nrhs <= 0) return;
    const MatrixView Av{const_cast<double *>(A), lda, n, n}, Bv{B, ldb, n, nrhs};
    if (uplo == 'U' || uplo == 'u') sytrs<UpperFill>(Av, ipiv, Bv); else sytrs<LowerFill>(Av, ipiv, Bv);
}
