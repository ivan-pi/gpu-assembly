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
#include <limits>

namespace {

struct MatrixView {
    double *data;
    int lda, rows, cols;
    __attribute__((always_inline)) double &operator()(int i, int j) const { return data[i + j * lda]; }
    __attribute__((always_inline)) double *col(int j) const { return data + j * lda; }
};

inline int find_pivot_row(const MatrixView A, int k) {
    const int m = A.rows;
    int kp = k;
    double pmax = std::fabs(A(k, k));
    for (int i = k + 1; i < m; i++) {
        const double aik = std::fabs(A(i, k));
        if (aik > pmax) { pmax = aik; kp = i; }
    }
    return kp;
}

inline void swap_rows(const MatrixView A, int k, int kp) {
    const int n = A.cols;
    for (int j = 0; j < n; j++) {
        const double tmp = A(k, j);
        A(k, j) = A(kp, j);
        A(kp, j) = tmp;
    }
}

inline void scale_column(const MatrixView A, int k, double Akk) {
    const int m = A.rows;
    double *__restrict colk = A.col(k);
    constexpr double sfmin = std::numeric_limits<double>::min();
    if (std::fabs(Akk) >= sfmin) {
        const double Akkinv = 1.0 / Akk;
        #pragma omp simd
        for (int i = k + 1; i < m; i++) colk[i] *= Akkinv;
    } else {
        #pragma omp simd
        for (int i = k + 1; i < m; i++) colk[i] /= Akk;
    }
}

inline void update_trailing(const MatrixView A, int k) {
    const int m = A.rows;
    const int n = A.cols;
    const double *__restrict pivcol = A.col(k);
    for (int j = k + 1; j < n; j++) {
        const double akj = A(k, j);
        double *__restrict colj = A.col(j);
        #pragma omp simd
        for (int i = k + 1; i < m; i++) colj[i] -= pivcol[i] * akj;
    }
}

} // namespace

void reclu_lu(int m, int n, double *Aptr, int lda, int *ipiv, int *info) {
    *info = 0;
    if (m < 0)                          *info = -1;
    else if (n < 0)                     *info = -2;
    else if (lda < (m < 1 ? 1 : m))     *info = -4;
    if (*info != 0) return;
    if (m == 0 || n == 0) return;

    const MatrixView A{Aptr, lda, m, n};
    const int minmn = m < n ? m : n;

    for (int k = 0; k < minmn; k++) {
        const int kp = find_pivot_row(A, k);
        ipiv[k] = kp + 1;
        const double Akk = A(kp, k);
        if (Akk != 0.0) {
            if (kp != k) swap_rows(A, k, kp);
            if (k < m - 1) scale_column(A, k, Akk);
        } else if (*info == 0) {
            *info = k + 1;
        }
        if (k < minmn - 1) update_trailing(A, k);
    }
}
