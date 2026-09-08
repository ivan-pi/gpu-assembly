/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#pragma once
/* Small dense LU with partial pivoting and the matching solve, tuned for the
 * stencil systems of RBF-FD (n ~ 10..100). Column-major, LAPACK conventions:
 *   - dgetrf layout: unit-lower L and U overwrite A, ipiv is 1-based,
 *     info > 0 marks the first exactly-zero pivot;
 *   - dgetrs(N):     B is overwritten with X = A^{-1} B.
 *
 * reclu_lu / reclu_solve_naive are the plain-loop references (the algorithm).
 * reclu_lu_opt / reclu_solve_opt are the AVX2+FMA kernels; on other targets
 * they compile to the naive code. reclu_solve_asm is reclu_solve_opt with its
 * two streaming loops in inline asm (pins instruction selection; +2..5%).
 *
 * Fast paths need padded storage: lda % 4 == 0 and lda >= roundup4(m).
 *   reclu_lu_opt then works in place and ZEROES rows m..roundup4(m)-1 of every
 *   column (they take part in the vector tiles); otherwise it copies through
 *   a scratch buffer. reclu_solve_opt needs those zero pad rows and falls back
 *   to the naive kernel if lda is not padded. Allocate matrices 32-byte
 *   aligned: split 256-bit stores cost ~35% in these store-bound kernels.
 * Limits: m <= RECLU_MAXM; sizes below RECLU_SMALL use the naive LU. */
#define RECLU_MAXM  128
#define RECLU_SMALL 20

#if defined(__GNUC__)
#define RECLU_API __attribute__((visibility("default")))
#else
#define RECLU_API
#endif
#ifdef __cplusplus
extern "C" {
#endif
RECLU_API void reclu_lu         (int m, int n, double *A, int lda, int *ipiv, int *info);
RECLU_API void reclu_lu_opt     (int m, int n, double *A, int lda, int *ipiv, int *info);
RECLU_API void reclu_solve_naive(int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb);
RECLU_API void reclu_solve_opt  (int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb);
RECLU_API void reclu_solve_asm  (int m, int nrhs, const double *LU, int lda, const int *ipiv, double *B, int ldb);

/* Symmetric indefinite: LDL^T with Bunch-Kaufman pivoting (dsytf2) and the
 * matching solve (dsytrs). uplo = 'U' or 'L'; only that triangle is read or
 * written. ipiv 1-based, negative entries mark 2x2 pivots (LAPACK layout).
 * reclu_sytf2/reclu_sytrs are the plain-loop references; the _opt kernels
 * need the same padded/aligned storage as reclu_lu_opt (lda % 4 == 0,
 * lda >= roundup4(n)); for 'L' rows n..roundup4(n)-1 are zeroed, for 'U'
 * they are read but left unchanged. */
RECLU_API void reclu_sytf2    (char uplo, int n, double *A, int lda, int *ipiv, int *info);
RECLU_API void reclu_sytf2_opt(char uplo, int n, double *A, int lda, int *ipiv, int *info);
RECLU_API void reclu_sytrs    (char uplo, int n, int nrhs, const double *A, int lda, const int *ipiv, double *B, int ldb);
RECLU_API void reclu_sytrs_opt(char uplo, int n, int nrhs, const double *A, int lda, const int *ipiv, double *B, int ldb);
#ifdef __cplusplus
}
#endif
