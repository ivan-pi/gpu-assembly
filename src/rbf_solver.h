#ifndef RBF_SOLVER_H
#define RBF_SOLVER_H

/*
 * rbf_solver.h -- iterative solution of the assembled sparse systems.
 *
 * A C interface over Eigen's iterative solvers (BiCGSTAB and conjugate
 * gradient, with no, Jacobi and incomplete-factorization
 * preconditioning), callable from C and C++ directly and from Fortran
 * through the module rbf_solver in rbf_solver.f90. Eigen stays behind
 * this header: it is built into the optional target rbf_solver
 * (-DGPU_ASSEMBLY_ENABLE_SOLVER=ON, Eigen 3.4 or newer) and no caller
 * needs it. See docs/solver.md.
 *
 * Names: the three functions carry the library's rbf_ prefix because
 * they are global symbols for the linker, as the other bind(c) entry
 * points of the library are. The enumerators are scoped by their
 * category instead, SOLVER_ and PRECOND_, and Fortran reaches the
 * functions through the generics solve_sparse and csr_mv, so a Fortran
 * caller never types the prefix.
 *
 * Two solvers. rbf_solve_csr_dp takes the matrix in CSR, the storage
 * the assembly produces (rbf::make_row_ptr in rbf_reorder.h gives the
 * row pointer of a fixed-width stencil graph), and rbf_solve_mf_dp
 * takes a matrix-vector product callback, for an operator that is
 * never formed, or that is applied on the device. rbf_csr_mv_dp is
 * Eigen's CSR matrix-vector product on its own, for the residuals and
 * time steps around a solve.
 *
 * Both solvers return a solver_status, and the optional arguments (the
 * ones documented as such) may be NULL, in which case Eigen's defaults
 * apply: BiCGSTAB, the Jacobi preconditioner, at most 2n iterations,
 * and a tolerance of machine epsilon.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* The Krylov method. Conjugate gradient assumes a symmetric positive
 * definite matrix; BiCGSTAB takes any nonsingular one. */
typedef enum { SOLVER_BICGSTAB = 0, SOLVER_CG = 1 } solver_method;

/* The preconditioner; CSR systems only, the matrix-free solvers run
 * unpreconditioned. ILU is the incomplete factorization that suits
 * the method: ILUT for BiCGSTAB, incomplete Cholesky for conjugate
 * gradient. */
typedef enum { PRECOND_NONE = 0, PRECOND_JACOBI = 1, PRECOND_ILU = 2 } solver_precond;

/* The status the solvers return; the values are Eigen's
 * ComputationInfo. NO_CONVERGENCE means the iteration limit was reached
 * with the residual still above the tolerance, and x holds the last
 * iterate. INVALID_INPUT covers a NULL array, a nonpositive size, a
 * row pointer that disagrees with nnz, and an enumerator outside the
 * two enums above. */
typedef enum {
    SOLVER_SUCCESS = 0,
    SOLVER_NUMERICAL_ISSUE = 1,
    SOLVER_NO_CONVERGENCE = 2,
    SOLVER_INVALID_INPUT = 3
} solver_status;

/*
 * The matrix-vector product of the matrix-free solver, in the shape of
 * a BLAS level-2 update:
 *
 *     y := y + alpha * A * x
 *
 * nr, nc   rows and columns of A, as passed to the solver
 * alpha    scaling of the product
 * x        input, length nc
 * y        accumulated into, length nr
 * data     the context pointer passed to the solver, unchanged
 */
typedef void (*rbf_matvec)(int nr, int nc, double alpha, const double* x, double* y, void* data);

/*
 * y := beta * y + alpha * A * x, with A in compressed sparse row form,
 * 0-based. Eigen's product, threaded with OpenMP over the rows when
 * the library was built with it. y is not read when beta is zero.
 *
 * nr, nc      rows and columns of A
 * nnz         number of stored entries; ia[nr] must equal it
 * val         the entries, length nnz
 * ia          row pointer, length nr + 1
 * ja          column indices, length nnz
 * alpha       scaling of the product
 * x           input, length nc
 * beta        scaling of y on input
 * y           updated, length nr
 */
void rbf_csr_mv_dp(int nr,
                   int nc,
                   int nnz,
                   const double* val,
                   const int* ia,
                   const int* ja,
                   double alpha,
                   const double* x,
                   double beta,
                   double* y);

/*
 * Solve A x = b with A in compressed sparse row form, 0-based.
 *
 * n           order of the matrix
 * nnz         number of stored entries; ia[n] must equal it
 * val         the entries, length nnz
 * ia          row pointer, length n + 1
 * ja          column indices, length nnz
 * b           right-hand side, length n
 * x           initial guess on input, the solution on output, length n
 * res_error   (optional) the relative residual |A x - b| / |b| reached
 * res_iter    (optional) iterations taken
 * method      (optional) solver_method, default SOLVER_BICGSTAB
 * precond     (optional) solver_precond, default PRECOND_JACOBI
 * max_iter    (optional) iteration limit, default 2 * n
 * tolerance   (optional) relative residual to stop at, default epsilon
 *
 * Returns a solver_status. The arrays are wrapped in place; the solver
 * copies nothing but what its preconditioner keeps.
 */
int rbf_solve_csr_dp(int n,
                     int nnz,
                     const double* val,
                     const int* ia,
                     const int* ja,
                     const double* b,
                     double* x,
                     double* res_error,
                     int* res_iter,
                     const solver_method* method,
                     const solver_precond* precond,
                     const int* max_iter,
                     const double* tolerance);

/*
 * Solve A x = b with A given by its matrix-vector product.
 *
 * nr, nc      rows and columns of A; the Krylov methods need nr == nc
 * mv          the product, see rbf_matvec
 * data        handed to mv on every call, may be NULL
 * b           right-hand side, length nr
 * x           initial guess on input, the solution on output, length nc
 * res_error   (optional) the relative residual |A x - b| / |b| reached
 * res_iter    (optional) iterations taken
 * method      (optional) solver_method, default SOLVER_BICGSTAB
 * max_iter    (optional) iteration limit, default 2 * nc
 * tolerance   (optional) relative residual to stop at, default epsilon
 *
 * Returns a solver_status.
 */
int rbf_solve_mf_dp(int nr,
                    int nc,
                    rbf_matvec mv,
                    void* data,
                    const double* b,
                    double* x,
                    double* res_error,
                    int* res_iter,
                    const solver_method* method,
                    const int* max_iter,
                    const double* tolerance);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RBF_SOLVER_H */
