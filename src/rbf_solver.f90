! rbf_solver -- the Fortran face of rbf_solver.h: iterative solution of
! the assembled sparse systems through Eigen, built into the optional
! target rbf_solver (see docs/solver.md).
!
!     use rbf_solver
!     status = solve_sparse(n, nnz, val, ia, ja, b, x, res_error, res_iter)
!     status = solve_sparse(n, nnz, val, ia, ja, b, x, method=RBF_SOLVER_CONJUGATE_GRADIENT, &
!                           precond=RBF_PRECOND_INCOMPLETE, tolerance=1.0e-10_c_double)
!
! The CSR arrays are 0-based, as the assembly leaves them; x is the
! initial guess on input and the solution on output. The trailing
! arguments are optional and default the way the C header says. The
! generic solve_sparse also takes an operator given by its
! matrix-vector product, a bind(c) procedure of the interface
! rbf_matvec_t with a context pointer passed through unchanged:
!
!     status = solve_sparse(n, n, matvec, c_loc(ctx), b, x, res_error, res_iter)
!
! Every status is one of the RBF_SOLVER_* enumerators below. csr_mv is
! Eigen's product of a CSR matrix with a vector, y := beta*y + alpha*A*x,
! for the residuals and time steps around a solve.
module rbf_solver
use, intrinsic :: iso_c_binding, only: c_int, c_double, c_ptr
implicit none
private

public :: solve_sparse, csr_mv
public :: rbf_solve_sparse_csr_dp, rbf_solve_sparse_mf_dp, rbf_csr_mv_dp
public :: rbf_matvec_t

! The Krylov method. Conjugate gradient assumes a symmetric positive
! definite matrix; BiCGSTAB takes any nonsingular one.
enum, bind(c)
    enumerator :: RBF_SOLVER_BICGSTAB = 0
    enumerator :: RBF_SOLVER_CONJUGATE_GRADIENT = 1
end enum
public :: RBF_SOLVER_BICGSTAB, RBF_SOLVER_CONJUGATE_GRADIENT

! The preconditioner of the CSR solver; INCOMPLETE is ILUT for BiCGSTAB
! and incomplete Cholesky for conjugate gradient.
enum, bind(c)
    enumerator :: RBF_PRECOND_IDENTITY = 0
    enumerator :: RBF_PRECOND_DIAGONAL = 1
    enumerator :: RBF_PRECOND_INCOMPLETE = 2
end enum
public :: RBF_PRECOND_IDENTITY, RBF_PRECOND_DIAGONAL, RBF_PRECOND_INCOMPLETE

! The status the solvers return; NO_CONVERGENCE leaves the last iterate
! in x.
enum, bind(c)
    enumerator :: RBF_SOLVER_SUCCESS = 0
    enumerator :: RBF_SOLVER_NUMERICAL_ISSUE = 1
    enumerator :: RBF_SOLVER_NO_CONVERGENCE = 2
    enumerator :: RBF_SOLVER_INVALID_INPUT = 3
end enum
public :: RBF_SOLVER_SUCCESS, RBF_SOLVER_NUMERICAL_ISSUE
public :: RBF_SOLVER_NO_CONVERGENCE, RBF_SOLVER_INVALID_INPUT

abstract interface
    ! The matrix-vector product of the matrix-free solver, a BLAS
    ! level-2 update y := y + alpha A x on an nr-by-nc operator; data
    ! is the context pointer given to the solver.
    subroutine rbf_matvec_t(nr, nc, alpha, x, y, data) bind(c)
        import c_int, c_double, c_ptr
        integer(c_int), value :: nr, nc
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(nc)
        real(c_double), intent(inout) :: y(nr)
        type(c_ptr), value :: data
    end subroutine
end interface

interface csr_mv

    ! y := beta*y + alpha*A*x with A in compressed sparse row form,
    ! 0-based; y is not read when beta is zero. Threaded over the rows
    ! when Eigen was built with OpenMP.
    subroutine rbf_csr_mv_dp(nr, nc, nnz, val, ia, ja, alpha, x, beta, y) bind(c)
        import c_int, c_double
        integer(c_int), value :: nr, nc, nnz
        real(c_double), intent(in) :: val(nnz)
        integer(c_int), intent(in) :: ia(nr + 1), ja(nnz)
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(nc)
        real(c_double), value :: beta
        real(c_double), intent(inout) :: y(nr)
    end subroutine

end interface

interface solve_sparse

    ! A x = b with A in compressed sparse row form, 0-based; the
    ! preconditioner defaults to DIAGONAL.
    function rbf_solve_sparse_csr_dp(n, nnz, val, ia, ja, b, x, &
            res_error, res_iter, method, precond, max_iter, tolerance) bind(c)
        import c_int, c_double
        integer(c_int), value :: n, nnz
        real(c_double), intent(in) :: val(nnz)
        integer(c_int), intent(in) :: ia(n + 1), ja(nnz)
        real(c_double), intent(in) :: b(n)
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out), optional :: res_error
        integer(c_int), intent(out), optional :: res_iter
        integer(c_int), intent(in), optional :: method     ! RBF_SOLVER_*, default BICGSTAB
        integer(c_int), intent(in), optional :: precond    ! RBF_PRECOND_*, default DIAGONAL
        integer(c_int), intent(in), optional :: max_iter   ! default 2 * n
        real(c_double), intent(in), optional :: tolerance  ! default epsilon(1.0_c_double)
        integer(c_int) :: rbf_solve_sparse_csr_dp
    end function

    ! A x = b with A given by its matrix-vector product mv, which
    ! receives data on every call; c_null_ptr when it needs none. The
    ! Krylov methods need nr == nc.
    function rbf_solve_sparse_mf_dp(nr, nc, mv, data, b, x, &
            res_error, res_iter, method, max_iter, tolerance) bind(c)
        import c_int, c_double, c_ptr, rbf_matvec_t
        integer(c_int), value :: nr, nc
        procedure(rbf_matvec_t) :: mv
        type(c_ptr), value :: data
        real(c_double), intent(in) :: b(nr)
        real(c_double), intent(inout) :: x(nc)
        real(c_double), intent(out), optional :: res_error
        integer(c_int), intent(out), optional :: res_iter
        integer(c_int), intent(in), optional :: method     ! RBF_SOLVER_*, default BICGSTAB
        integer(c_int), intent(in), optional :: max_iter   ! default 2 * nc
        real(c_double), intent(in), optional :: tolerance  ! default epsilon(1.0_c_double)
        integer(c_int) :: rbf_solve_sparse_mf_dp
    end function

end interface

end module
