! rbf_solver -- the Fortran face of rbf_solver.h: iterative solution of
! the assembled sparse systems through Eigen, built into the optional
! target rbf_solver (see docs/solver.md).
!
!     use rbf_solver
!     status = solve_sparse(n, nnz, val, ia, ja, b, x, res_error, res_iter)
!     status = solve_sparse(n, nnz, val, ia, ja, b, x, method=SOLVER_CG, &
!                           precond=PRECOND_ILU, tolerance=1.0e-10_c_double)
!
! The CSR arrays are 0-based, as the assembly leaves them; x is the
! initial guess on input and the solution on output. The trailing
! arguments are optional and default the way the C header says. The
! generic solve_sparse also takes an operator given by its
! matrix-vector product, a bind(c) procedure of the interface matvec_t
! with a context pointer passed through unchanged:
!
!     status = solve_sparse(n, n, my_matvec, c_loc(ctx), b, x, res_error, res_iter)
!
! Every status is one of the SOLVER_* enumerators below. csr_mv is
! Eigen's product of a CSR matrix with a vector, y := beta*y + alpha*A*x,
! for the residuals and time steps around a solve.
!
! The C functions behind the generics carry the library's rbf_ prefix,
! as global symbols for the linker; the module keeps them private, so
! the prefix never appears in a Fortran caller.
module rbf_solver
use, intrinsic :: iso_c_binding, only: c_int, c_double, c_ptr
implicit none
private

public :: solve_sparse, csr_mv, matvec_t

! The Krylov method. Conjugate gradient assumes a symmetric positive
! definite matrix; BiCGSTAB takes any nonsingular one.
enum, bind(c)
    enumerator :: SOLVER_BICGSTAB = 0
    enumerator :: SOLVER_CG = 1
end enum
public :: SOLVER_BICGSTAB, SOLVER_CG

! The preconditioner of the CSR solver; ILU is the incomplete
! factorization that suits the method, ILUT for BiCGSTAB and
! incomplete Cholesky for conjugate gradient.
enum, bind(c)
    enumerator :: PRECOND_NONE = 0
    enumerator :: PRECOND_JACOBI = 1
    enumerator :: PRECOND_ILU = 2
end enum
public :: PRECOND_NONE, PRECOND_JACOBI, PRECOND_ILU

! The status the solvers return; NO_CONVERGENCE leaves the last iterate
! in x.
enum, bind(c)
    enumerator :: SOLVER_SUCCESS = 0
    enumerator :: SOLVER_NUMERICAL_ISSUE = 1
    enumerator :: SOLVER_NO_CONVERGENCE = 2
    enumerator :: SOLVER_INVALID_INPUT = 3
end enum
public :: SOLVER_SUCCESS, SOLVER_NUMERICAL_ISSUE
public :: SOLVER_NO_CONVERGENCE, SOLVER_INVALID_INPUT

abstract interface
    ! The matrix-vector product of the matrix-free solver, a BLAS
    ! level-2 update y := y + alpha A x on an nr-by-nc operator; data
    ! is the context pointer given to the solver.
    subroutine matvec_t(nr, nc, alpha, x, y, data) bind(c)
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
    ! preconditioner defaults to PRECOND_JACOBI.
    function rbf_solve_csr_dp(n, nnz, val, ia, ja, b, x, &
            res_error, res_iter, method, precond, max_iter, tolerance) bind(c)
        import c_int, c_double
        integer(c_int), value :: n, nnz
        real(c_double), intent(in) :: val(nnz)
        integer(c_int), intent(in) :: ia(n + 1), ja(nnz)
        real(c_double), intent(in) :: b(n)
        real(c_double), intent(inout) :: x(n)
        real(c_double), intent(out), optional :: res_error
        integer(c_int), intent(out), optional :: res_iter
        integer(c_int), intent(in), optional :: method     ! SOLVER_*, default BICGSTAB
        integer(c_int), intent(in), optional :: precond    ! PRECOND_*, default JACOBI
        integer(c_int), intent(in), optional :: max_iter   ! default 2 * n
        real(c_double), intent(in), optional :: tolerance  ! default epsilon(1.0_c_double)
        integer(c_int) :: rbf_solve_csr_dp
    end function

    ! A x = b with A given by its matrix-vector product mv, which
    ! receives data on every call; c_null_ptr when it needs none. The
    ! Krylov methods need nr == nc.
    function rbf_solve_mf_dp(nr, nc, mv, data, b, x, &
            res_error, res_iter, method, max_iter, tolerance) bind(c)
        import c_int, c_double, c_ptr, matvec_t
        integer(c_int), value :: nr, nc
        procedure(matvec_t) :: mv
        type(c_ptr), value :: data
        real(c_double), intent(in) :: b(nr)
        real(c_double), intent(inout) :: x(nc)
        real(c_double), intent(out), optional :: res_error
        integer(c_int), intent(out), optional :: res_iter
        integer(c_int), intent(in), optional :: method     ! SOLVER_*, default BICGSTAB
        integer(c_int), intent(in), optional :: max_iter   ! default 2 * nc
        real(c_double), intent(in), optional :: tolerance  ! default epsilon(1.0_c_double)
        integer(c_int) :: rbf_solve_mf_dp
    end function

end interface

end module
