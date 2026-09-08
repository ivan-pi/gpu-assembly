! Tests for the module rbf_solver: the Fortran face of rbf_solver.h,
! exercised the way a Fortran driver would call it -- the generic
! solve_sparse with optional arguments present and absent, and the
! matrix-free form with a bind(c) callback and a context pointer. The
! system is the one-dimensional three-point Laplacian with Dirichlet
! boundary, symmetric positive definite, and the right-hand side is
! manufactured from a known solution.
!
! Built and run with the C++ test through CMake:
!
!   cmake -B build -DGPU_ASSEMBLY_ENABLE_SOLVER=ON
!   cmake --build build
!   ctest --test-dir build

! The matrix, its product, and the callbacks. Module procedures rather
! than internal ones: a bind(c) callback must not need a trampoline.
module test_solver_support
use, intrinsic :: iso_c_binding, only: c_int, c_double, c_ptr, c_f_pointer
use rbf_ellpack, only: ellpack_mv
implicit none
private

public :: csr, laplacian_1d, apply, matvec, identity
public :: ell, to_ellpack, ellpack_matvec
public :: check, failures

! In CSR, 0-based, handed to the callback through a pointer
type :: csr
    integer(c_int) :: n
    integer(c_int), allocatable :: ia(:), ja(:)
    real(c_double), allocatable :: val(:)
end type

! The same in ELLPACK, a(n, nnzrow), for the rbf_ellpack product
type :: ell
    integer(c_int) :: n, nnzrow
    integer(c_int), allocatable :: ja(:, :)
    real(c_double), allocatable :: a(:, :)
end type

integer :: failures = 0

contains

    subroutine check(cond, what)
        logical, intent(in) :: cond
        character(*), intent(in) :: what
        if (.not. cond) then
            print '(a,a)', "FAIL ", what
            failures = failures + 1
        end if
    end subroutine

    ! The three-point Laplacian -u'' with Dirichlet boundary
    subroutine laplacian_1d(n, A)
        integer, intent(in) :: n
        type(csr), intent(out) :: A
        integer :: i, p
        A%n = n
        allocate(A%ia(n + 1), A%ja(3*n - 2), A%val(3*n - 2))
        p = 0
        A%ia(1) = 0
        do i = 1, n
            if (i > 1) then
                p = p + 1
                A%ja(p) = i - 2
                A%val(p) = -1
            end if
            p = p + 1
            A%ja(p) = i - 1
            A%val(p) = 2
            if (i < n) then
                p = p + 1
                A%ja(p) = i
                A%val(p) = -1
            end if
            A%ia(i + 1) = p
        end do
    end subroutine

    ! y := y + alpha A x
    subroutine apply(A, alpha, x, y)
        type(csr), intent(in) :: A
        real(c_double), intent(in) :: alpha, x(:)
        real(c_double), intent(inout) :: y(:)
        integer :: i, p
        do i = 1, A%n
            do p = A%ia(i) + 1, A%ia(i + 1)
                y(i) = y(i) + alpha*A%val(p)*x(A%ja(p) + 1)
            end do
        end do
    end subroutine

    ! The product of the matrix behind data, of the interface matvec_t
    subroutine matvec(nr, nc, alpha, x, y, data) bind(c)
        integer(c_int), value :: nr, nc
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(nc)
        real(c_double), intent(inout) :: y(nr)
        type(c_ptr), value :: data
        type(csr), pointer :: A
        call c_f_pointer(data, A)
        call check(nr == A%n .and. nc == A%n, "matvec: dimensions")
        call apply(A, alpha, x, y)
    end subroutine

    ! The tridiagonal matrix as ELLPACK with three entries per row; the
    ! end rows repeat the diagonal index with a zero entry
    subroutine to_ellpack(A, E)
        type(csr), intent(in) :: A
        type(ell), intent(out) :: E
        integer :: i, j, p
        E%n = A%n
        E%nnzrow = 3
        allocate(E%ja(A%n, 3), E%a(A%n, 3))
        do i = 1, A%n
            E%ja(i, :) = i - 1
            E%a(i, :) = 0
            j = 0
            do p = A%ia(i) + 1, A%ia(i + 1)
                j = j + 1
                E%ja(i, j) = A%ja(p)
                E%a(i, j) = A%val(p)
            end do
        end do
    end subroutine

    ! rbf_ellpack's product as the operator of the matrix-free solver
    subroutine ellpack_matvec(nr, nc, alpha, x, y, data) bind(c)
        integer(c_int), value :: nr, nc
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(nc)
        real(c_double), intent(inout) :: y(nr)
        type(c_ptr), value :: data
        type(ell), pointer :: E
        call c_f_pointer(data, E)
        call check(nr == E%n .and. nc == E%n, "ellpack_matvec: dimensions")
        call ellpack_mv(E%n, E%nnzrow, alpha, E%a, E%ja, E%n, x, 1.0_c_double, y)
    end subroutine

    ! The identity, needing no context
    subroutine identity(nr, nc, alpha, x, y, data) bind(c)
        integer(c_int), value :: nr, nc
        real(c_double), value :: alpha
        real(c_double), intent(in) :: x(nc)
        real(c_double), intent(inout) :: y(nr)
        type(c_ptr), value :: data
        y = y + alpha*x
    end subroutine

end module

program test_solver
use, intrinsic :: iso_c_binding, only: c_int, c_double, c_loc, c_null_ptr
use rbf_solver
use test_solver_support
implicit none

integer, parameter :: n = 200
type(csr), target :: A
type(ell), target :: E
real(c_double) :: u(n), b(n), x(n), y(n), err
integer(c_int) :: status, iter
integer :: i

call laplacian_1d(n, A)
do i = 1, n
    u(i) = sin(0.05_c_double*i) + 0.01_c_double*i
end do
b = 0
call apply(A, 1.0_c_double, u, b)

! Every optional argument absent: BiCGSTAB with the Jacobi preconditioner
x = 0
status = solve_sparse(A%n, size(A%val), A%val, A%ia, A%ja, b, x)
call check(status == SOLVER_SUCCESS, "csr defaults: status")
call check(maxval(abs(x - u)) < 1.0e-8_c_double, "csr defaults: solution")

! Conjugate gradient with incomplete Cholesky, keywords, the results asked for
x = 0
status = solve_sparse(A%n, size(A%val), A%val, A%ia, A%ja, b, x, &
    res_error=err, res_iter=iter, method=SOLVER_CG, &
    precond=PRECOND_ILU, tolerance=1.0e-12_c_double)
call check(status == SOLVER_SUCCESS, "csr cg/ichol: status")
call check(err >= 0 .and. err <= 1.0e-12_c_double, "csr cg/ichol: residual")
! Eigen counts completed passes, so an exact preconditioner reports 0
call check(iter >= 0 .and. iter <= 2*n, "csr cg/ichol: iterations")
call check(maxval(abs(x - u)) < 1.0e-9_c_double, "csr cg/ichol: solution")

! Too few iterations: NO_CONVERGENCE, with the count reported
x = 0
status = solve_sparse(A%n, size(A%val), A%val, A%ia, A%ja, b, x, &
    res_iter=iter, max_iter=1, tolerance=1.0e-14_c_double)
call check(status == SOLVER_NO_CONVERGENCE, "csr limit: status")
call check(iter == 1, "csr limit: iterations")

! Matrix-free, the matrix reaching the callback through the context pointer
x = 0
status = solve_sparse(n, n, matvec, c_loc(A), b, x, res_error=err, res_iter=iter, &
    method=SOLVER_CG, tolerance=1.0e-12_c_double)
call check(status == SOLVER_SUCCESS, "mf cg: status")
call check(err >= 0 .and. err <= 1.0e-12_c_double, "mf cg: residual")
call check(maxval(abs(x - u)) < 1.0e-9_c_double, "mf cg: solution")

! Matrix-free with the ELLPACK product of rbf_ellpack as the operator
call to_ellpack(A, E)
x = 0
status = solve_sparse(n, n, ellpack_matvec, c_loc(E), b, x, res_error=err, &
    tolerance=1.0e-12_c_double)
call check(status == SOLVER_SUCCESS, "mf ellpack: status")
call check(maxval(abs(x - u)) < 1.0e-9_c_double, "mf ellpack: solution")

! Eigen's CSR product against the reference: y = A u, then y = -0.5 y + 2 A u
y = huge(1.0_c_double)
call csr_mv(A%n, A%n, size(A%val), A%val, A%ia, A%ja, 1.0_c_double, u, 0.0_c_double, y)
call check(maxval(abs(y - b)) < 1.0e-14_c_double, "csr_mv: y = A x")
y = 1
call csr_mv(A%n, A%n, size(A%val), A%val, A%ia, A%ja, 2.0_c_double, u, -0.5_c_double, y)
call check(maxval(abs(y - (2*b - 0.5_c_double))) < 1.0e-14_c_double, "csr_mv: beta, alpha")

! The callback without a context: the identity, so x = b
x = 0
status = solve_sparse(n, n, identity, c_null_ptr, b, x, res_iter=iter)
call check(status == SOLVER_SUCCESS, "mf identity: status")
call check(maxval(abs(x - b)) < 1.0e-12_c_double, "mf identity: solution")

if (failures > 0) then
    print '(a,i0,a)', "test_solver_fortran: ", failures, " failure(s)"
    error stop 1
end if
print '(a)', "test_solver_fortran: all tests passed"

end program
