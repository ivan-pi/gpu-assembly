! Tests for the module rbf_ellpack: the product against a plain triple
! loop, on a banded matrix with a pseudo-random pattern of nnzrow
! entries per row, for the three cases of beta: zero, where y must not
! be read, one, and a general value. The arrays have a leading
! dimension larger than n, and the padding rows are filled with garbage
! to show they are never read.
!
! Built and run through CMake:
!
!   cmake -B build
!   cmake --build build
!   ctest --test-dir build
program test_ellpack
use, intrinsic :: iso_c_binding, only: c_int, c_double
use rbf_ellpack
implicit none

integer, parameter :: wp = c_double
integer, parameter :: n = 1001, nnzrow = 7, lda = 1024

real(wp) :: a(lda, nnzrow)
integer(c_int) :: ja(lda, nnzrow)
real(wp) :: x(n), y(n), y0(n), yref(n)
integer :: i, j, failures
integer(c_int) :: seed

failures = 0
seed = 12345

! The pattern: entry j of row i points at a node within the band
! [i - 3, i + 3], wrapped, in scrambled order and with duplicates
! (a stencil is not allowed those, the product does not care); the
! padding rows are poisoned
a = huge(1.0_wp)
ja = -1
do i = 1, n
    do j = 1, nnzrow
        ja(i, j) = int(modulo(i - 1 + next(seed, 7) - 3, n), c_int)  ! 0-based
        a(i, j) = 0.5_wp - real(next(seed, 1000), wp)/1000
    end do
end do
do i = 1, n
    x(i) = real(next(seed, 1000), wp)/1000 - 0.5_wp
    y0(i) = real(next(seed, 1000), wp)/1000
end do

! y = A x
yref = 0
do i = 1, n
    do j = 1, nnzrow
        yref(i) = yref(i) + a(i, j)*x(ja(i, j) + 1)
    end do
end do

! beta = 0: y not read
y = huge(1.0_wp)
call ellpack_mv(n, nnzrow, 1.0_wp, a, ja, lda, x, 0.0_wp, y)
call check(close(y, yref), "y = A x")
y = huge(1.0_wp)
call ellpack_mv(n, nnzrow, 3.0_wp, a, ja, lda, x, 0.0_wp, y)
call check(close(y, 3*yref), "y = alpha A x")

! beta = 1: the accumulating form of the matrix-free solver's callback
y = y0
call ellpack_mv(n, nnzrow, 2.0_wp, a, ja, lda, x, 1.0_wp, y)
call check(close(y, y0 + 2*yref), "y = y + alpha A x")

! general beta
y = y0
call ellpack_mv(n, nnzrow, 2.0_wp, a, ja, lda, x, -0.5_wp, y)
call check(close(y, -0.5_wp*y0 + 2*yref), "y = beta y + alpha A x")

if (failures > 0) then
    print '(a,i0,a)', "test_ellpack: ", failures, " failure(s)"
    error stop 1
end if
print '(a)', "test_ellpack: all tests passed"

contains

    subroutine check(cond, what)
        logical, intent(in) :: cond
        character(*), intent(in) :: what
        if (.not. cond) then
            print '(a,a)', "FAIL ", what
            failures = failures + 1
        end if
    end subroutine

    logical function close(a, b)
        real(wp), intent(in) :: a(:), b(:)
        close = all(abs(a - b) <= 1.0e-13_wp*(1 + abs(b)))
    end function

    ! A linear congruential generator; the values only need to be irregular
    integer function next(seed, m)
        integer(c_int), intent(inout) :: seed
        integer, intent(in) :: m
        seed = int(modulo(1103515245_8*seed + 12345_8, 2147483648_8), c_int)
        next = int(modulo(seed/65536, m))
    end function

end program
