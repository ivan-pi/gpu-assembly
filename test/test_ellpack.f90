! Tests for the module rbf_ellpack: the products of both layouts against
! a plain triple loop, on a banded matrix with a pseudo-random pattern
! of nnzrow entries per row, with and without alpha and beta, and the
! streaming step against direction-by-direction products. Padding rows
! of the col layout are filled with garbage to show they are not read.
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
integer, parameter :: n = 1001, nnzrow = 7, lda = 1024, qdirs = 4, ldpdf = 1040

real(wp) :: a_row(nnzrow, n), a_col(lda, nnzrow)
integer(c_int) :: ja_row(nnzrow, n), ja_col(lda, nnzrow)
real(wp) :: x(n), y(n), y0(n), yref(n)
real(wp) :: a_row_q(nnzrow, n, qdirs), a_col_q(lda, nnzrow, qdirs)
real(wp) :: fold(ldpdf, qdirs), fnew(ldpdf, qdirs), fref(ldpdf, qdirs)
integer :: i, j, q, failures
integer(c_int) :: seed

failures = 0
seed = 12345

! The pattern: entry j of row i points at a node within the band
! [i - 3, i + 3], wrapped, in scrambled order and with duplicates
! (a stencil is not allowed those, the product does not care)
do i = 1, n
    do j = 1, nnzrow
        ja_row(j, i) = int(modulo(i - 1 + next(seed, 7) - 3, n), c_int)  ! 0-based
        a_row(j, i) = 0.5_wp - real(next(seed, 1000), wp)/1000
    end do
end do
do i = 1, n
    x(i) = real(next(seed, 1000), wp)/1000 - 0.5_wp
    y0(i) = real(next(seed, 1000), wp)/1000
end do

! The same matrix in the col layout, with poisoned padding
a_col = huge(1.0_wp)
ja_col = -1
do j = 1, nnzrow
    a_col(1:n, j) = a_row(j, :)
    ja_col(1:n, j) = ja_row(j, :)
end do

! y = A x
yref = 0
do i = 1, n
    do j = 1, nnzrow
        yref(i) = yref(i) + a_row(j, i)*x(ja_row(j, i) + 1)
    end do
end do

y = huge(1.0_wp)  ! not read when beta is absent
call ellpack_mv_row(n, nnzrow, a_row, ja_row, x, y)
call check(close(y, yref), "row: y = A x")
y = huge(1.0_wp)
call ellpack_mv_col(n, nnzrow, lda, a_col, ja_col, x, y)
call check(close(y, yref), "col: y = A x")

! y = beta y + alpha A x
y = y0
call ellpack_mv_row(n, nnzrow, a_row, ja_row, x, y, alpha=2.0_wp, beta=-0.5_wp)
call check(close(y, -0.5_wp*y0 + 2*yref), "row: y = beta y + alpha A x")
y = y0
call ellpack_mv_col(n, nnzrow, lda, a_col, ja_col, x, y, alpha=2.0_wp, beta=-0.5_wp)
call check(close(y, -0.5_wp*y0 + 2*yref), "col: y = beta y + alpha A x")

! beta given as zero: y still not read
y = huge(1.0_wp)
call ellpack_mv_row(n, nnzrow, a_row, ja_row, x, y, alpha=3.0_wp, beta=0.0_wp)
call check(close(y, 3*yref), "row: beta = 0")
y = huge(1.0_wp)
call ellpack_mv_col(n, nnzrow, lda, a_col, ja_col, x, y, alpha=3.0_wp, beta=0.0_wp)
call check(close(y, 3*yref), "col: beta = 0")

! Streaming: one matrix per direction, the rest direction copied
a_col_q = huge(1.0_wp)
do q = 1, qdirs
    a_row_q(:, :, q) = a_row*q
    do j = 1, nnzrow
        a_col_q(1:n, j, q) = a_row_q(j, :, q)
    end do
end do
fold = 0
do q = 1, qdirs
    do i = 1, n
        fold(i, q) = real(next(seed, 1000), wp)/1000
    end do
end do
fref = 0
fref(1:n, 1) = fold(1:n, 1)
do q = 2, qdirs
    do i = 1, n
        do j = 1, nnzrow
            fref(i, q) = fref(i, q) + a_row_q(j, i, q)*fold(ja_row(j, i) + 1, q)
        end do
    end do
end do

fnew = 0
call ellpack_stream_row(n, qdirs, fnew, fold, ldpdf, ja_row, a_row_q, nnzrow)
call check(all(abs(fnew - fref) <= 1.0e-13_wp), "stream row")
fnew = 0
call ellpack_stream_col(n, qdirs, fnew, fold, ldpdf, ja_col, a_col_q, lda, nnzrow)
call check(all(abs(fnew - fref) <= 1.0e-13_wp), "stream col")

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
