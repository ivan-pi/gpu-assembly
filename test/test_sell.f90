! Tests for the module rbf_sell: packing the fixed-row-length CSR into
! chunks and the product against a plain triple loop, on a banded
! matrix with a pseudo-random pattern of nnzrow entries per row, for
! chunk lengths that do and do not divide n, and for the three cases
! of beta: zero, where y must not be read, one, and a general value.
! The CSR arrays have a leading dimension larger than nnzrow, with the
! padding poisoned, and the padding rows of the last chunk are checked
! to hold zero values and in-bounds indices.
!
! Built and run through CMake:
!
!   cmake -B build
!   cmake --build build
!   ctest --test-dir build
program test_sell
use, intrinsic :: iso_c_binding, only: c_int, c_double
use rbf_sell
implicit none

integer, parameter :: wp = c_double
integer, parameter :: n = 1001, nnzrow = 7, lda = 8

real(wp) :: a(lda, n)
integer(c_int) :: ja(lda, n)
real(wp), allocatable :: asell(:, :, :)
integer(c_int), allocatable :: jsell(:, :, :)
real(wp) :: x(n), y(n), y0(n), yref(n)
integer :: i, j, c, nchunks, failures
integer(c_int) :: seed

failures = 0
seed = 12345

a = huge(1.0_wp)
ja = -1
do i = 1, n
    do j = 1, nnzrow
        ja(j, i) = int(modulo(i - 1 + next(seed, 7) - 3, n), c_int)  ! 0-based
        a(j, i) = 0.5_wp - real(next(seed, 1000), wp)/1000
    end do
end do
do i = 1, n
    x(i) = real(next(seed, 1000), wp)/1000 - 0.5_wp
    y0(i) = real(next(seed, 1000), wp)/1000
end do

yref = 0
do i = 1, n
    do j = 1, nnzrow
        yref(i) = yref(i) + a(j, i)*x(ja(j, i) + 1)
    end do
end do

call run_chunk(7)     ! divides 1001: no padding
call run_chunk(8)     ! 126 chunks, 7 padding rows
call run_chunk(32)
call run_chunk(1024)  ! one chunk, mostly padding

if (failures > 0) then
    print '(a,i0,a)', "test_sell: ", failures, " failure(s)"
    error stop 1
end if
print '(a)', "test_sell: all tests passed"

contains

    subroutine run_chunk(c)
        integer, intent(in) :: c
        character(len=32) :: tag
        write (tag, '(a,i0,a)') "c=", c, ": "

        nchunks = sell_nchunks(n, c)
        call check(nchunks == (n + c - 1)/c, trim(tag)//"nchunks")
        if (allocated(asell)) deallocate(asell, jsell)
        allocate(asell(c, nnzrow, nchunks), jsell(c, nnzrow, nchunks))
        asell = huge(1.0_wp)
        jsell = -1
        call sell_pack(n, nnzrow, a, ja, lda, c, asell, jsell)

        ! the padding rows of the last chunk: zero values, valid indices
        do i = n - (nchunks - 1)*c + 1, c
            call check(all(asell(i, :, nchunks) == 0), trim(tag)//"padding values")
            call check(all(jsell(i, :, nchunks) >= 0 .and. jsell(i, :, nchunks) < n), &
                trim(tag)//"padding indices")
        end do
        ! a row in the middle, packed where it should be: row i (1-based)
        ! sits in chunk (i - 1)/c + 1 at lane mod(i - 1, c) + 1
        i = n/2
        call check(all(asell(mod(i - 1, c) + 1, :, (i - 1)/c + 1) == a(1:nnzrow, i)) .and. &
                   all(jsell(mod(i - 1, c) + 1, :, (i - 1)/c + 1) == ja(1:nnzrow, i)), &
                   trim(tag)//"a packed row")

        y = huge(1.0_wp)
        call sell_mv(n, nnzrow, c, 1.0_wp, asell, jsell, x, 0.0_wp, y)
        call check(close(y, yref), trim(tag)//"y = A x")
        y = huge(1.0_wp)
        call sell_mv(n, nnzrow, c, 3.0_wp, asell, jsell, x, 0.0_wp, y)
        call check(close(y, 3*yref), trim(tag)//"y = alpha A x")
        y = y0
        call sell_mv(n, nnzrow, c, 2.0_wp, asell, jsell, x, 1.0_wp, y)
        call check(close(y, y0 + 2*yref), trim(tag)//"y = y + alpha A x")
        y = y0
        call sell_mv(n, nnzrow, c, 2.0_wp, asell, jsell, x, -0.5_wp, y)
        call check(close(y, -0.5_wp*y0 + 2*yref), trim(tag)//"y = beta y + alpha A x")
    end subroutine

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
