! rbf_ellpack -- matrix-vector products for matrices with a fixed number
! of entries per row, nnzrow, which is what a fixed-width stencil graph
! gives: the column indices and the values are rectangular arrays and
! there is no row pointer. Two layouts, differing in which index runs
! fastest:
!
!   row layout   a(lda, n)        the entries of a row are contiguous;
!                                 lda >= nnzrow, entries past nnzrow
!                                 are padding
!   col layout   a(lda, nnzrow)   the j-th entries of all rows are
!                                 contiguous; lda >= n, rows past n
!                                 are padding
!
! The row layout is compressed sparse row with a fixed row length, the
! row pointer implicit as i*lda, and it is the layout the library
! produces: the stencils ja(k, n) of NodeSet::stencils are its index
! array and the weights the assembly kernels write beside them its
! values. The col layout is ELLPACK proper, the transposed storage that
! lets the SIMD lanes take consecutive rows, and is the one to convert
! to where the product dominates the run time.
!
! Both products are the BLAS-shaped update
!
!     y := beta*y + alpha*A*x
!
! with the BLAS convention that y is not read when beta is zero, and
! the BLAS argument order: the dimensions, alpha, the matrix with its
! leading dimension, x, beta, y. Indices are 0-based, as the stencils
! come; padding is never read.
!
! Parallelism: threads take blocks of rows (an OpenMP parallel do over
! the rows, static schedule) and the SIMD lanes take what is contiguous
! in the layout -- in the row layout the entries of a row, a gather and
! a short reduction, in the col layout the rows of a block, where the
! same j-th entry of consecutive rows sits in consecutive memory. The
! two directives are kept apart, one loop each, since a combined
! parallel do simd would want schedule(simd:static) to split the rows
! along the vector length, and not every compiler takes that. The file
! is preprocessed (.F90) for the one compiler guard, on the simd
! reduction of the row layout, and for the block size below.
module rbf_ellpack
use, intrinsic :: iso_c_binding, only: c_int, c_double
implicit none
private

public :: ellpack_mv_row, ellpack_mv_col

integer, parameter :: wp = c_double

! Rows per block of the col layout kernel: the block's partial sums
! stay in a small stack array while its nnzrow entries are swept, so
! the sweep is a set of short contiguous runs through a and ja and one
! pass over y. The value matters: measured on a 21-point stencil over
! a million nodes, 32 to 64 rows run about 1.5x faster than 8 or than
! 256 and above, with either compiler and at one thread or four
! (docs/solver.md has the table). The preprocessor symbol is for
! measuring it again on another machine.
#ifndef RBF_ELLPACK_NBLOCK
#define RBF_ELLPACK_NBLOCK 32
#endif
integer, parameter :: nblock = RBF_ELLPACK_NBLOCK

contains

    ! The dot product of one row with x: the SIMD lanes take the
    ! entries, a gather and a short reduction. The reduction clause is
    ! what lets the compiler reassociate the sum; without it a
    ! floating-point reduction stays scalar unless fast-math says
    ! otherwise. flang (20, at least) cannot lower a reduction on a simd
    ! loop yet -- "not yet implemented: Unhandled clause reduction in
    ! omp.simd" -- so there the loop runs scalar; lift the guard once it
    ! can.
    function row_dot(nnzrow, a, ja, x) result(t)
        integer, intent(in) :: nnzrow
        real(wp), intent(in) :: a(nnzrow)
        integer(c_int), intent(in) :: ja(nnzrow)
        real(wp), intent(in) :: x(0:*)
        real(wp) :: t
        integer :: j
        t = 0
#ifndef __flang__
        !$omp simd reduction(+:t)
#endif
        do j = 1, nnzrow
            t = t + a(j)*x(ja(j))
        end do
    end function

    ! y := beta*y + alpha*A*x, A in the row layout: a(j, i) is the j-th
    ! entry of row i and multiplies x(ja(j, i)), for j up to nnzrow;
    ! lda >= nnzrow is the leading dimension of a and ja.
    subroutine ellpack_mv_row(n, nnzrow, alpha, a, ja, lda, x, beta, y)
        integer, intent(in) :: n, nnzrow, lda
        real(wp), intent(in) :: alpha
        real(wp), intent(in) :: a(lda, 0:n - 1)
        integer(c_int), intent(in) :: ja(lda, 0:n - 1)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(in) :: beta
        real(wp), intent(inout) :: y(0:n - 1)

        integer :: i

        ! Threads take blocks of rows; the row itself is row_dot's
        if (beta == 0) then
            !$omp parallel do schedule(static)
            do i = 0, n - 1
                y(i) = alpha*row_dot(nnzrow, a(:, i), ja(:, i), x)
            end do
        else
            !$omp parallel do schedule(static)
            do i = 0, n - 1
                y(i) = beta*y(i) + alpha*row_dot(nnzrow, a(:, i), ja(:, i), x)
            end do
        end if
    end subroutine

    ! y := beta*y + alpha*A*x, A in the col layout: a(i, j) is the j-th
    ! entry of row i and multiplies x(ja(i, j)); lda >= n is the leading
    ! dimension of a and ja.
    subroutine ellpack_mv_col(n, nnzrow, alpha, a, ja, lda, x, beta, y)
        integer, intent(in) :: n, nnzrow, lda
        real(wp), intent(in) :: alpha
        real(wp), intent(in) :: a(0:lda - 1, nnzrow)
        integer(c_int), intent(in) :: ja(0:lda - 1, nnzrow)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(in) :: beta
        real(wp), intent(inout) :: y(0:n - 1)

        real(wp) :: yt(0:nblock - 1)
        integer :: i, j, ii, iend

        ! Each thread sweeps the entries of its blocks of rows; within
        ! a block the partial sums of consecutive rows are the lanes.
        !$omp parallel do schedule(static) private(yt, i, j, iend)
        do ii = 0, n - 1, nblock
            iend = min(ii + nblock - 1, n - 1)
            yt = 0
            do j = 1, nnzrow
                !$omp simd
                do i = ii, iend
                    yt(i - ii) = yt(i - ii) + a(i, j)*x(ja(i, j))
                end do
            end do
            if (beta == 0) then
                !$omp simd
                do i = ii, iend
                    y(i) = alpha*yt(i - ii)
                end do
            else
                !$omp simd
                do i = ii, iend
                    y(i) = beta*y(i) + alpha*yt(i - ii)
                end do
            end if
        end do
    end subroutine

end module
