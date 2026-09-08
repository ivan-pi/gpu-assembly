! rbf_ellpack -- matrix-vector products for matrices in ELLPACK storage,
! the natural storage of a fixed-width stencil graph: every row holds
! the same number of entries, nnzrow, so the column indices and the
! values are rectangular arrays and there is no row pointer. The
! stencils ja(k, n) of NodeSet::stencils are already the index array of
! the row layout, and the weights the assembly kernels write beside
! them are its values.
!
! Two layouts, differing in which index runs fastest:
!
!   row layout   a(nnzrow, n)   the entries of a row are contiguous
!   col layout   a(lda, nnzrow) the j-th entries of all rows are
!                               contiguous; lda >= n, the padding rows
!                               are never read
!
! and for each the product y := beta*y + alpha*A*x, with the BLAS
! convention that y is not read when beta is zero, plus the streaming
! step of a lattice Boltzmann scheme, one such product per direction.
!
! Parallelism: threads take blocks of rows (an OpenMP parallel do over
! the rows, static schedule) and the SIMD lanes take what is contiguous
! in the layout -- in the row layout the entries of a row, a reduction
! over a short loop, in the col layout the rows of a block, where the
! same j-th entry of consecutive rows sits in consecutive memory. The
! two directives are kept apart, one loop each, since a combined
! parallel do simd would want schedule(simd:static) to split the rows
! along the vector length, and not every compiler takes that. The file
! is preprocessed (.F90) for the one compiler guard, on the simd
! reduction of the row layout.
module rbf_ellpack
use, intrinsic :: iso_c_binding, only: c_int, c_double
implicit none
private

public :: ellpack_mv_row, ellpack_mv_col
public :: ellpack_stream_row, ellpack_stream_col

integer, parameter :: wp = c_double

! Rows per block of the col layout kernel; the block's partial sums
! stay in a small stack array while the entries are swept.
integer, parameter :: nblock = 64

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
    ! entry of row i and multiplies x(ja(j, i)). Indices are 0-based, as
    ! the stencils come; alpha and beta default to 1 and 0.
    subroutine ellpack_mv_row(n, nnzrow, a, ja, x, y, alpha, beta)
        integer, intent(in) :: n, nnzrow
        real(wp), intent(in) :: a(nnzrow, 0:n - 1)
        integer(c_int), intent(in) :: ja(nnzrow, 0:n - 1)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(inout) :: y(0:n - 1)
        real(wp), intent(in), optional :: alpha, beta

        real(wp) :: alpha_, beta_
        integer :: i

        alpha_ = 1
        if (present(alpha)) alpha_ = alpha
        beta_ = 0
        if (present(beta)) beta_ = beta

        ! Threads take blocks of rows; the row itself is row_dot's
        if (beta_ == 0) then
            !$omp parallel do schedule(static)
            do i = 0, n - 1
                y(i) = alpha_*row_dot(nnzrow, a(:, i), ja(:, i), x)
            end do
        else
            !$omp parallel do schedule(static)
            do i = 0, n - 1
                y(i) = beta_*y(i) + alpha_*row_dot(nnzrow, a(:, i), ja(:, i), x)
            end do
        end if
    end subroutine

    ! y := beta*y + alpha*A*x, A in the col layout: a(i, j) is the j-th
    ! entry of row i and multiplies x(ja(i, j)); lda >= n is the leading
    ! dimension of both arrays. Indices are 0-based; alpha and beta
    ! default to 1 and 0.
    subroutine ellpack_mv_col(n, nnzrow, lda, a, ja, x, y, alpha, beta)
        integer, intent(in) :: n, nnzrow, lda
        real(wp), intent(in) :: a(0:lda - 1, nnzrow)
        integer(c_int), intent(in) :: ja(0:lda - 1, nnzrow)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(inout) :: y(0:n - 1)
        real(wp), intent(in), optional :: alpha, beta

        real(wp) :: alpha_, beta_
        real(wp) :: yt(0:nblock - 1)
        integer :: i, j, ii, iend

        alpha_ = 1
        if (present(alpha)) alpha_ = alpha
        beta_ = 0
        if (present(beta)) beta_ = beta

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
            if (beta_ == 0) then
                !$omp simd
                do i = ii, iend
                    y(i) = alpha_*yt(i - ii)
                end do
            else
                !$omp simd
                do i = ii, iend
                    y(i) = beta_*y(i) + alpha_*yt(i - ii)
                end do
            end if
        end do
    end subroutine

    ! The streaming step: fnew(:, q) = A_q fold(:, q) for the directions
    ! q = 1 .. qdirs-1, each with its own matrix a(:, :, q) over the one
    ! index array ja, and fnew(:, 0) = fold(:, 0), the rest direction,
    ! which streams nowhere. The populations are stored one direction
    ! after the other with a leading dimension ldpdf >= n, as the D2Q9
    ! kernels store them. Row layout, a(nnzrow, n, qdirs).
    subroutine ellpack_stream_row(n, qdirs, fnew, fold, ldpdf, ja, a, nnzrow) bind(c)
        integer(c_int), intent(in), value :: n, qdirs, ldpdf, nnzrow
        real(wp), intent(inout) :: fnew(ldpdf, 0:qdirs - 1)
        real(wp), intent(in) :: fold(ldpdf, 0:qdirs - 1)
        integer(c_int), intent(in) :: ja(nnzrow, 0:n - 1)
        real(wp), intent(in) :: a(nnzrow, 0:n - 1, 0:qdirs - 1)

        integer :: i, q

        !$omp parallel do schedule(static)
        do i = 1, n
            fnew(i, 0) = fold(i, 0)
        end do

        do q = 1, qdirs - 1
            call ellpack_mv_row(n, nnzrow, a(:, :, q), ja, fold(1:n, q), fnew(1:n, q))
        end do
    end subroutine

    ! The same in the col layout, a(lda, nnzrow, qdirs) and ja(lda, nnzrow).
    subroutine ellpack_stream_col(n, qdirs, fnew, fold, ldpdf, ja, a, lda, nnzrow) bind(c)
        integer(c_int), intent(in), value :: n, qdirs, ldpdf, lda, nnzrow
        real(wp), intent(inout) :: fnew(ldpdf, 0:qdirs - 1)
        real(wp), intent(in) :: fold(ldpdf, 0:qdirs - 1)
        integer(c_int), intent(in) :: ja(lda, nnzrow)
        real(wp), intent(in) :: a(lda, nnzrow, 0:qdirs - 1)

        integer :: i, q

        !$omp parallel do schedule(static)
        do i = 1, n
            fnew(i, 0) = fold(i, 0)
        end do

        do q = 1, qdirs - 1
            call ellpack_mv_col(n, nnzrow, lda, a(:, :, q), ja, fold(1:n, q), fnew(1:n, q))
        end do
    end subroutine

end module
