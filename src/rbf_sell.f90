! rbf_sell -- the matrix-vector product for a matrix in sliced ELLPACK
! storage (SELL-C, Kreutzer et al. 2014, without the sorting, since
! every row has the same number of entries): the rows are cut into
! chunks of c, and within a chunk the storage is ELLPACK,
!
!     a(c, nnzrow, nchunks), ja(c, nnzrow, nchunks)
!
! with a(i, j, k) the j-th entry of the i-th row of chunk k, that is
! of row (k - 1)*c + i - 1 in the 0-based count of the vectors, and
! multiplying x(ja(i, j, k)). Consecutive rows are consecutive in
! memory, so the SIMD lanes take the rows of a chunk, and a whole
! chunk is one contiguous run of c*nnzrow values and as many indices,
! so the product streams two sequential arrays, as the CSR product
! does, instead of the 2*nnzrow strided streams of plain ELLPACK. The
! rows past n in the last chunk are padding: zero values and any valid
! index, which sell_pack writes.
!
! The product is the BLAS-shaped update
!
!     y := beta*y + alpha*A*x
!
! with the BLAS convention that y is not read when beta is zero, and
! the BLAS argument order: the dimensions, alpha, the matrix, x, beta,
! y. c is a run-time dimension rather than a compile-time one so that
! it can be measured without rebuilding; a multiple of the vector
! length, 8 for AVX-512 doubles, keeps the inner loops free of
! remainders. Indices are 0-based, as the stencils come.
!
! Parallelism: threads take chunks (an OpenMP parallel do, static
! schedule) and the SIMD lanes take the rows of a chunk, with the
! chunk's partial sums in a stack array of length c.
module rbf_sell
use, intrinsic :: iso_c_binding, only: c_int, c_double
implicit none
private

public :: sell_nchunks, sell_pack, sell_mv

integer, parameter :: wp = c_double

interface sell_mv
    module procedure sell_mv_dp
end interface

contains

    ! Chunks of c rows that hold n rows, the last one padded.
    pure function sell_nchunks(n, c) result(nchunks)
        integer, intent(in) :: n, c
        integer :: nchunks
        nchunks = (n + c - 1)/c
    end function

    ! From the fixed-row-length CSR of rbf_csr, a(lda, n) and ja(lda, n)
    ! with lda >= nnzrow, into asell(c, nnzrow, nchunks) and
    ! jsell(c, nnzrow, nchunks); the padding rows get zero values and
    ! the indices of row n - 1, so the gather stays in bounds.
    subroutine sell_pack(n, nnzrow, a, ja, lda, c, asell, jsell)
        integer, intent(in) :: n, nnzrow, lda, c
        real(wp), intent(in) :: a(lda, 0:n - 1)
        integer(c_int), intent(in) :: ja(lda, 0:n - 1)
        real(wp), intent(out) :: asell(c, nnzrow, *)
        integer(c_int), intent(out) :: jsell(c, nnzrow, *)

        integer :: i, j, k, row, nchunks

        nchunks = sell_nchunks(n, c)
        !$omp parallel do schedule(static) private(i, j, row)
        do k = 1, nchunks
            do j = 1, nnzrow
                do i = 1, c
                    row = min((k - 1)*c + i - 1, n - 1)
                    jsell(i, j, k) = ja(j, row)
                    asell(i, j, k) = a(j, row)
                end do
            end do
        end do
        ! the padding rows' values, in the last chunk only
        do i = n - (nchunks - 1)*c + 1, c
            asell(i, :, nchunks) = 0
        end do
    end subroutine

    ! y := beta*y + alpha*A*x
    subroutine sell_mv_dp(n, nnzrow, c, alpha, a, ja, x, beta, y)
        integer, intent(in) :: n, nnzrow, c
        real(wp), intent(in) :: alpha
        real(wp), intent(in) :: a(c, nnzrow, *)
        integer(c_int), intent(in) :: ja(c, nnzrow, *)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(in) :: beta
        real(wp), intent(inout) :: y(0:n - 1)

        real(wp) :: sums(c)  ! the chunk's partial sums
        integer :: k      ! chunk
        integer :: first  ! first row of the chunk, 0-based
        integer :: nrows  ! rows in the chunk, c except for the last
        integer :: i      ! row within the chunk, the SIMD lane
        integer :: j      ! entry within the row

        !$omp parallel do schedule(static) private(sums, first, nrows, i, j)
        do k = 1, sell_nchunks(n, c)
            first = (k - 1)*c
            nrows = min(c, n - first)
            sums = 0
            do j = 1, nnzrow
                !$omp simd
                do i = 1, c
                    sums(i) = sums(i) + a(i, j, k)*x(ja(i, j, k))
                end do
            end do
            if (beta == 0) then
                !$omp simd
                do i = 1, nrows
                    y(first + i - 1) = alpha*sums(i)
                end do
            else
                !$omp simd
                do i = 1, nrows
                    y(first + i - 1) = beta*y(first + i - 1) + alpha*sums(i)
                end do
            end if
        end do
    end subroutine

end module
