! rbf_ellpack -- the matrix-vector product for a matrix in ELLPACK
! storage: every row holds the same number of entries, nnzrow, and the
! values and column indices are the rectangular arrays
!
!     a(lda, nnzrow), ja(lda, nnzrow), lda >= n
!
! with a(i, j) the j-th entry of row i, multiplying x(ja(i, j)). The
! j-th entries of consecutive rows are consecutive in memory, which is
! what lets the SIMD lanes take consecutive rows; the rows past n are
! padding and never read. Indices are 0-based, as the stencils come.
!
! A fixed-width stencil graph and the weights assembled beside it are
! this storage transposed, ja(nnzrow, n): that is compressed sparse row
! with a fixed row length and an implicit row pointer, and its product
! is csr_mv in rbf_csr, with the same argument list. Measured on kNN
! matrices of a point cloud, that product is as fast as this one at
! its best block size and faster at any other (docs/solver.md), so
! the transpose is not worth making for the product alone; this
! module is for data that is in ELLPACK already.
!
! The product is the BLAS-shaped update
!
!     y := beta*y + alpha*A*x
!
! with the BLAS convention that y is not read when beta is zero, and
! the BLAS argument order: the dimensions, alpha, the matrix with its
! leading dimension, x, beta, y.
!
! Parallelism: threads take blocks of rows (an OpenMP parallel do over
! the blocks, static schedule) and within a block the SIMD lanes take
! the rows. The two directives sit on separate loops, since a combined
! parallel do simd would want schedule(simd:static) to split the rows
! along the vector length, and not every compiler takes that. The file
! is preprocessed (.F90) for the block size below.
module rbf_ellpack
use, intrinsic :: iso_c_binding, only: c_int, c_double
implicit none
private

public :: ellpack_mv

integer, parameter :: wp = c_double

! Rows per block: the block's partial sums stay in a stack array while
! its nnzrow entries are swept, so the sweep is 2*nnzrow contiguous
! runs through a and ja, one block long each, and one pass over y. The
! value matters and the optimum is not stable across machines: on one
! 4-core Xeon 32 rows ran 1.5x faster than 1024, on another 1024 ran
! 3x faster than 32 (docs/solver.md has both). Short runs leave the
! 2*nnzrow streams to the hardware prefetcher, which may or may not
! keep up with that many; long runs make every stream sequential for
! kilobytes at a time and cost only 8 bytes of stack per row. 1024 was
! within 10% of the CSR product of rbf_csr on both machines and is the
! default; the preprocessor symbol is for measuring again.
#ifndef RBF_ELLPACK_NBLOCK
#define RBF_ELLPACK_NBLOCK 1024
#endif
integer, parameter :: nblock = RBF_ELLPACK_NBLOCK

contains

    ! y := beta*y + alpha*A*x
    subroutine ellpack_mv(n, nnzrow, alpha, a, ja, lda, x, beta, y)
        integer, intent(in) :: n, nnzrow, lda
        real(wp), intent(in) :: alpha
        real(wp), intent(in) :: a(0:lda - 1, nnzrow)
        integer(c_int), intent(in) :: ja(0:lda - 1, nnzrow)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(in) :: beta
        real(wp), intent(inout) :: y(0:n - 1)

        real(wp) :: sums(0:nblock - 1)  ! the block's partial sums, a small stack array
        integer :: first  ! first row of the block
        integer :: nrows  ! rows in the block, nblock except for the last
        integer :: k      ! row within the block, the SIMD lane
        integer :: j      ! entry within the row

        !$omp parallel do schedule(static) private(sums, nrows, k, j)
        do first = 0, n - 1, nblock
            nrows = min(nblock, n - first)
            sums = 0
            do j = 1, nnzrow
                !$omp simd
                do k = 0, nrows - 1
                    sums(k) = sums(k) + a(first + k, j)*x(ja(first + k, j))
                end do
            end do
            if (beta == 0) then
                !$omp simd
                do k = 0, nrows - 1
                    y(first + k) = alpha*sums(k)
                end do
            else
                !$omp simd
                do k = 0, nrows - 1
                    y(first + k) = beta*y(first + k) + alpha*sums(k)
                end do
            end if
        end do
    end subroutine

end module
