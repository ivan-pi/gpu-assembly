! rbf_csr -- the matrix-vector product for a matrix in compressed
! sparse row storage with a fixed row length, the pattern of a
! k-nearest-neighbour graph over a point cloud: every row holds the
! same number of entries, nnzrow, so the row pointer is implicit and
! the values and column indices are the rectangular arrays
!
!     a(lda, n), ja(lda, n), lda >= nnzrow
!
! with a(j, i) the j-th entry of row i, multiplying x(ja(j, i)). The
! entries of a row are contiguous; the entries past nnzrow are padding
! and never read. Indices are 0-based, as the stencils come. The
! stencils ja(k, n) of NodeSet::stencils are this index array with
! lda = k, and the weights the assembly kernels write beside them are
! its values, so a driver multiplies with what the assembly left.
!
! The product is the BLAS-shaped update
!
!     y := beta*y + alpha*A*x
!
! with the BLAS convention that y is not read when beta is zero, and
! the BLAS argument order: the dimensions, alpha, the matrix with its
! leading dimension, x, beta, y -- the argument list of ellpack_mv in
! rbf_ellpack, the transposed storage, only the meaning of lda
! differs. The general CSR product, with a row pointer, is csr_mv in
! rbf_solver; the two generics merge when both modules are used, and
! the arguments tell them apart.
!
! Parallelism: threads take rows (an OpenMP parallel do, static
! schedule) and the SIMD lanes take the entries of a row, a gather and
! a short reduction. The file is preprocessed (.F90) for the one
! compiler guard on that reduction.
module rbf_csr
use, intrinsic :: iso_c_binding, only: c_int, c_double
implicit none
private

public :: csr_mv

integer, parameter :: wp = c_double

interface csr_mv
    module procedure csr_mv_dp
end interface

contains

    ! The dot product of one row with x. The reduction clause is what
    ! lets the compiler reassociate the sum; without it a
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

    ! y := beta*y + alpha*A*x
    subroutine csr_mv_dp(n, nnzrow, alpha, a, ja, lda, x, beta, y)
        integer, intent(in) :: n, nnzrow, lda
        real(wp), intent(in) :: alpha
        real(wp), intent(in) :: a(lda, 0:n - 1)
        integer(c_int), intent(in) :: ja(lda, 0:n - 1)
        real(wp), intent(in) :: x(0:n - 1)
        real(wp), intent(in) :: beta
        real(wp), intent(inout) :: y(0:n - 1)

        integer :: i

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

end module
