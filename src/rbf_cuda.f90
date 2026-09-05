! rbf_cuda.f90 -- GPU assembly of RBF-FD interpolation matrices
!
! Fills the augmented saddle-point system for polyharmonic-spline (PHS)
! RBF-FD with quadratic polynomial augmentation,
!
!     M = [ A   P ]      A(k,l) = |x_k - x_l|^3
!         [ P^T 0 ]      P(k,:) = [1, x_k, y_k, x_k^2, x_k y_k, y_k^2]
!
! for every stencil of a neighbour graph given in CSR form (ia, ja, 0-based).
!
! Parallel layout:
!   - one thread block per stencil (blockIdx%x - 1 = stencil index)
!   - threads stride over the entries of the stencil's matrix
!   - stencil coordinates are staged in shared memory once per block
!   - P^T is copied from P with one thread per column, after a barrier
!
! Build:
!  nvfortran -cuda rbf_cuda.f90
!
module rbf_cuda
    implicit none
    private

    public :: dp, fill_kernel, fill_rhs, print_matrix
    public :: store_to_csr

    integer, parameter :: dp = kind(1.0d0)

    ! Maximum stencil size
    integer, parameter :: NMAX = 64

contains
!
! TODO: generalize to multiple stencils, currently only one output matrix M
!
attributes(global) subroutine fill_kernel(nstencils,x,y,ia,ja,M,ldm)
    integer, value :: nstencils
    real(dp), intent(in), device :: x(0:nstencils-1), y(0:nstencils-1)  ! Points
    integer, intent(in), device :: ia(0:nstencils), ja(0:*)     ! Neighbor graph
    integer, value :: ldm
    real(dp), intent(out), device :: M(ldm,*)

    real(dp), shared :: xs(NMAX), ys(NMAX)
    integer :: tid, iaa, iab, n, k, col, row, idx
    real(dp) :: rsqr


    idx = blockIdx%x - 1

    tid = threadIdx%x

    iaa = ia(idx)
    iab = ia(idx+1)-1

    ! Stencil size
    n = iab - iaa + 1

    ! Load coordinates into shared memory
    do k = tid, n, blockDim%x
        xs(k) = x(ja(iaa + k - 1))
        ys(k) = y(ja(iaa + k - 1))
    end do
    call syncthreads()

    ! Fill PHS section

    ! Loop over RBF columns
    do col = 1, n
        do k = tid, n, blockDim%x
            rsqr = (xs(col) - xs(k))**2 + (ys(col) - ys(k))**2
            M(k,col) = rsqr * sqrt(rsqr)
        end do
    end do

    ! Polynomial part
    do k = tid, n, blockDim%x
        M(k,n+1) = 1.0_dp
        M(k,n+2) = xs(k)
        M(k,n+3) = ys(k)
        M(k,n+4) = xs(k)**2
        M(k,n+5) = xs(k)*ys(k)
        M(k,n+6) = ys(k)**2
    end do
    call syncthreads()


    ! Transpose (lower-left block): one thread per column k
    ! (recomputing would remove the thread barrier)
    do k = tid, n, blockDim%x
        do row = n+1, n+6
            M(row,k) = M(k,row)
        end do
    end do

    do k = tid, 6, blockDim%x
        M(n+1:n+6,n+k) = 0.0_dp
    end do

end subroutine

! Interpolation operators, fill the right hand side,
! B = [Lphi, Lp] for columns j = 1 to nc, where each
! column performs interpolation at the point (xc(j),yc(j))
attributes(global) subroutine fill_rhs(n,x,y,nc,xc,yc,B,ldb)
    integer, value :: n, nc
    real(dp), intent(in) :: x(n), y(n)
    real(dp), intent(in) :: xc(nc), yc(nc)

    real(dp), intent(out) :: B(ldb,*) ! atleast nc columns

    integer :: tid, col, k
    real(dp) :: xcc, ycc, rsqr

    tid = threadIdx%x

    ! RBF rows: B(k,col) = phi(|| (xc,yc)_col - (x,y)_k ||)
    do col = 1, nc
        xcc = xc(col); ycc = yc(col)
        do k = tid, n, blockDim%x
            rsqr = (xcc - x(k))**2 + (ycc - y(k))**2
            B(k,col) = rsqr*sqrt(rsqr)
        end do
    end do

    ! Polynomial rows: B(n+i,col) = p_i(xc_col, yc_col)
    do col = tid, nc, blockDim%x
        xcc = xc(col); ycc = yc(col)
        B(n+1,col) = 1.0_dp
        B(n+2,col) = xcc
        B(n+3,col) = ycc
        B(n+4,col) = xcc**2
        B(n+5,col) = xcc*ycc
        B(n+6,col) = ycc**2
    end do

end subroutine

!> Pack stencil weights into CSR storage
attributes(device) subroutine store_to_csr(nstencils,ia,va,W,ldw)
    integer, value :: nstencils, ldw
    integer, intent(in) :: ia(0:nstencils)
    real(dp), intent(inout) :: va(0:*)
    real(dp), intent(in) :: W(ldw,2) ! Laplacian weights

    integer :: iaa, iab, n

    iaa = ia(idx)
    iab = ia(idx+1)-1

    ! Local stencil size
    n = iab - iaa + 1

    do k = threadIdx%x, n, blockDim%x
        VA(iaa+k-1) = W(k,1) + W(k,2)    ! p_xx + p_yy
    end do

end subroutine

subroutine print_matrix(nt, ld, M, label)
    integer,  intent(in) :: nt, ld
    real(dp), intent(in) :: M(ld, *)
    character(*), intent(in), optional :: label
    integer :: i

    if (present(label)) print '(a)', label
    do i = 1, nt
        print '(*(f10.4))', M(i, 1:nt)
    end do
    print *
end subroutine

end module

