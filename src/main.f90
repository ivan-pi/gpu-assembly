program main

    use cudafor, only: cudaDeviceSynchronize, cudaSuccess, cudaGetErrorString
    use rbf_cuda, only: dp, fill_kernel, fill_rhs, NMAX, print_matrix

    interface
        attributes(global) subroutine solve_dp(A,ipiv,B,info) bind(c,name="solve_dp")
            use, intrinsic :: iso_c_binding, only: c_double, c_int
            real(c_double), intent(inout) :: A(*)  ! [15,15]
            integer(c_int), intent(inout) :: ipiv(*) ! 15
            real(c_double), intent(inout) :: B(*)  ! [15, 9]
            integer(c_int), intent(out) :: info
        end subroutine
    end interface

    integer, parameter :: n = 9, nt = n + 6, nrhs = n

    real(dp), managed :: x(n), y(n)

    integer :: idx, nstencils, ldm, ldb, istat
    integer, managed :: ia(0:1), ja(0:8)
    real(dp), allocatable, managed :: M(:,:), B(:,:)
    integer, allocatable, managed :: ipiv(:)
    integer, managed :: info

    nstencils = 1
    x = [real(dp) :: 0, 1, 0, -1, 0, 1, 1, -1, -1]
    y = [real(dp) :: 0, 0, 1, 0, -1, 1, -1, -1, 1]

    ia = [0,9]
    ja = [0,1,2,3,4,5,6,7,8]

    x0 = 0.0_dp
    y0 = 0.0_dp

    ldm = nt
    allocate(M(ldm, nt), ipiv(nt))

    ldb = nt
    allocate(B(ldb, nrhs))

    idx = 0
    if (n > NMAX) error stop "Aborting."
    call fill_kernel<<<1,32>>>(nstencils,x,y,ia,ja,M,ldm)

    istat = cudaDeviceSynchronize()
    if (istat /= cudaSuccess) then
        write(*,'(A)') cudaGetErrorString(istat)
        error stop
    end if

    ! Print matrix
    call print_matrix(nt,nt,M,ldm,"M = ")

    call fill_rhs<<<1,32>>>(n,x,y,nrhs,(0.5*x),(0.5*y),B,ldb)

    call solve_dp<<<1,32>>>(M, ipiv, B, info)
    print *, "info (solve) = ", info

    call print_matrix(nt,nrhs,B,ldb,"B = ")

end program
