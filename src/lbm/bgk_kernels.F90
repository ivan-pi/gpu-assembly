! bgk_kernels.F90 -- D2Q9 BGK collision kernels (CUDA Fortran)
!
! Collision step of the lattice Boltzmann BGK model on the D2Q9 lattice,
! one thread per node. bgk_kernel_split has a C++ twin of the same name
! in bgk_kernels.cu; the two are kept in step line by line.
!
! Lattice directions c_a and weights w_a:
!
!     6  2  5       0           rest             w0 = 4/9
!      \ | /        1, 2, 3, 4  E, N, W, S       ws = 1/9
!     3--0--1       5, 6, 7, 8  NE, NW, SW, SE   wd = 1/36
!      / | \
!     7  4  8
!
! Equilibrium, with the factor 3 folded into omega_w0/ws/wd:
!
!     feq_a = 3 w_a rho [ 1/3 - u.u/2 + c_a.u + 3/2 (c_a.u)^2 ]
!                        \__ indp __/
!
! Storage is planar (structure of arrays): direction a of node i is
! pdf(i,a) in the rank-2 form and pdf(i + n*a) in the flat form; both
! address the same memory. The two kernels differ only in that:
!
!   bgk_kernel        takes pdf(n,0:8)
!   bgk_kernel_split  takes the flat pdf(9*n) and also stores indp, the
!                     direction-independent part of the equilibrium
!
! Build with nvfortran -cuda. Without -cuda the kernels are dropped and
! bgk_collide falls back to d2q9_collide from the collision_bgk module,
! which is not part of this repository.

module bgk_kernels
    implicit none
    private

    public :: wp, w0, ws, wd
    public :: bgk_collide
#ifdef _CUDA
    public :: bgk_kernel, bgk_kernel_split
#endif

    ! working precision; matches real_t in bgk_kernels.cu
    integer, parameter :: wp = kind(1.0e0)

    ! D2Q9 weights: rest, straight (axis), diagonal
    real(wp), parameter :: w0 = 4.0_wp / 9.0_wp
    real(wp), parameter :: ws = 1.0_wp / 9.0_wp
    real(wp), parameter :: wd = 1.0_wp / 36.0_wp

    real(wp), parameter :: one_third    = 1.0_wp / 3.0_wp
    real(wp), parameter :: one_half     = 1.0_wp / 2.0_wp
    real(wp), parameter :: three_halves = 3.0_wp / 2.0_wp

contains

    ! Collide all n nodes: launches bgk_kernel, or the host routine
    ! d2q9_collide when built without CUDA.
    subroutine bgk_collide(n, omega, pdf, rho, ux, uy)
#ifndef _CUDA
        use collision_bgk, only: d2q9_collide
#endif
        integer, intent(in) :: n
        real(wp), intent(in) :: omega
        real(wp), intent(inout) :: pdf(n,0:8)
        real(wp), intent(out) :: rho(n), ux(n), uy(n)
!@cuf   attributes(managed) :: pdf, rho, ux, uy

#ifdef _CUDA
        integer, parameter :: tpb = 32   ! threads per block

        call bgk_kernel<<<(n + tpb - 1) / tpb, tpb>>>(n, omega, pdf, rho, ux, uy)
#else
        call d2q9_collide(n, rho, ux, uy, pdf, omega)
#endif

    end subroutine bgk_collide

#ifdef _CUDA

    ! One node per thread; pdf(i,a) is direction a of node i.
    attributes(global) subroutine bgk_kernel(n, omega, pdf, rho, ux, uy)
        integer, value :: n
        real(wp), value :: omega
        real(wp), intent(inout) :: pdf(n,0:8)
        real(wp), intent(out) :: rho(n), ux(n), uy(n)

        real(wp) :: omegabar, omega_w0, omega_ws, omega_wd
        real(wp) :: f(0:8)
        real(wp) :: rho_i, invrho, ux_i, uy_i, uxsq, uysq, indp_i
        real(wp) :: velxpy, velxmy
        real(wp) :: vel_trm_13, vel_trm_24, vel_trm_57, vel_trm_68
        integer :: i

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        if (i > n) return

        omegabar = 1.0_wp - omega
        omega_w0 = 3.0_wp * omega * w0
        omega_ws = 3.0_wp * omega * ws
        omega_wd = 3.0_wp * omega * wd

        ! populations of node i
        f = pdf(i,0:8)

        ! density
        rho_i = (((f(5) + f(7)) + (f(6) + f(8))) + ((f(1) + f(3)) + (f(2) + f(4)))) + f(0)
        rho(i) = rho_i
        invrho = 1.0_wp / rho_i

        ! velocity
        ux_i = invrho * (((f(5) - f(7)) + (f(8) - f(6))) + (f(1) - f(3)))
        uy_i = invrho * (((f(5) - f(7)) + (f(6) - f(8))) + (f(2) - f(4)))
        ux(i) = ux_i
        uy(i) = uy_i
        uxsq = ux_i * ux_i
        uysq = uy_i * uy_i

        ! direction-independent part of the equilibrium
        indp_i = one_third - one_half * (uxsq + uysq)

        ! relax towards equilibrium; opposite directions share the even terms
        pdf(i,0) = omegabar * f(0) + omega_w0 * rho_i * indp_i

        vel_trm_13 = indp_i + three_halves * uxsq
        pdf(i,1) = omegabar * f(1) + omega_ws * rho_i * (vel_trm_13 + ux_i)
        pdf(i,3) = omegabar * f(3) + omega_ws * rho_i * (vel_trm_13 - ux_i)

        vel_trm_24 = indp_i + three_halves * uysq
        pdf(i,2) = omegabar * f(2) + omega_ws * rho_i * (vel_trm_24 + uy_i)
        pdf(i,4) = omegabar * f(4) + omega_ws * rho_i * (vel_trm_24 - uy_i)

        velxpy = ux_i + uy_i
        vel_trm_57 = indp_i + three_halves * velxpy * velxpy
        pdf(i,5) = omegabar * f(5) + omega_wd * rho_i * (vel_trm_57 + velxpy)
        pdf(i,7) = omegabar * f(7) + omega_wd * rho_i * (vel_trm_57 - velxpy)

        velxmy = ux_i - uy_i
        vel_trm_68 = indp_i + three_halves * velxmy * velxmy
        pdf(i,6) = omegabar * f(6) + omega_wd * rho_i * (vel_trm_68 - velxmy)
        pdf(i,8) = omegabar * f(8) + omega_wd * rho_i * (vel_trm_68 + velxmy)

    end subroutine bgk_kernel

    ! One node per thread; pdf(i + n*a) is direction a of node i and indp
    ! receives the direction-independent part of the equilibrium.
    attributes(global) subroutine bgk_kernel_split(n, omega, pdf, rho, ux, uy, indp)
        integer, value :: n
        real(wp), value :: omega
        real(wp), intent(inout) :: pdf(9*n)
        real(wp), intent(out) :: rho(n), ux(n), uy(n), indp(n)

        real(wp) :: omegabar, omega_w0, omega_ws, omega_wd
        real(wp) :: f(0:8)
        real(wp) :: rho_i, invrho, ux_i, uy_i, uxsq, uysq, indp_i
        real(wp) :: velxpy, velxmy
        real(wp) :: vel_trm_13, vel_trm_24, vel_trm_57, vel_trm_68
        integer :: i, a

        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        if (i > n) return

        omegabar = 1.0_wp - omega
        omega_w0 = 3.0_wp * omega * w0
        omega_ws = 3.0_wp * omega * ws
        omega_wd = 3.0_wp * omega * wd

        ! populations of node i
        do a = 0, 8
            f(a) = pdf(i + n*a)
        end do

        ! density
        rho_i = (((f(5) + f(7)) + (f(6) + f(8))) + ((f(1) + f(3)) + (f(2) + f(4)))) + f(0)
        rho(i) = rho_i
        invrho = 1.0_wp / rho_i

        ! velocity
        ux_i = invrho * (((f(5) - f(7)) + (f(8) - f(6))) + (f(1) - f(3)))
        uy_i = invrho * (((f(5) - f(7)) + (f(6) - f(8))) + (f(2) - f(4)))
        ux(i) = ux_i
        uy(i) = uy_i
        uxsq = ux_i * ux_i
        uysq = uy_i * uy_i

        ! direction-independent part of the equilibrium
        indp_i = one_third - one_half * (uxsq + uysq)
        indp(i) = indp_i

        ! relax towards equilibrium; opposite directions share the even terms
        pdf(i) = omegabar * f(0) + omega_w0 * rho_i * indp_i

        vel_trm_13 = indp_i + three_halves * uxsq
        pdf(i + n*1) = omegabar * f(1) + omega_ws * rho_i * (vel_trm_13 + ux_i)
        pdf(i + n*3) = omegabar * f(3) + omega_ws * rho_i * (vel_trm_13 - ux_i)

        vel_trm_24 = indp_i + three_halves * uysq
        pdf(i + n*2) = omegabar * f(2) + omega_ws * rho_i * (vel_trm_24 + uy_i)
        pdf(i + n*4) = omegabar * f(4) + omega_ws * rho_i * (vel_trm_24 - uy_i)

        velxpy = ux_i + uy_i
        vel_trm_57 = indp_i + three_halves * velxpy * velxpy
        pdf(i + n*5) = omegabar * f(5) + omega_wd * rho_i * (vel_trm_57 + velxpy)
        pdf(i + n*7) = omegabar * f(7) + omega_wd * rho_i * (vel_trm_57 - velxpy)

        velxmy = ux_i - uy_i
        vel_trm_68 = indp_i + three_halves * velxmy * velxmy
        pdf(i + n*6) = omegabar * f(6) + omega_wd * rho_i * (vel_trm_68 - velxmy)
        pdf(i + n*8) = omegabar * f(8) + omega_wd * rho_i * (vel_trm_68 + velxmy)

    end subroutine bgk_kernel_split

#endif

end module bgk_kernels
