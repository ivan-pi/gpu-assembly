module rbf_benchmarks
use rbf_precision, only: wp
implicit none
private

public :: benchmark_case
public :: shear_layer
public :: barotropic_vortex

public :: velocity

type, abstract :: benchmark_case
end type

type, extends(benchmark_case) :: shear_layer
    real(wp) :: u0, L
    real(wp) :: k = 80.0_wp
    real(wp) :: delta = 0.05_wp
contains
    procedure, pass(case) :: velocity => velocity
end type

real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)


type, extends(benchmark_case) :: barotropic_vortex
    real(wp) :: U0, cx, cy, Rc, eps, rho0 = 1.0_wp
contains
    procedure, pass(case) :: set_fields
end type

contains

    subroutine velocity(case,x,y,ux,uy)
        class(shear_layer), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: ux(:), uy(:)

        real(wp) :: xd, yd
        integer :: i

        uy = vely(x/case%L,case%u0,case%k,case%delta)
        ux = velx(y/case%L,case%u0,case%k,case%delta)

    contains

        elemental function velx(y,u0,k,delta)
            real(wp), intent(in) :: y, u0, k, delta
            real(wp) :: velx
            if (y <= 0.5_wp) then
                velx = u0*tanh(k*(y - 0.25_wp))
            else
                velx = u0*tanh(k*(0.75_wp - y))
            end if
        end function

        elemental function vely(x,u0,k,delta)
            real(wp), intent(in) :: x, u0, k, delta
            real(wp) :: vely
            vely = u0*delta*sin(2*pi*(x + 0.25_wp))
        end function

    end subroutine


    subroutine set_fields(case,x,y,rho,ux,uy,csqr)
        class(barotropic_vortex), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: rho(:), ux(:), uy(:)
        real(wp), intent(in) :: csqr

        real(wp) :: ssqr, mach

        associate(xc => x - case%cx, yc => y - case%cy)        

            ssqr = 2*case%rc**2
            ux  = case%U0 - case%eps * (yc/case%rc) * bell(xc,yc,sigmasqr=ssqr)
            uy  =           case%eps * (xc/case%rc) * bell(xc,yc,sigmasqr=ssqr)

            mach = case%eps / sqrt(csqr)  ! Ma = eps / cs

            !
            ! \rho(r) = \rho_0 * \exp(-eps^2/(2 c_s^2) \exp(-r^2/R_c^2))
            !
            rho = case%rho0 * exp(-0.5_wp * mach**2 * bell(xc,yc,sigmasqr=ssqr/2.0_wp))

        end associate

    contains

        elemental function bell(x,y,sigmasqr)
            real(wp), intent(in) :: x, y, sigmasqr
            real(wp) :: bell
            bell = exp(-(x**2 + y**2) / sigmasqr )
        end function

    end subroutine

end module
