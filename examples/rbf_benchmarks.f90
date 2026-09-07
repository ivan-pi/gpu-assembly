! Analytic flow fields on a doubly periodic box, for verifying solvers
! without external geometry. This is the Fortran counterpart of
! rbf_flow_benchmarks.h; benchmarks that need a genuine point cloud
! (cavities, channels, cylinders, ...) are deliberately not included.
module rbf_benchmarks
use rbf_precision, only: wp
implicit none
private

public :: benchmark_case
public :: shear_wave
public :: shear_layer
public :: barotropic_vortex

real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)

type, abstract :: benchmark_case
end type

!
! Transverse plane wave with wave vector (kx, ky),
!
!     u(x,t) = u0 * sin(kx*x + ky*y + phase) * exp(-nu*|k|^2*t) * e_perp
!
! with e_perp = (-ky, kx)/|k|, so the field is divergence free and the
! pressure stays uniform. The amplitude decays with the time constant
! 1/(nu*|k|^2); measuring it at two instants recovers the viscosity.
!
type, extends(benchmark_case) :: shear_wave
    real(wp) :: kx, ky, nu, u0
    real(wp) :: phase = 0.0_wp
contains
    procedure, pass(case) :: ksqr => shear_wave_ksqr
    procedure, pass(case) :: time_constant => shear_wave_time_constant
    procedure, pass(case) :: decay_time => shear_wave_decay_time
    procedure, pass(case) :: amplitude => shear_wave_amplitude
    procedure, pass(case) :: viscosity => shear_wave_viscosity
    procedure, pass(case) :: velocity => shear_wave_velocity
    procedure, pass(case) :: stress_tensor => shear_wave_stress_tensor
end type

!
! Doubly periodic shear layer: two tanh layers of thickness ~1/k at
! y/L = 1/4 and 3/4, perturbed by a sinusoidal uy of amplitude delta.
!
type, extends(benchmark_case) :: shear_layer
    real(wp) :: u0, L
    real(wp) :: k = 80.0_wp
    real(wp) :: delta = 0.05_wp
contains
    procedure, pass(case) :: velocity => shear_layer_velocity
end type

!
! Barotropic vortex of Wissocq, Boussuge & Sagaut (2020),
! Phys. Rev. E 101, 043306.
!
type, extends(benchmark_case) :: barotropic_vortex
    real(wp) :: U0, cx, cy, Rc, eps, rho0 = 1.0_wp
contains
    procedure, pass(case) :: set_fields
end type

contains

    !
    ! SHEAR WAVE
    !

    pure function shear_wave_ksqr(case) result(ksqr)
        class(shear_wave), intent(in) :: case
        real(wp) :: ksqr
        ksqr = case%kx**2 + case%ky**2
    end function

    pure function shear_wave_time_constant(case) result(tau)
        class(shear_wave), intent(in) :: case
        real(wp) :: tau
        tau = 1.0_wp/(case%nu*case%ksqr())
    end function

    ! Time at which the amplitude has dropped to the fraction frac of u0
    pure function shear_wave_decay_time(case,frac) result(t)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: frac
        real(wp) :: t
        t = -case%time_constant()*log(frac)
    end function

    pure function shear_wave_amplitude(case,time) result(a)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: time
        real(wp) :: a
        a = case%u0*exp(-time/case%time_constant())
    end function

    ! Viscosity recovered from the amplitudes a0 and a1 measured at the
    ! times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    pure function shear_wave_viscosity(case,a0,a1,t0,t1) result(nu)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: a0, a1, t0, t1
        real(wp) :: nu
        nu = -log(a1/a0)/(case%ksqr()*(t1 - t0))
    end function

    subroutine shear_wave_velocity(case,x,y,ux,uy,time)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: ux(:), uy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: t, knorm, a

        t = 0.0_wp
        if (present(time)) t = time

        knorm = sqrt(case%ksqr())
        a = case%amplitude(t)

        associate(s => a*sin(case%kx*x + case%ky*y + case%phase))
            ux = -s*case%ky/knorm
            uy =  s*case%kx/knorm
        end associate

    end subroutine

    ! Strain rate S = (grad u + grad u^T)/2
    subroutine shear_wave_stress_tensor(case,x,y,sxx,sxy,syy,time)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: sxx(:), sxy(:), syy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: t, knorm, a

        t = 0.0_wp
        if (present(time)) t = time

        knorm = sqrt(case%ksqr())
        a = case%amplitude(t)

        ! grad u = u0 cos(k.x) e_perp (x) k
        associate(c => a*cos(case%kx*x + case%ky*y + case%phase)/knorm)
            sxx = -c*case%kx*case%ky
            sxy = 0.5_wp*c*(case%kx**2 - case%ky**2)
            syy = -sxx
        end associate

    end subroutine

    !
    ! SHEAR LAYER
    !

    subroutine shear_layer_velocity(case,x,y,ux,uy)
        class(shear_layer), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: ux(:), uy(:)

        uy = vely(x/case%L,case%u0,case%delta)
        ux = velx(y/case%L,case%u0,case%k)

    contains

        elemental function velx(y,u0,k)
            real(wp), intent(in) :: y, u0, k
            real(wp) :: velx
            if (y <= 0.5_wp) then
                velx = u0*tanh(k*(y - 0.25_wp))
            else
                velx = u0*tanh(k*(0.75_wp - y))
            end if
        end function

        elemental function vely(x,u0,delta)
            real(wp), intent(in) :: x, u0, delta
            real(wp) :: vely
            vely = u0*delta*sin(2*pi*(x + 0.25_wp))
        end function

    end subroutine

    !
    ! BAROTROPIC VORTEX
    !

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
