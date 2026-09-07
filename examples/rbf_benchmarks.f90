! Analytic flow fields on a doubly periodic box, for verifying solvers
! without external geometry. This is the Fortran counterpart of
! rbf_flow_benchmarks.h and offers the same cases with the same names;
! benchmarks that need a genuine point cloud (cavities, channels,
! cylinders, ...) are deliberately not included.
!
! Common interface, on any case:
!
!     call case%fields(x, y, p, ux, uy [, time])
!
! evaluates the fields at the points (x(i), y(i)). The scalar p is the
! pressure, except for the barotropic vortex where it is the density
! (the case is defined that way, p = cs^2 rho).
!
! The decaying cases (shear_wave, taylor_green) are exact solutions of
! the incompressible Navier-Stokes equations and additionally offer
!
!     ksqr(), time_constant(), decay_time(frac),
!     amplitude(time), viscosity(a0, a1, t0, t1)
!     stress_tensor(x, y, sxx, sxy, syy [, time])
!
! where stress_tensor is the strain rate S = (grad u + grad u^T)/2.
! The other two (shear_layer, barotropic_vortex) are initial conditions
! only: they accept the time argument so that every case conforms to
! the deferred binding of benchmark_case, but ignore it.
!
! For a box of side L the wave numbers must be integer multiples of
! 2*pi/L for the field to be periodic.
module rbf_benchmarks
use rbf_precision, only: wp
implicit none
private

public :: benchmark_case
public :: shear_wave
public :: taylor_green
public :: shear_layer
public :: barotropic_vortex
public :: pi

real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)

type, abstract :: benchmark_case
contains
    procedure(fields_interface), deferred, pass(case) :: fields
end type

abstract interface
    subroutine fields_interface(case,x,y,p,ux,uy,time)
        import :: benchmark_case, wp
        class(benchmark_case), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time
    end subroutine
end interface

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
    procedure, pass(case) :: fields => shear_wave_fields
    procedure, pass(case) :: stress_tensor => shear_wave_stress_tensor
    procedure, pass(case) :: ksqr => shear_wave_ksqr
    procedure, pass(case) :: time_constant => shear_wave_time_constant
    procedure, pass(case) :: decay_time => shear_wave_decay_time
    procedure, pass(case) :: amplitude => shear_wave_amplitude
    procedure, pass(case) :: viscosity => shear_wave_viscosity
end type

!
! Doubly periodic array of counter-rotating vortices with wave numbers
! kx and ky. The amplitudes are chosen so the field is divergence free
! for any kx, ky; the velocity decays with the time constant
! 1/(nu*(kx^2 + ky^2)) and the pressure with half of it.
!
type, extends(benchmark_case) :: taylor_green
    real(wp) :: kx, ky, nu, u0
contains
    procedure, pass(case) :: fields => taylor_green_fields
    procedure, pass(case) :: stress_tensor => taylor_green_stress_tensor
    procedure, pass(case) :: ksqr => taylor_green_ksqr
    procedure, pass(case) :: time_constant => taylor_green_time_constant
    procedure, pass(case) :: decay_time => taylor_green_decay_time
    procedure, pass(case) :: amplitude => taylor_green_amplitude
    procedure, pass(case) :: viscosity => taylor_green_viscosity
end type

!
! Doubly periodic shear layer: two tanh layers of thickness ~1/k at
! y/L = 1/4 and 3/4, perturbed by a sinusoidal uy of amplitude delta
! (Minion & Brown, 1997). Initial condition only; p is returned as zero.
!
type, extends(benchmark_case) :: shear_layer
    real(wp) :: u0, L
    real(wp) :: k = 80.0_wp
    real(wp) :: delta = 0.05_wp
contains
    procedure, pass(case) :: fields => shear_layer_fields
end type

!
! Barotropic vortex of Wissocq, Boussuge & Sagaut (2020),
! Phys. Rev. E 101, 043306. Initial condition only; returns the density
! in place of the pressure (p = cs^2 rho), csqr being the squared sound
! speed of the lattice.
!
type, extends(benchmark_case) :: barotropic_vortex
    real(wp) :: U0, cx, cy, Rc, eps
    real(wp) :: rho0 = 1.0_wp
    real(wp) :: csqr = 1.0_wp/3.0_wp  ! D2Q9 in lattice units
contains
    procedure, pass(case) :: fields => barotropic_vortex_fields
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

    ! Velocity amplitude at time t
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

    subroutine shear_wave_fields(case,x,y,p,ux,uy,time)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: knorm, a

        knorm = sqrt(case%ksqr())
        a = case%amplitude(time_or_zero(time))

        associate(s => a*sin(case%kx*x + case%ky*y + case%phase))
            ux = -s*case%ky/knorm
            uy =  s*case%kx/knorm
        end associate
        p = 0.0_wp

    end subroutine

    ! Strain rate S = (grad u + grad u^T)/2
    subroutine shear_wave_stress_tensor(case,x,y,sxx,sxy,syy,time)
        class(shear_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: sxx(:), sxy(:), syy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: knorm, a

        knorm = sqrt(case%ksqr())
        a = case%amplitude(time_or_zero(time))

        ! grad u = u0 cos(k.x) e_perp (x) k
        associate(c => a*cos(case%kx*x + case%ky*y + case%phase)/knorm)
            sxx = -c*case%kx*case%ky
            sxy = 0.5_wp*c*(case%kx**2 - case%ky**2)
            syy = -sxx
        end associate

    end subroutine

    !
    ! TAYLOR-GREEN VORTEX
    !

    pure function taylor_green_ksqr(case) result(ksqr)
        class(taylor_green), intent(in) :: case
        real(wp) :: ksqr
        ksqr = case%kx**2 + case%ky**2
    end function

    pure function taylor_green_time_constant(case) result(tau)
        class(taylor_green), intent(in) :: case
        real(wp) :: tau
        tau = 1.0_wp/(case%nu*case%ksqr())
    end function

    ! Time at which the amplitude has dropped to the fraction frac of u0
    pure function taylor_green_decay_time(case,frac) result(t)
        class(taylor_green), intent(in) :: case
        real(wp), intent(in) :: frac
        real(wp) :: t
        t = -case%time_constant()*log(frac)
    end function

    ! Velocity amplitude at time t
    pure function taylor_green_amplitude(case,time) result(a)
        class(taylor_green), intent(in) :: case
        real(wp), intent(in) :: time
        real(wp) :: a
        a = case%u0*exp(-time/case%time_constant())
    end function

    ! Viscosity recovered from the amplitudes a0 and a1 measured at the
    ! times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    pure function taylor_green_viscosity(case,a0,a1,t0,t1) result(nu)
        class(taylor_green), intent(in) :: case
        real(wp), intent(in) :: a0, a1, t0, t1
        real(wp) :: nu
        nu = -log(a1/a0)/(case%ksqr()*(t1 - t0))
    end function

    subroutine taylor_green_fields(case,x,y,p,ux,uy,time)
        class(taylor_green), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: a, kykx, kxky

        a = case%amplitude(time_or_zero(time))
        kykx = case%ky/case%kx
        kxky = case%kx/case%ky

        associate(kx => case%kx, ky => case%ky)
            ux = -a*sqrt(kykx)*cos(kx*x)*sin(ky*y)
            uy =  a*sqrt(kxky)*sin(kx*x)*cos(ky*y)
            p  = -0.25_wp*a**2*(kykx*cos(2*kx*x) + kxky*cos(2*ky*y))
        end associate

    end subroutine

    ! Strain rate S = (grad u + grad u^T)/2
    subroutine taylor_green_stress_tensor(case,x,y,sxx,sxy,syy,time)
        class(taylor_green), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: sxx(:), sxy(:), syy(:)
        real(wp), intent(in), optional :: time

        real(wp) :: a

        a = case%amplitude(time_or_zero(time))

        associate(kx => case%kx, ky => case%ky)
            sxx = a*sqrt(kx*ky)*sin(kx*x)*sin(ky*y)
            sxy = 0.5_wp*a*(sqrt(kx**3/ky) - sqrt(ky**3/kx))*cos(kx*x)*cos(ky*y)
            syy = -sxx  ! follows from div u = 0
        end associate

    end subroutine

    !
    ! SHEAR LAYER
    !

    subroutine shear_layer_fields(case,x,y,p,ux,uy,time)
        class(shear_layer), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time  ! initial condition only, ignored

        uy = vely(x/case%L,case%u0,case%delta)
        ux = velx(y/case%L,case%u0,case%k)
        p = 0.0_wp

        if (present(time)) continue  ! silences the unused-argument warning

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

    subroutine barotropic_vortex_fields(case,x,y,p,ux,uy,time)
        class(barotropic_vortex), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)  ! p holds the density
        real(wp), intent(in), optional :: time  ! initial condition only, ignored

        real(wp) :: ssqr, mach

        associate(xc => x - case%cx, yc => y - case%cy)

            ssqr = 2*case%rc**2
            ux  = case%U0 - case%eps * (yc/case%rc) * bell(xc,yc,sigmasqr=ssqr)
            uy  =           case%eps * (xc/case%rc) * bell(xc,yc,sigmasqr=ssqr)

            mach = case%eps / sqrt(case%csqr)  ! Ma = eps / cs

            !
            ! \rho(r) = \rho_0 * \exp(-eps^2/(2 c_s^2) \exp(-r^2/R_c^2))
            !
            p = case%rho0 * exp(-0.5_wp * mach**2 * bell(xc,yc,sigmasqr=ssqr/2.0_wp))

        end associate

        if (present(time)) continue  ! silences the unused-argument warning

    contains

        elemental function bell(x,y,sigmasqr)
            real(wp), intent(in) :: x, y, sigmasqr
            real(wp) :: bell
            bell = exp(-(x**2 + y**2) / sigmasqr )
        end function

    end subroutine

    !
    ! HELPERS
    !

    pure function time_or_zero(time) result(t)
        real(wp), intent(in), optional :: time
        real(wp) :: t
        t = 0.0_wp
        if (present(time)) t = time
    end function

end module
