! Analytic flow fields on a doubly periodic box, for verifying a lattice
! Boltzmann implementation without boundary conditions. This is the
! Fortran counterpart of rbf_flow_benchmarks.h and offers the same cases
! with the same names.
!
! The box comes first: periodic_box holds the side lengths and turns
! integer mode numbers into wave numbers, so every field built on it is
! periodic by construction.
!
! Common interface, on any case:
!
!     call case%fields(x, y, p, ux, uy [, time])
!
! evaluates the fields at the points (x(i), y(i)). The scalar p is the
! pressure, or the density for the two cases defined through the density
! (acoustic_wave, barotropic_vortex). All types extend benchmark_case,
! whose deferred binding this is; the initial-condition cases accept the
! time and ignore it.
!
! Cases:
!   shear_modes       superposition of shear waves with one |k|: exact
!                     decaying solution of the incompressible NS
!                     equations; shear_wave() and taylor_green() build
!                     the two classic instances; body_force holds it
!                     steady (Kolmogorov flow, four-roll mill)
!   acoustic_wave     standing sound wave, linear isothermal acoustics
!   shear_layer       initial condition, roll-up of two tanh layers
!   barotropic_vortex initial condition, convected Gaussian vortex
!
! The constructor functions stop with a message on parameters the
! formulas cannot take (in C++ these are asserts).
module rbf_benchmarks
use rbf_precision, only: wp
implicit none
private

public :: pi
public :: periodic_box
public :: benchmark_case
public :: mode, shear_modes, shear_wave, taylor_green
public :: acoustic_wave
public :: shear_layer
public :: barotropic_vortex

real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)

!
! [0, Lx) x [0, Ly) with periodic images. The wave number of the mode
! (nx, ny) is (2 pi nx/Lx, 2 pi ny/Ly). In lattice units Lx = nx cells.
!
type :: periodic_box
    real(wp) :: Lx, Ly
contains
    procedure :: wavenumber => box_wavenumber
    procedure :: wrap => box_wrap
    procedure :: minimum_image => box_minimum_image
end type

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
! SHEAR MODES
!
! A superposition of transverse plane waves on the box,
!
!     u(x,t) = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m
!
! with e_m = (-ky, kx)/|k| perpendicular to k_m, so every mode is
! divergence free, and all modes sharing the same |k|^2. Then the
! vorticity is |k|^2 times the stream function, the nonlinear term is a
! gradient absorbed by the pressure, and the field is an exact solution
! of the incompressible Navier-Stokes equations that decays with the
! time constant 1/(nu |k|^2). The pressure is
!
!     p = -(|u - U|^2 + (sum_m a_m cos(k_m.x + phi_m))^2)/2 + mean
!
! decaying at twice the rate, normalized to zero mean, which assumes
! the modes have distinct wave vectors (k_m /= +-k_n, checked). The
! uniform drift U (zero by default) moves the pattern without changing
! it, which a Galilean invariant scheme must reproduce.
!
! One mode is the shear wave, the classic viscosity measurement; two
! modes (nx, ny) and (nx, -ny) are the Taylor-Green vortex. Applying
! body_force holds the time-zero field steady: with one mode along y
! that is Kolmogorov flow, with the Taylor-Green pair the four-roll mill.
!
type :: mode
    real(wp) :: amplitude       ! velocity amplitude
    integer :: nx, ny           ! mode numbers on the box
    real(wp) :: phase = 0.0_wp
end type

type, extends(benchmark_case) :: shear_modes
    type(periodic_box) :: box
    type(mode), allocatable :: modes(:)
    real(wp) :: nu
    real(wp) :: drift(2) = 0.0_wp
contains
    procedure, pass(case) :: fields => shear_modes_fields
    procedure, pass(case) :: stress_tensor => shear_modes_stress_tensor
    procedure, pass(case) :: body_force => shear_modes_body_force
    procedure, pass(case) :: ksqr => shear_modes_ksqr
    procedure, pass(case) :: time_constant => shear_modes_time_constant
    procedure, pass(case) :: decay => shear_modes_decay
    procedure, pass(case) :: decay_time => shear_modes_decay_time
    procedure, pass(case) :: viscosity => shear_modes_viscosity
end type

interface shear_modes
    module procedure :: shear_modes_constructor
end interface

!
! ACOUSTIC WAVE
!
! A standing sound wave released from rest, in linear isothermal
! acoustics with the equation of state p = cs^2 rho:
!
!     rho = rho0 (1 + delta cos(k.x + phi) exp(-gamma t)
!                          (cos(W t) + gamma/W sin(W t)))
!     u   = delta cs^2 |k|/W exp(-gamma t) sin(W t) sin(k.x + phi) k/|k|
!
! with the damping rate gamma = (nu + nu_bulk) |k|^2/2 (two dimensions)
! and the frequency W = sqrt(cs^2 |k|^2 - gamma^2). The period gives the
! sound speed, the damping the sum of shear and bulk viscosity, so with
! the shear viscosity from a shear wave it measures the bulk viscosity.
! The BGK collision on D2Q9 has nu_bulk = nu, the default. Valid for
! delta << 1; the velocity amplitude is about delta cs.
!
type, extends(benchmark_case) :: acoustic_wave
    type(periodic_box) :: box
    integer :: nx, ny
    real(wp) :: delta, nu, nu_bulk
    real(wp) :: csqr = 1.0_wp/3.0_wp   ! D2Q9 in lattice units
    real(wp) :: rho0 = 1.0_wp
    real(wp) :: phase = 0.0_wp
contains
    procedure, pass(case) :: fields => acoustic_wave_fields
    procedure, pass(case) :: ksqr => acoustic_wave_ksqr
    procedure, pass(case) :: sound_speed => acoustic_wave_sound_speed
    procedure, pass(case) :: damping_rate => acoustic_wave_damping_rate
    procedure, pass(case) :: frequency => acoustic_wave_frequency
    procedure, pass(case) :: period => acoustic_wave_period
    procedure, pass(case) :: longitudinal_viscosity => acoustic_wave_longitudinal_viscosity
end type

interface acoustic_wave
    module procedure :: acoustic_wave_constructor
end interface

!
! Doubly periodic shear layer: two tanh layers of thickness ~1/k at
! y/Ly = 1/4 and 3/4, perturbed by a sinusoidal uy of amplitude delta
! (Minion & Brown, 1997). Initial condition only; p is returned as zero.
!
type, extends(benchmark_case) :: shear_layer
    type(periodic_box) :: box
    real(wp) :: u0
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
    ! PERIODIC BOX
    !

    pure function box_wavenumber(box,nx,ny) result(k)
        class(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp) :: k(2)
        k = [2*pi*nx/box%Lx, 2*pi*ny/box%Ly]
    end function

    ! The point mapped into the box
    pure function box_wrap(box,xy) result(w)
        class(periodic_box), intent(in) :: box
        real(wp), intent(in) :: xy(2)
        real(wp) :: w(2)
        w = xy - [box%Lx, box%Ly]*floor(xy/[box%Lx, box%Ly])
    end function

    ! The shortest of the displacement and its periodic images
    pure function box_minimum_image(box,d) result(m)
        class(periodic_box), intent(in) :: box
        real(wp), intent(in) :: d(2)
        real(wp) :: m(2)
        m = d - [box%Lx, box%Ly]*anint(d/[box%Lx, box%Ly])
    end function

    !
    ! SHEAR MODES
    !

    function shear_modes_constructor(box,modes,nu,drift) result(case)
        type(periodic_box), intent(in) :: box
        type(mode), intent(in) :: modes(:)
        real(wp), intent(in) :: nu
        real(wp), intent(in), optional :: drift(2)
        type(shear_modes) :: case

        integer :: m, n
        real(wp) :: ksqr

        if (box%Lx <= 0 .or. box%Ly <= 0) error stop "shear_modes: box sides must be positive"
        if (nu <= 0) error stop "shear_modes: viscosity must be positive"
        if (size(modes) == 0) error stop "shear_modes: at least one mode"

        case%box = box
        allocate(case%modes, source=modes)
        case%nu = nu
        if (present(drift)) case%drift = drift

        ksqr = case%ksqr()
        do m = 1, size(modes)
            if (modes(m)%nx == 0 .and. modes(m)%ny == 0) &
                error stop "shear_modes: mode numbers must not both be zero"
            do n = 1, m - 1
                if ((modes(m)%nx == modes(n)%nx .and. modes(m)%ny == modes(n)%ny) .or. &
                    (modes(m)%nx == -modes(n)%nx .and. modes(m)%ny == -modes(n)%ny)) &
                    error stop "shear_modes: wave vectors must be distinct up to sign"
            end do
            if (abs(ksqr_of(box,modes(m)) - ksqr) > 1.0e-12_wp*ksqr) &
                error stop "shear_modes: all modes must share the same |k|^2"
        end do
    end function

    ! One shear wave along the mode (nx, ny) with velocity amplitude u0
    function shear_wave(box,nx,ny,u0,nu,phase,drift) result(case)
        type(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp), intent(in) :: u0, nu
        real(wp), intent(in), optional :: phase, drift(2)
        type(shear_modes) :: case
        real(wp) :: ph
        ph = 0.0_wp
        if (present(phase)) ph = phase
        case = shear_modes(box, [mode(u0, nx, ny, ph)], nu, drift)
    end function

    ! Taylor-Green vortex with the mode numbers (nx, ny) on the box:
    !   ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y)
    !   uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y)
    !   p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y))
    function taylor_green(box,nx,ny,u0,nu,drift) result(case)
        type(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp), intent(in) :: u0, nu
        real(wp), intent(in), optional :: drift(2)
        type(shear_modes) :: case
        real(wp) :: k(2), a
        ! the amplitudes take sqrt(ky/kx), sqrt(kx/ky) and sqrt(kx*ky)
        if (nx*ny <= 0) error stop "taylor_green: mode numbers must be nonzero and of the same sign"
        k = box%wavenumber(nx,ny)
        ! cos(kx x) cos(ky y) = [cos(kx x + ky y) + cos(kx x - ky y)]/2; the
        ! sign carries the convention above, which flips with (kx, ky)
        a = sign(u0, real(nx,wp))*sqrt((k(1)**2 + k(2)**2)/(4*k(1)*k(2)))
        case = shear_modes(box, [mode(a, nx, ny), mode(a, nx, -ny)], nu, drift)
    end function

    pure function ksqr_of(box,m) result(ksqr)
        type(periodic_box), intent(in) :: box
        type(mode), intent(in) :: m
        real(wp) :: ksqr, k(2)
        k = box%wavenumber(m%nx,m%ny)
        ksqr = k(1)**2 + k(2)**2
    end function

    pure function shear_modes_ksqr(case) result(ksqr)
        class(shear_modes), intent(in) :: case
        real(wp) :: ksqr
        ksqr = ksqr_of(case%box,case%modes(1))
    end function

    pure function shear_modes_time_constant(case) result(tau)
        class(shear_modes), intent(in) :: case
        real(wp) :: tau
        tau = 1.0_wp/(case%nu*case%ksqr())
    end function

    ! Decay factor of the velocity at time t
    pure function shear_modes_decay(case,time) result(d)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: time
        real(wp) :: d
        d = exp(-time/case%time_constant())
    end function

    ! Time at which the velocity has dropped to the fraction frac
    pure function shear_modes_decay_time(case,frac) result(t)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: frac
        real(wp) :: t
        t = -case%time_constant()*log(frac)
    end function

    ! Viscosity recovered from the velocity amplitudes a0 and a1 measured
    ! at the times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    pure function shear_modes_viscosity(case,a0,a1,t0,t1) result(nu)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: a0, a1, t0, t1
        real(wp) :: nu
        nu = -log(a1/a0)/(case%ksqr()*(t1 - t0))
    end function

    ! Undecayed velocity of the modes and c = sum_m a_m cos(theta_m),
    ! followed along the drift
    pure subroutine shear_modes_pattern(case,x,y,time,ux,uy,c)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out) :: ux(:), uy(:), c(:)
        real(wp) :: k(2), knorm
        integer :: m
        knorm = sqrt(case%ksqr())
        ux = 0; uy = 0; c = 0
        do m = 1, size(case%modes)
            k = case%box%wavenumber(case%modes(m)%nx,case%modes(m)%ny)
            associate(a => case%modes(m)%amplitude, &
                      th => k(1)*(x - case%drift(1)*time) + k(2)*(y - case%drift(2)*time) &
                            + case%modes(m)%phase)
                ux = ux - a*sin(th)*k(2)/knorm
                uy = uy + a*sin(th)*k(1)/knorm
                c  = c  + a*cos(th)
            end associate
        end do
    end subroutine

    subroutine shear_modes_fields(case,x,y,p,ux,uy,time)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time
        real(wp) :: t, d, mean
        t = time_or_zero(time)
        d = case%decay(t)
        mean = sum(case%modes%amplitude**2)/2   ! of (|u|^2 + c^2)/2 over the box
        call shear_modes_pattern(case,x,y,t,ux,uy,p)
        p  = d*d*(-(ux**2 + uy**2 + p**2)/2 + mean)
        ux = case%drift(1) + d*ux
        uy = case%drift(2) + d*uy
    end subroutine

    ! Strain rate S = (grad u + grad u^T)/2
    subroutine shear_modes_stress_tensor(case,x,y,sxx,sxy,syy,time)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: sxx(:), sxy(:), syy(:)
        real(wp), intent(in), optional :: time
        real(wp) :: t, d, k(2), knorm
        integer :: m
        t = time_or_zero(time)
        d = case%decay(t)
        knorm = sqrt(case%ksqr())
        sxx = 0; sxy = 0
        do m = 1, size(case%modes)
            k = case%box%wavenumber(case%modes(m)%nx,case%modes(m)%ny)
            associate(c => case%modes(m)%amplitude/knorm*cos( &
                      k(1)*(x - case%drift(1)*t) + k(2)*(y - case%drift(2)*t) + case%modes(m)%phase))
                sxx = sxx - c*k(1)*k(2)
                sxy = sxy + c*(k(1)**2 - k(2)**2)/2
            end associate
        end do
        sxx = d*sxx
        sxy = d*sxy
        syy = -sxx
    end subroutine

    ! Body force that holds the time-zero field steady (in the frame
    ! drifting with U): f = nu |k|^2 (u(x, 0) - U) followed along the drift
    subroutine shear_modes_body_force(case,x,y,fx,fy,time)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: fx(:), fy(:)
        real(wp), intent(in), optional :: time
        real(wp) :: c(size(x))
        call shear_modes_pattern(case,x,y,time_or_zero(time),fx,fy,c)
        fx = case%nu*case%ksqr()*fx
        fy = case%nu*case%ksqr()*fy
    end subroutine

    !
    ! ACOUSTIC WAVE
    !

    function acoustic_wave_constructor(box,nx,ny,delta,nu,nu_bulk,csqr,rho0,phase) result(case)
        type(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp), intent(in) :: delta, nu
        real(wp), intent(in), optional :: nu_bulk, csqr, rho0, phase
        type(acoustic_wave) :: case

        if (box%Lx <= 0 .or. box%Ly <= 0) error stop "acoustic_wave: box sides must be positive"
        if (nx == 0 .and. ny == 0) error stop "acoustic_wave: mode numbers must not both be zero"
        if (delta <= 0) error stop "acoustic_wave: relative density amplitude must be positive"
        if (nu <= 0) error stop "acoustic_wave: viscosity must be positive"

        case%box = box
        case%nx = nx
        case%ny = ny
        case%delta = delta
        case%nu = nu
        case%nu_bulk = nu
        if (present(nu_bulk)) case%nu_bulk = nu_bulk
        if (present(csqr)) case%csqr = csqr
        if (present(rho0)) case%rho0 = rho0
        if (present(phase)) case%phase = phase

        if (case%csqr <= 0) error stop "acoustic_wave: squared sound speed must be positive"
        if (case%rho0 <= 0) error stop "acoustic_wave: reference density must be positive"
        if (case%csqr*case%ksqr() <= case%damping_rate()**2) &
            error stop "acoustic_wave: wave must be underdamped"
    end function

    pure function acoustic_wave_ksqr(case) result(ksqr)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: ksqr, k(2)
        k = case%box%wavenumber(case%nx,case%ny)
        ksqr = k(1)**2 + k(2)**2
    end function

    pure function acoustic_wave_sound_speed(case) result(cs)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: cs
        cs = sqrt(case%csqr)
    end function

    pure function acoustic_wave_damping_rate(case) result(gamma)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: gamma
        gamma = (case%nu + case%nu_bulk)*case%ksqr()/2
    end function

    pure function acoustic_wave_frequency(case) result(w)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: w
        w = sqrt(case%csqr*case%ksqr() - case%damping_rate()**2)
    end function

    pure function acoustic_wave_period(case) result(tp)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: tp
        tp = 2*pi/case%frequency()
    end function

    ! Sum of shear and bulk viscosity from a measured damping rate
    pure function acoustic_wave_longitudinal_viscosity(case,gamma) result(nul)
        class(acoustic_wave), intent(in) :: case
        real(wp), intent(in) :: gamma
        real(wp) :: nul
        nul = 2*gamma/case%ksqr()
    end function

    ! p holds the density
    subroutine acoustic_wave_fields(case,x,y,p,ux,uy,time)
        class(acoustic_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time
        real(wp) :: t, k(2), knorm, g, w, e, u(size(x))
        t = time_or_zero(time)
        k = case%box%wavenumber(case%nx,case%ny)
        knorm = sqrt(case%ksqr())
        g = case%damping_rate()
        w = case%frequency()
        e = exp(-g*t)
        associate(th => k(1)*x + k(2)*y + case%phase)
            p = case%rho0*(1 + case%delta*cos(th)*e*(cos(w*t) + g/w*sin(w*t)))
            u = case%delta*case%csqr*knorm/w*e*sin(w*t)*sin(th)
        end associate
        ux = u*k(1)/knorm
        uy = u*k(2)/knorm
    end subroutine

    !
    ! SHEAR LAYER
    !

    subroutine shear_layer_fields(case,x,y,p,ux,uy,time)
        class(shear_layer), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp), intent(in), optional :: time  ! initial condition only, ignored

        uy = vely(x/case%box%Lx,case%u0,case%delta)
        ux = velx(y/case%box%Ly,case%u0,case%k)
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
