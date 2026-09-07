! Analytic flow fields on a doubly periodic box, for verifying a lattice
! Boltzmann implementation without boundary conditions. The physics and
! the measurement recipes are in docs/periodic_benchmarks.md; the C++
! header rbf_flow_benchmarks.h offers the same cases.
!
!     call case%fields(x, y, time, p, ux, uy)   ! time-dependent cases
!     call case%fields(x, y, p, ux, uy)         ! initial conditions
!
! evaluates the fields at the points (x(i), y(i)). The scalar p is the
! pressure, or the density for acoustic_wave and barotropic_vortex. The
! constructor functions stop with a message on parameters the formulas
! cannot take (the C++ constructors assert the same).
module rbf_benchmarks
use rbf_precision, only: wp
implicit none
private

public :: pi
public :: periodic_box
public :: mode, shear_modes, shear_wave, taylor_green
public :: acoustic_wave
public :: shear_layer
public :: barotropic_vortex

real(wp), parameter :: pi = 4.0_wp*atan(1.0_wp)

!
! [0, Lx) x [0, Ly) with periodic images. The mode (nx, ny) has the wave
! number (2 pi nx/Lx, 2 pi ny/Ly); in lattice units Lx = nx cells.
!
type :: periodic_box
    real(wp) :: Lx, Ly
contains
    procedure :: wavenumber => box_wavenumber
    procedure :: wrap => box_wrap
    procedure :: minimum_image => box_minimum_image
end type

interface periodic_box
    module procedure :: periodic_box_constructor
end interface

!
! SHEAR MODES: shear waves sharing one |k|, an exact decaying solution
! of the incompressible Navier-Stokes equations,
!
!     u = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m,
!     e_m = (-ky, kx)/|k|,
!     p = -(|u - U|^2 + (sum_m a_m cos(...))^2)/2 + mean.
!
! One mode is the shear wave, the pair (nx, ny), (nx, -ny) the
! Taylor-Green vortex; see shear_wave() and taylor_green() below.
! body_force gives the force that holds the time-zero field steady
! (Kolmogorov flow, four-roll mill).
!
type :: mode
    real(wp) :: amplitude       ! velocity amplitude
    integer :: nx, ny           ! mode numbers on the box
    real(wp) :: phase = 0.0_wp
end type

type :: shear_modes
    type(periodic_box) :: box
    type(mode), allocatable :: modes(:)
    real(wp) :: nu
    real(wp) :: drift(2) = 0.0_wp   ! uniform velocity carrying the pattern
contains
    procedure :: fields => shear_modes_fields
    procedure :: stress_tensor => shear_modes_stress_tensor
    procedure :: body_force => shear_modes_body_force
    procedure :: ksqr => shear_modes_ksqr
    procedure :: time_constant => shear_modes_time_constant
    procedure :: decay => shear_modes_decay
    procedure :: decay_time => shear_modes_decay_time
end type

interface shear_modes
    module procedure :: shear_modes_constructor
end interface

!
! ACOUSTIC WAVE: standing sound wave released from rest, in linear
! isothermal acoustics (p = cs^2 rho, delta << 1),
!
!     rho = rho0 (1 + delta cos(k.x + phi) e^{-gamma t} (cos Wt + gamma/W sin Wt))
!     u   = delta cs^2 |k|/W e^{-gamma t} sin Wt sin(k.x + phi) k/|k|
!     gamma = (nu + nu_bulk) |k|^2/2,   W = sqrt(cs^2 |k|^2 - gamma^2)
!
! nu_bulk defaults to nu, the value of the BGK collision on D2Q9.
!
type :: acoustic_wave
    type(periodic_box) :: box
    integer :: nx, ny
    real(wp) :: delta, nu, nu_bulk
    real(wp) :: csqr = 1.0_wp/3.0_wp   ! D2Q9 in lattice units
    real(wp) :: rho0 = 1.0_wp
    real(wp) :: phase = 0.0_wp
contains
    procedure :: fields => acoustic_wave_fields
    procedure :: ksqr => acoustic_wave_ksqr
    procedure :: damping_rate => acoustic_wave_damping_rate
    procedure :: frequency => acoustic_wave_frequency
    procedure :: period => acoustic_wave_period
end type

interface acoustic_wave
    module procedure :: acoustic_wave_constructor
end interface

!
! SHEAR LAYER (Minion & Brown, 1997): two tanh layers at y/Ly = 1/4 and
! 3/4 of thickness ~1/k, perturbed by uy = u0 delta sin(2 pi (x/Lx + 1/4)).
! Initial condition only; p is returned as zero.
!
type :: shear_layer
    type(periodic_box) :: box
    real(wp) :: u0
    real(wp) :: k = 80.0_wp
    real(wp) :: delta = 0.05_wp
contains
    procedure :: fields => shear_layer_fields
end type

!
! BAROTROPIC VORTEX (Wissocq, Boussuge & Sagaut, Phys. Rev. E 101,
! 043306 (2020), Eqs. 3, 4, 20): Gaussian vortex of radius Rc and
! strength eps convected at U0, with the density in balance with the
! athermal equation of state p = cs^2 rho. Initial condition only; p
! holds the density. Distances to the centre are taken through the
! periodic images.
!
type :: barotropic_vortex
    type(periodic_box) :: box
    real(wp) :: U0, center(2), Rc, eps
    real(wp) :: rho0 = 1.0_wp
    real(wp) :: csqr = 1.0_wp/3.0_wp  ! D2Q9 in lattice units
contains
    procedure :: fields => barotropic_vortex_fields
end type

contains

    !
    ! PERIODIC BOX
    !

    function periodic_box_constructor(Lx,Ly) result(box)
        real(wp), intent(in) :: Lx, Ly
        type(periodic_box) :: box
        if (Lx <= 0 .or. Ly <= 0) error stop "periodic_box: box sides must be positive"
        box%Lx = Lx
        box%Ly = Ly
    end function

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
        w(1) = xy(1) - box%Lx*floor(xy(1)/box%Lx)
        w(2) = xy(2) - box%Ly*floor(xy(2)/box%Ly)
    end function

    ! The shortest of the displacement and its periodic images
    pure function box_minimum_image(box,d) result(m)
        class(periodic_box), intent(in) :: box
        real(wp), intent(in) :: d(2)
        real(wp) :: m(2)
        m(1) = d(1) - box%Lx*anint(d(1)/box%Lx)
        m(2) = d(2) - box%Ly*anint(d(2)/box%Ly)
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
            if (abs(ksqr_of(box,modes(m)) - ksqr) > 8*epsilon(ksqr)*ksqr) &
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

    ! Taylor-Green vortex, the mode pair (nx, ny), (nx, -ny), which is
    !   ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y)
    !   uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y)
    !   p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y))
    ! with kx, ky the wave numbers of |nx|, |ny|; the signs of the mode
    ! numbers do not matter.
    function taylor_green(box,nx,ny,u0,nu,drift) result(case)
        type(periodic_box), intent(in) :: box
        integer, intent(in) :: nx, ny
        real(wp), intent(in) :: u0, nu
        real(wp), intent(in), optional :: drift(2)
        type(shear_modes) :: case
        real(wp) :: k(2), a
        if (nx == 0 .or. ny == 0) error stop "taylor_green: needs both mode numbers"
        k = box%wavenumber(abs(nx),abs(ny))
        a = u0*sqrt((k(1)**2 + k(2)**2)/(4*k(1)*k(2)))
        case = shear_modes(box, [mode(a, abs(nx), abs(ny)), mode(a, abs(nx), -abs(ny))], nu, drift)
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

    ! Undecayed sums over the modes, without the drift, followed along
    ! it: velocity (ux, uy), c = sum_m a_m cos(theta_m), and the strain
    ! rate (sxx, sxy); each output only when asked for
    pure subroutine shear_modes_evaluate(case,x,y,time,ux,uy,c,sxx,sxy)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out), optional :: ux(:), uy(:), c(:), sxx(:), sxy(:)

        integer :: i, m, nm
        real(wp) :: knorm, k(2), th, sn, cs, s1, s2, s3, s4, s5
        real(wp), dimension(size(case%modes)) :: kx, ky, ex, ey, a, ph

        nm = size(case%modes)
        knorm = sqrt(case%ksqr())
        do m = 1, nm
            k = case%box%wavenumber(case%modes(m)%nx,case%modes(m)%ny)
            kx(m) = k(1); ky(m) = k(2)
            a(m) = case%modes(m)%amplitude
            ph(m) = case%modes(m)%phase
            ex(m) = -a(m)*ky(m)/knorm
            ey(m) =  a(m)*kx(m)/knorm
        end do

        do i = 1, size(x)
            s1 = 0; s2 = 0; s3 = 0; s4 = 0; s5 = 0
            do m = 1, nm
                th = kx(m)*(x(i) - case%drift(1)*time) + ky(m)*(y(i) - case%drift(2)*time) + ph(m)
                sn = sin(th); cs = cos(th)
                s1 = s1 + sn*ex(m)
                s2 = s2 + sn*ey(m)
                s3 = s3 + cs*a(m)
                s4 = s4 - cs*a(m)*kx(m)*ky(m)/knorm
                s5 = s5 + cs*a(m)*(kx(m)**2 - ky(m)**2)/(2*knorm)
            end do
            if (present(ux)) ux(i) = s1
            if (present(uy)) uy(i) = s2
            if (present(c)) c(i) = s3
            if (present(sxx)) sxx(i) = s4
            if (present(sxy)) sxy(i) = s5
        end do
    end subroutine

    subroutine shear_modes_fields(case,x,y,time,p,ux,uy)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        call shear_modes_evaluate(case,x,y,time,ux=ux,uy=uy,c=p)
        associate(d => case%decay(time), mean => sum(case%modes%amplitude**2)/2)
            p  = d*d*(-(ux**2 + uy**2 + p**2)/2 + mean)
            ux = case%drift(1) + d*ux
            uy = case%drift(2) + d*uy
        end associate
    end subroutine

    ! Strain rate S = (grad u + grad u^T)/2
    subroutine shear_modes_stress_tensor(case,x,y,time,sxx,sxy,syy)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out) :: sxx(:), sxy(:), syy(:)
        call shear_modes_evaluate(case,x,y,time,sxx=sxx,sxy=sxy)
        associate(d => case%decay(time))
            sxx = d*sxx
            sxy = d*sxy
        end associate
        syy = -sxx
    end subroutine

    ! Body force holding the time-zero field steady in the drifting
    ! frame: f = nu |k|^2 (u(x, 0) - U), followed along the drift
    subroutine shear_modes_body_force(case,x,y,time,fx,fy)
        class(shear_modes), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out) :: fx(:), fy(:)
        call shear_modes_evaluate(case,x,y,time,ux=fx,uy=fy)
        associate(f => case%nu*case%ksqr())
            fx = f*fx
            fy = f*fy
        end associate
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

        if (case%nu_bulk < 0) error stop "acoustic_wave: bulk viscosity must not be negative"
        if (case%csqr <= 0 .or. case%rho0 <= 0) &
            error stop "acoustic_wave: sound speed and density must be positive"
        if (case%csqr*case%ksqr() <= case%damping_rate()**2) &
            error stop "acoustic_wave: wave must be underdamped"
    end function

    pure function acoustic_wave_ksqr(case) result(ksqr)
        class(acoustic_wave), intent(in) :: case
        real(wp) :: ksqr, k(2)
        k = case%box%wavenumber(case%nx,case%ny)
        ksqr = k(1)**2 + k(2)**2
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

    ! p holds the density
    subroutine acoustic_wave_fields(case,x,y,time,p,ux,uy)
        class(acoustic_wave), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:), time
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp) :: k(2), u
        integer :: i
        k = case%box%wavenumber(case%nx,case%ny)
        associate(g => case%damping_rate(), w => case%frequency())
        associate(ct => exp(-g*time)*(cos(w*time) + g/w*sin(w*time)), &
                  st => case%delta*case%csqr/w*exp(-g*time)*sin(w*time))   ! |k| cancels against k/|k|
            do i = 1, size(x)
                associate(th => k(1)*x(i) + k(2)*y(i) + case%phase)
                    p(i) = case%rho0*(1 + case%delta*cos(th)*ct)
                    u = st*sin(th)
                end associate
                ux(i) = u*k(1)
                uy(i) = u*k(2)
            end do
        end associate
        end associate
    end subroutine

    !
    ! SHEAR LAYER
    !

    subroutine shear_layer_fields(case,x,y,p,ux,uy)
        class(shear_layer), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        associate(yd => y/case%box%Ly)
            ux = case%u0*tanh(case%k*merge(yd - 0.25_wp, 0.75_wp - yd, yd <= 0.5_wp))
        end associate
        uy = case%u0*case%delta*sin(2*pi*(x/case%box%Lx + 0.25_wp))
        p = 0.0_wp
    end subroutine

    !
    ! BAROTROPIC VORTEX
    !

    ! p holds the density
    subroutine barotropic_vortex_fields(case,x,y,p,ux,uy)
        class(barotropic_vortex), intent(in) :: case
        real(wp), intent(in) :: x(:), y(:)
        real(wp), intent(out) :: p(:), ux(:), uy(:)
        real(wp) :: r(2), g
        integer :: i
        do i = 1, size(x)
            r = case%box%minimum_image([x(i), y(i)] - case%center)
            g = exp(-(r(1)**2 + r(2)**2)/(2*case%Rc**2))
            ux(i) = case%U0 - case%eps*(r(2)/case%Rc)*g
            uy(i) =           case%eps*(r(1)/case%Rc)*g
            p(i) = case%rho0*exp(-case%eps**2/case%csqr*g*g/2)
        end do
    end subroutine

end module
