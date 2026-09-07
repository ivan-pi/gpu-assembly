# Periodic benchmarks

Analytic flow fields on a doubly periodic box, as C++ functors in
`examples/rbf_flow_benchmarks.h` (namespace `flow_benchmarks`) and as
Fortran types in `examples/rbf_benchmarks.f90` (module `rbf_benchmarks`),
with the same names and parameters on both sides. They verify a lattice
Boltzmann implementation without boundary conditions, so the dissipation
and dispersion of the bulk scheme can be studied on their own. Cases that
need boundary conditions (Kovasznay, Poiseuille, Couette, the cavity)
are not here.

| Case | Scalar | Exact solution of |
|---|---|---|
| [Shear modes](#shear-modes) | pressure | incompressible Navier-Stokes |
| [Acoustic wave](#acoustic-wave) | density | linear isothermal Navier-Stokes |
| [Shear layer](#shear-layer) | pressure (zero) | none, initial condition |
| [Barotropic vortex](#barotropic-vortex) | density | none, initial condition |

## Box and interface

Every case is built on the library's periodic box with sides `Lx` and
`Ly` -- `rbf::spatial::PeriodicBox<T, 2>` from [spatial.md](spatial.md),
in Fortran the module `rbf_periodic_box` in `src/` -- since the box also
serves the stencil search and assembly. `wavenumber(box, nx, ny)` turns
integer mode numbers into the wave numbers `(2 pi nx/Lx, 2 pi ny/Ly)`,
so every field is periodic by construction. In lattice units the box is
`nx` by `ny` cells and the fundamental mode is `(1, 1)`.

A case evaluates to a scalar and the two velocity components at a point.
The time-dependent cases take the time; the initial conditions do not:

```cpp
using namespace flow_benchmarks;
PeriodicBox<double, 2> box{{64.0, 64.0}};
auto tg = taylor_green(box, 1, 1, u0, nu);
auto [p, ux, uy] = tg({x, y}, time);
```

```fortran
use rbf_periodic_box, only: periodic_box
use rbf_benchmarks, only: shear_modes, taylor_green
type(periodic_box) :: box
type(shear_modes) :: tg
box = periodic_box(64.0_wp, 64.0_wp)
tg = taylor_green(box, 1, 1, u0, nu)
call tg%fields(x, y, time, p, ux, uy)
```

The scalar is the pressure, or the density for the acoustic wave and the
barotropic vortex, which are defined through it. For an athermal lattice
`p = cs^2 rho`, so a pressure field is initialized as
`rho = rho0 + p/cs^2` and a density field is used as is.

Constructors check their parameters: C++ with `assert` (compiled out by
`NDEBUG`), Fortran with `error stop` and the same messages. The Mach
number `u0/cs` is not checked and should stay small.

## Shear modes

A superposition of transverse plane waves on the box,

```
u(x, t) = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m
```

with velocity amplitudes `a_m`, mode numbers giving `k_m`, phases, and
`e_m = (-ky, kx)/|k|` perpendicular to `k_m` so that each mode is
divergence free. All modes must share the same `|k|^2`. Then the
vorticity is `|k|^2` times the stream function, the nonlinear term is a
gradient absorbed by the pressure, and the field is an exact solution of
the incompressible Navier-Stokes equations decaying with the time
constant `1/(nu |k|^2)`. The pressure

```
p = -(|u - U|^2 + (sum_m a_m cos(k_m.x + phi_m))^2)/2 + mean
```

decays at twice the rate and has zero mean, which assumes distinct wave
vectors (`k_m != +-k_n`, checked). `U` is a uniform drift, zero by
default: the pattern is carried along unchanged, which a Galilean
invariant scheme must reproduce, and the viscosity measured with a drift
exposes the velocity-dependent error of the standard collision.

Parameters: the box, the modes, `nu`, and optionally the drift. Methods:
`ksqr`, `time_constant`, `decay(time)`, `decay_time(frac)`,
`stress_tensor` for the strain rate `(grad u + grad u^T)/2` as
`{sxx, sxy, syy}`, and `body_force`.

- **`shear_wave(box, nx, ny, u0, nu, phase, drift)`** is one mode: the
  velocity is perpendicular to `k` and the pressure uniform, so the
  equations reduce to diffusion. This is the standard viscosity
  measurement; `(n, 0)` against `(n, n)` exposes anisotropy.
- **`taylor_green(box, nx, ny, u0, nu, drift)`** is the pair `(nx, ny)`,
  `(nx, -ny)` with the amplitudes that give

  ```
  ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y) exp(-t/td)
  uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y) exp(-t/td)
  p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y)) exp(-2 t/td)
  ```

  for the wave numbers of `|nx|, |ny|`. The nonlinear term is active and
  balanced by the pressure, so this also checks the pressure and the
  convective part of the scheme.
- Other superpositions work as long as `|k|^2` matches, for instance
  `(3, 4)` with `(5, 0)` on a square box: an asymmetric field that
  catches errors the symmetric cases cancel out.

**Forcing.** `body_force` returns `nu |k|^2 (u(x, 0) - U)`, followed
along the drift. Applied as a body force it makes the time-zero field a
steady solution. One mode along `y`, `shear_wave(box, 0, n, ...)`, is
then Kolmogorov flow, and the Taylor-Green pair the four-roll mill; both
verify the forcing scheme and the viscosity in steady state without a
fit. Kolmogorov flow is unstable above a Reynolds number of about
`sqrt(2)` based on the forcing, so keep it laminar.

**Measuring the viscosity.** Initialize with the field at `t = 0`, run,
and record the maximum velocity magnitude over the box at every step. It
follows `u0 exp(-t/tau)` with `tau = 1/(nu |k|^2)`; a nonlinear
least-squares fit of `A exp(-t/tau)` gives `tau`, hence
`nu = 1/(tau |k|^2)`, to be compared with the nominal
`cs^2 (tau_lbm - 1/2)`. Skip the first iterations, which carry an
oscillation from the initialization. On a scattered node set no node
needs to sit at the crest, so the kinetic energy or the projection onto
the mode is a more robust amplitude than the maximum.

**Initializing the populations.** With the equilibrium alone the
non-equilibrium part builds up during the first steps and produces an
initial layer. With the strain rate the populations can be initialized as

```
f_i = f_i^eq(rho, u) - w_i rho tau/cs^2 (c_i c_i - cs^2 I) : S
```

with `tau` the relaxation time in lattice units, `nu = cs^2 (tau - 1/2)`.

## Acoustic wave

A standing sound wave released from rest, in linear isothermal acoustics
with `p = cs^2 rho`:

```
rho = rho0 (1 + delta cos(k.x + phi) exp(-gamma t) (cos(W t) + gamma/W sin(W t)))
u   = delta cs^2 |k|/W exp(-gamma t) sin(W t) sin(k.x + phi) k/|k|
```

with the damping rate `gamma = (nu + nu_bulk) |k|^2/2` in two dimensions
and the frequency `W = sqrt(cs^2 |k|^2 - gamma^2)`. Parameters: the box,
the mode numbers, the relative density amplitude `delta`, `nu`, and
optionally `nu_bulk` (default `nu`), `csqr` (default 1/3), `rho0`
(default 1) and a phase. Methods: `ksqr`, `damping_rate`, `frequency`,
`period`.

The shear wave covers dissipation; this is the dispersion half. The
period gives `W` and the damping gives `gamma`, hence
`cs^2 = (W^2 + gamma^2)/|k|^2` and `nu + nu_bulk = 2 gamma/|k|^2`, so
with the shear viscosity known from a shear wave this measures the
sound speed and the bulk viscosity. The `nu + nu_bulk` is the
two-dimensional form of the isothermal sound absorption
`(2 nu (1 - 1/D) + nu_bulk)`. The BGK collision has the stress
`rho nu (grad u + grad u^T)` without the trace subtraction, which is a
bulk viscosity `2 nu/D` (Dellar, 2001): `nu` on D2Q9, the default here,
and `2 nu/3` in three dimensions; a multiple relaxation time collision
sets it independently. The solution is linear, valid for `delta << 1`;
the velocity amplitude is about `delta cs`.

## Shear layer

The doubly periodic shear layer of Minion and Brown (1997):

```
ux = u0 tanh(k (y/Ly - 1/4))    for y/Ly <= 1/2
ux = u0 tanh(k (3/4 - y/Ly))    for y/Ly >  1/2
uy = u0 delta sin(2 pi (x/Lx + 1/4))
```

Parameters: the box, `u0`, and `k = 80`, `delta = 0.05` by default (the
"thin" layer of the reference). Initial condition only, pressure zero.
It has no analytic solution and tests stability at low resolution:
under-resolved runs produce spurious secondary vortices or blow up.

## Barotropic vortex

The vortex initialization of Wissocq, Boussuge and Sagaut (2020), Phys.
Rev. E 101, 043306, Eqs. (3), (4) and (20): a Gaussian vortex of radius
`Rc` and strength `eps` convected at `U0`, with the density in balance
with the athermal equation of state,

```
ux  = U0 - eps (y - cy)/Rc exp(-r^2/(2 Rc^2))
uy  =      eps (x - cx)/Rc exp(-r^2/(2 Rc^2))
rho = rho0 exp(-eps^2/(2 cs^2) exp(-r^2/Rc^2))
```

where the displacement from the centre is taken through the periodic
images. Parameters: the box, `U0`, the centre, `Rc`, `eps`, and
`rho0 = 1`, `csqr = 1/3` by default. Initial condition only; the scalar
is the density. The vortex Mach number is `eps/cs`.

The barotropic density, as opposed to the isothermal one, puts the
initial state in balance with the lattice Boltzmann equation of state so
that no spurious acoustic wave is emitted at start up. Running the
vortex through the box for one or more turnover times and comparing with
the initial field checks the transport of a vortical structure and the
acoustic behaviour of the scheme.
