# Periodic benchmarks

Analytic flow fields on a doubly periodic box, provided as C++ functors
in `examples/rbf_flow_benchmarks.h` (namespace `flow_benchmarks`) and as
Fortran derived types in `examples/rbf_benchmarks.f90` (module
`rbf_benchmarks`), with the same names and parameters on both sides.
Their main use is verifying a lattice Boltzmann implementation: on a
periodic box no boundary condition has to be implemented, so the
dissipation and dispersion behaviour of the bulk scheme can be studied on
its own, and any discrepancy with the analytic field is the scheme's own.

| Case | Scalar | Time dependent | Verifies |
|---|---|---|---|
| [Shear modes](#shear-modes) | pressure | yes | viscosity, pressure, nonlinear terms, Galilean invariance, forcing |
| [Acoustic wave](#acoustic-wave) | density | yes | sound speed, bulk viscosity |
| [Shear layer](#shear-layer) | pressure (zero) | no | stability, roll-up |
| [Barotropic vortex](#barotropic-vortex) | density | no | acoustics, vortex transport |

The shear modes case covers the shear wave, the Taylor-Green vortex,
Kolmogorov flow and the four-roll mill, which are all instances of one
construction.

## The box

Everything is built on a `periodic_box` with sides `Lx` and `Ly`. The
box turns integer mode numbers `(nx, ny)` into the wave numbers
`(2 pi nx/Lx, 2 pi ny/Ly)`, so a field built on it is periodic by
construction and no wave number is ever typed by hand. In lattice units
the box is `nx` by `ny` cells and the fundamental mode is `(1, 1)`. The
box also wraps a point into itself and gives the minimum-image
displacement between two points, which a periodic stencil search needs.

```cpp
using namespace flow_benchmarks;
periodic_box<double> box{64.0, 64.0};
auto tg = taylor_green(box, 1, 1, u0, nu);
```

```fortran
type(periodic_box) :: box
type(shear_modes) :: tg
box = periodic_box(64.0_wp, 64.0_wp)
tg = taylor_green(box, 1, 1, u0, nu)
```

## Common interface

Every case evaluates to a scalar and the two velocity components at a
point. In C++ a case is a functor, in Fortran a type with a `fields`
binding that takes arrays of points:

```cpp
auto [p, ux, uy] = tg({x, y});         // initial condition
auto [p, ux, uy] = tg({x, y}, time);   // time-dependent cases
```

```fortran
call tg%fields(x, y, p, ux, uy)        ! initial condition
call tg%fields(x, y, p, ux, uy, time)  ! time-dependent cases
```

The scalar is the pressure, except for the acoustic wave and the
barotropic vortex, which are defined through the density. For an
athermal lattice the two are related by `p = cs^2 rho`, so a pressure
field is initialized as `rho = rho0 + p/cs^2` and a density field is
used as is.

The two initial-condition cases take no time argument in C++, so
evaluating them at a later time does not compile. In Fortran they accept
the optional `time` and ignore it, so that all types conform to the
deferred `fields` binding of the abstract parent `benchmark_case` and a
driver can hold any case as `class(benchmark_case)`.

## Shear modes

A superposition of transverse plane waves on the box,

```
u(x, t) = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m
```

where each mode has a velocity amplitude `a_m`, mode numbers `(nx, ny)`
giving `k_m`, and a phase, and `e_m = (-ky, kx)/|k|` is perpendicular to
`k_m` so that each mode is divergence free. All modes must share the
same `|k|^2`. Then the vorticity is `|k|^2` times the stream function,
the nonlinear term is a gradient absorbed by the pressure, and the field
is an exact solution of the incompressible Navier-Stokes equations that
decays with the time constant `1/(nu |k|^2)`. The pressure

```
p = -(|u - U|^2 + (sum_m a_m cos(k_m.x + phi_m))^2)/2 + mean
```

decays at twice the rate and is normalized to zero mean, which assumes
distinct wave vectors (`k_m != +-k_n`, checked). `U` is a uniform drift,
zero by default: the pattern is carried along unchanged, which a
Galilean invariant scheme must reproduce, and the viscosity measured
with a drift exposes the velocity-dependent error of the standard
lattice Boltzmann collision.

Parameters: the box, the list of modes, `nu`, and optionally the drift.
Methods:

| Method | Meaning |
|---|---|
| `ksqr()` | the common `kx^2 + ky^2` |
| `time_constant()` | `1/(nu ksqr)`, the e-folding time of the velocity |
| `decay(time)` | `exp(-time/time_constant)` |
| `decay_time(frac)` | time at which the velocity has dropped to the fraction `frac` |
| `viscosity(a0, a1, t0, t1)` | viscosity from two measured amplitudes, a quick estimate |
| `stress_tensor` | strain rate `S = (grad u + grad u^T)/2` as `{sxx, sxy, syy}` |
| `body_force` | the force that holds the time-zero field steady |

Two factory functions build the classic instances:

- **`shear_wave(box, nx, ny, u0, nu, phase, drift)`**: one mode. The
  velocity is perpendicular to `k` and the pressure uniform, so the
  Navier-Stokes equations reduce to the diffusion equation. This is the
  standard viscosity measurement, see below. Choosing `(n, 0)` and
  `(n, n)` exposes any anisotropy of the discretization.
- **`taylor_green(box, nx, ny, u0, nu, drift)`**: the pair `(nx, ny)`
  and `(nx, -ny)`, with the amplitudes that reproduce the usual form

  ```
  ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y) exp(-t/td)
  uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y) exp(-t/td)
  p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y)) exp(-2 t/td)
  ```

  Here the nonlinear term is active and balanced by the pressure, so
  this also checks the pressure and the convective part of the scheme.

Other superpositions are allowed as long as `|k|^2` matches, for instance
`(3, 4)` together with `(5, 0)` on a square box: an asymmetric field that
catches errors the symmetric cases cancel out.

**Forcing.** `body_force` returns `nu |k|^2 (u(x, 0) - U)`, followed
along the drift. Applied as a body force, the time-zero field is a
steady solution instead of a decaying one, with the time-zero pressure.
One mode along `y`, `shear_wave(box, 0, n, ...)`, is then Kolmogorov
flow, the unidirectional sinusoidal shear; the Taylor-Green pair is the
four-roll mill. Both verify the forcing scheme and the viscosity in
steady state without any fit. Kolmogorov flow is unstable above a
Reynolds number of about `sqrt(2)` based on the forcing, so keep it
laminar for verification.

**Measuring the viscosity.** Initialize with the field at `t = 0`, run,
and record the maximum of the velocity magnitude over the box at every
step. The maximum follows `u0 exp(-t/tau)` with `tau = 1/(nu |k|^2)`, so
a nonlinear least-squares fit of `A exp(-t/tau)` to the recorded curve
gives `tau` and from it `nu = 1/(tau |k|^2)`, to be compared with the
nominal `cs^2 (tau_lbm - 1/2)`. The first iterations usually carry an
oscillation from the initialization and must be skipped before fitting.
`viscosity(a0, a1, t0, t1)` is the quick two-point version of the same
estimate, fine for a sanity check but sensitive to the choice of the two
instants. On a scattered node set no node needs to sit at the crest, so
the kinetic energy or the projection onto the mode is a more robust
amplitude than the maximum. Keep `u0` small compared with `cs` so that
compressibility does not pollute the measurement.

**Initializing the populations.** The strain rate is what a consistent
initialization needs. Initializing with the equilibrium alone leaves the
non-equilibrium part to build up during the first steps and produces an
initial layer; with the strain rate the populations can be initialized
as

```
f_i = f_i^eq(rho, u) - w_i rho tau/cs^2 (c_i c_i - cs^2 I) : S
```

with `tau` the relaxation time in lattice units, `nu = cs^2 (tau - 1/2)`.

The grid-based type `taylor_green_t` in `examples/rbf_taylor_green.f90`
samples the Taylor-Green case at the cell centres `(i - 1/2, j - 1/2)` of
an `nx` by `ny` lattice, with arrays laid out as `(y, x)`, and optionally
returns the strain rate. It holds a `shear_modes` built with
`taylor_green` on the box `nx` by `ny` and delegates all formulas to it.

## Acoustic wave

A standing sound wave released from rest, in linear isothermal acoustics
with the equation of state `p = cs^2 rho`:

```
rho = rho0 (1 + delta cos(k.x + phi) exp(-gamma t) (cos(W t) + gamma/W sin(W t)))
u   = delta cs^2 |k|/W exp(-gamma t) sin(W t) sin(k.x + phi) k/|k|
```

with the damping rate `gamma = (nu + nu_bulk) |k|^2/2` in two dimensions
and the frequency `W = sqrt(cs^2 |k|^2 - gamma^2)`. Parameters: the box,
the mode numbers, the relative density amplitude `delta`, `nu`, and
optionally `nu_bulk` (default `nu`), `csqr` (default 1/3), `rho0`
(default 1) and a phase. Methods: `ksqr`, `sound_speed`, `damping_rate`,
`frequency`, `period`, and `longitudinal_viscosity(gamma)`, which
returns `nu + nu_bulk` from a measured damping rate.

The shear wave only covers dissipation; this is the dispersion half.
The period of the density oscillation gives the sound speed, and its
damping the sum of shear and bulk viscosity, so with the shear viscosity
known from a shear wave it measures the bulk viscosity. The BGK
collision on D2Q9 has `nu_bulk = nu`, which is the default; a multiple
relaxation time collision sets it independently. The solution is linear,
valid for `delta << 1`; the velocity amplitude is about `delta cs`.

## Shear layer

The doubly periodic shear layer of Minion and Brown (1997): two `tanh`
layers of thickness about `1/k` at `y/Ly = 1/4` and `3/4`, perturbed by
a sinusoidal `uy` of amplitude `delta u0` that rolls them up into
vortices.

```
ux = u0 tanh(k (y/Ly - 1/4))    for y/Ly <= 1/2
ux = u0 tanh(k (3/4 - y/Ly))    for y/Ly >  1/2
uy = u0 delta sin(2 pi (x/Lx + 1/4))
```

Parameters: the box, `u0`, and `k = 80`, `delta = 0.05` by default (the
"thin" layer of the reference). Initial condition only, pressure
returned as zero. It has no analytic solution and is used to test the
stability of a scheme at low resolution: under-resolved runs produce
spurious secondary vortices or blow up, and the roll-up can be compared
with a converged reference.

## Barotropic vortex

The isentropic vortex initialization of Wissocq, Boussuge and Sagaut
(2020), Phys. Rev. E 101, 043306, Eqs. (3), (4) and (20): a Gaussian
vortex of radius `Rc` and strength `eps` convected at `U0`, with a
density field consistent with the athermal lattice Boltzmann equation of
state,

```
ux  = U0 - eps (y - cy)/Rc exp(-r^2/(2 Rc^2))
uy  =      eps (x - cx)/Rc exp(-r^2/(2 Rc^2))
rho = rho0 exp(-eps^2/(2 cs^2) exp(-r^2/Rc^2))
```

Parameters: `U0`, the centre `(cx, cy)`, `Rc`, `eps`, and `rho0 = 1`,
`csqr = 1/3` (D2Q9 in lattice units) by default. Initial condition only;
the scalar returned is the density, not the pressure. The vortex Mach
number is `eps/cs`.

The point of the barotropic density, as opposed to the usual isothermal
one, is that the initial state is then in balance with the lattice
Boltzmann equation of state and emits no spurious acoustic wave at start
up. Running the vortex through the periodic box for one or more turnover
times and comparing with the initial field checks the transport of a
vortical structure and the acoustic behaviour of the scheme.

## Parameter checks

The C++ constructors assert the constraints their formulas rely on, in
the style of the rest of the library (`assert` with a message, compiled
out with `NDEBUG`). The Fortran constructor functions stop with the same
messages; the two initial-condition types use the default structure
constructor and carry no checks.

| Case | Constraint |
|---|---|
| `periodic_box` | `Lx > 0`, `Ly > 0` |
| `shear_modes` | `nu > 0`, at least one mode, no mode `(0, 0)`, wave vectors distinct up to sign, equal `|k|^2` |
| `taylor_green` | `nx ny > 0` (both nonzero, same sign) |
| `acoustic_wave` | mode not `(0, 0)`, `delta > 0`, `nu > 0`, `csqr > 0`, `rho0 > 0`, underdamped |
| `shear_layer` | `k > 0` (C++ only) |
| `barotropic_vortex` | `Rc > 0`, `rho0 > 0`, `csqr > 0` (C++ only) |

The time-dependent cases also assert a non-negative time in every
evaluation, since a benchmark starts from the initial field and a
negative time would grow the amplitude; `decay_time` asserts a fraction
in `(0, 1]` and `viscosity` two amplitudes of the same sign at two
distinct instants. Not checked is the Mach number `u0/cs`, which should
stay small.

## Not periodic

Kovasznay flow, plane Poiseuille and Couette flow, and the lid-driven
cavity also live on a rectangle generated on the fly, but they need
boundary conditions and so belong to a separate set.
