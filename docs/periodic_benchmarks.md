# Periodic benchmarks

Analytic flow fields on a doubly periodic box, provided as C++ functors
in `examples/rbf_flow_benchmarks.h` (namespace `flow_benchmarks`) and as
Fortran derived types in `examples/rbf_benchmarks.f90` (module
`rbf_benchmarks`), with the same names and parameters on both sides.
Their main use is verifying a lattice Boltzmann implementation: a periodic
box needs no boundary treatment and no external geometry, so the point
cloud can be generated on the fly and any discrepancy with the analytic
field is the scheme's own. Benchmarks that need a genuine point cloud
(cavities, channels, cylinders, plates) are deliberately not included.

| Case | Scalar | Time dependent | Strain rate | Verifies |
|---|---|---|---|---|
| [Shear wave](#shear-wave) | pressure (zero) | yes | yes | viscosity, diffusion |
| [Taylor-Green vortex](#taylor-green-vortex) | pressure | yes | yes | viscosity, pressure, nonlinear terms |
| [Shear layer](#shear-layer) | pressure (zero) | no | no | stability, roll-up |
| [Barotropic vortex](#barotropic-vortex) | density | no | no | acoustics, vortex transport |

## Common interface

Every case evaluates to a scalar and the two velocity components at a
point. In C++ a case is a functor, in Fortran a type with a `fields`
binding that takes arrays of points:

```cpp
flow_benchmarks::taylor_green<double> tg{kx, ky, nu, u0};
auto [p, ux, uy] = tg({x, y});         // initial condition
auto [p, ux, uy] = tg({x, y}, time);   // decaying cases only
```

```fortran
type(taylor_green) :: tg
tg = taylor_green(kx=kx, ky=ky, nu=nu, u0=u0)
call tg%fields(x, y, p, ux, uy)        ! initial condition
call tg%fields(x, y, p, ux, uy, time)  ! decaying cases only
```

The scalar is the pressure, except for the barotropic vortex where it is
the density, because that case is defined barotropically. For an athermal
lattice the two are related by `p = cs^2 rho`, so a pressure field is
initialized as `rho = rho0 + p/cs^2` and a density field is used as is.

The two decaying cases are exact solutions of the incompressible
Navier-Stokes equations and share a second set of methods:

| Method | Meaning |
|---|---|
| `ksqr()` | `kx^2 + ky^2` |
| `time_constant()` | `1/(nu ksqr)`, the e-folding time of the velocity |
| `decay_time(frac)` | time at which the amplitude has dropped to `frac u0` |
| `amplitude(time)` | `u0 exp(-time/time_constant)` |
| `viscosity(a0, a1, t0, t1)` | viscosity recovered from two measured amplitudes |
| `stress_tensor` | strain rate `S = (grad u + grad u^T)/2` as `{sxx, sxy, syy}` |

The strain rate is what a consistent initialization of the populations
needs. Initializing with the equilibrium alone leaves the non-equilibrium
part to build up during the first steps and produces an initial layer;
with the strain rate the populations can be initialized as

```
f_i = f_i^eq(rho, u) - w_i rho tau/cs^2 (c_i c_i - cs^2 I) : S
```

with `tau` the relaxation time in lattice units, `nu = cs^2 (tau - 1/2)`.

The other two cases are initial conditions only. In C++ they take no time
argument, so evaluating them at a later time does not compile. In Fortran
they accept the optional `time` and ignore it, so that all four types
conform to the deferred `fields` binding of the abstract parent
`benchmark_case` and a driver can hold any case as
`class(benchmark_case)`.

All cases are periodic on a box of side `L` only if the wave numbers are
integer multiples of `2 pi/L`. In lattice units on an `nx` by `ny` grid
that is `kx = 2 pi/nx` and `ky = 2 pi/ny` for the fundamental mode.

## Shear wave

A transverse plane wave along the wave vector `k = (kx, ky)`:

```
u(x, t) = u0 sin(k.x + phase) exp(-nu |k|^2 t) e_perp,   e_perp = (-ky, kx)/|k|
```

The velocity is perpendicular to `k`, so the field is divergence free and
the pressure stays uniform; the Navier-Stokes equations reduce to the
diffusion equation and the amplitude decays exponentially. Parameters:
`kx`, `ky`, `nu`, `u0` and an optional `phase`.

This is the standard way to measure the effective viscosity of a scheme.
Initialize with the field at `t = 0`, run, and at two instants project the
velocity onto the mode to get its amplitude, for instance as

```
a(t) = 2/N sum_j u(x_j, t) . e_perp sin(k.x_j + phase)
```

over the `N` points of the box. Then `viscosity(a0, a1, t0, t1)` returns
`-ln(a1/a0) / (|k|^2 (t1 - t0))`, to be compared with the nominal
`cs^2 (tau - 1/2)`. Choosing `k` along a lattice axis and along a
diagonal exposes any anisotropy of the discretization. Keep `u0` small
compared with `cs` so that compressibility does not pollute the
measurement.

## Taylor-Green vortex

A doubly periodic array of counter-rotating vortices:

```
ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y) exp(-t/td)
uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y) exp(-t/td)
p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y)) exp(-2 t/td)
```

with `td = 1/(nu (kx^2 + ky^2))`. The amplitudes are chosen so the field
is divergence free for any `kx`, `ky`; with `kx = ky` the off-diagonal
strain rate vanishes. Parameters: `kx`, `ky`, `nu`, `u0`.

Unlike the shear wave the nonlinear term is active here and is balanced
by the pressure, so this case also checks the pressure and the convective
part of the scheme. The velocity decays with `td` and the pressure with
`td/2`; the same `viscosity` measurement as for the shear wave applies to
the velocity amplitude.

The grid-based type `taylor_green_t` in `examples/rbf_taylor_green.f90`
samples this case at the cell centres `(i - 1/2, j - 1/2)` of an `nx` by
`ny` lattice, with arrays laid out as `(y, x)`, and optionally returns the
strain rate. It holds a point-based `taylor_green` and delegates all
formulas to it.

## Shear layer

The doubly periodic shear layer of Minion and Brown (1997): two `tanh`
layers of thickness about `1/k` at `y/L = 1/4` and `3/4`, perturbed by a
sinusoidal `uy` of amplitude `delta u0` that rolls them up into vortices.

```
ux = u0 tanh(k (y/L - 1/4))    for y/L <= 1/2
ux = u0 tanh(k (3/4 - y/L))    for y/L >  1/2
uy = u0 delta sin(2 pi (x/L + 1/4))
```

Parameters: `u0`, `L`, and `k = 80`, `delta = 0.05` by default (the "thin"
layer of the reference). Initial condition only, pressure returned as
zero. It has no analytic solution and is used to test the stability of a
scheme at low resolution: under-resolved runs produce spurious secondary
vortices or blow up, and the roll-up can be compared with a converged
reference.

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
