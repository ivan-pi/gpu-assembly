# RBF-FD weights on the host

The Fortran module `rbf_fd` (`src/rbf_fd.f90`) computes RBF-FD weights
stencil by stencil with LAPACK. It is the host counterpart of the CUDA
header `rbf_operators.h`: the same basis, the same matrix, the same
monomial order, so the weights of the two can be compared directly, and
it serves the tests and the cases too small for the GPU.

## The approximation

Polyharmonic splines phi(r) = r^q, q odd, augmented with the 2-D
monomials of degree at most p, of which there are npoly(p) =
(p+1)(p+2)/2. For a stencil of n nodes the weights w of an operator L
solve the saddle-point system

```
[ A    P ] [ w      ]   [ L phi ]     A(k,l) = phi(|x_l - x_k|)
[ P^T  0 ] [ lambda ] = [ L m   ]     P(k,i) = m_i(x_k)
```

with the right-hand side the operator applied to the basis at the
evaluation point. Several operators are several right-hand sides of
the same factorization. The trailing polynomial coefficients lambda are
discarded. A stencil needs at least npoly(p) nodes for P to have full
column rank; the constructor and `rbf_fd_weights` stop otherwise.

Coordinates are stencil-local: displacements from the stencil's own
node, with periodic images already resolved (`periodic_box%minimum_image`
of `rbf_periodic_box`). Derivatives at the node are therefore taken at
(0, 0), and an interpolation weight at any point of the stencil's
footprint.

The operators, as the codes the right-hand sides carry:

| Code | L |
|---|---|
| `OP_VALUE` | the value, that is interpolation |
| `OP_DX`, `OP_DY` | first derivatives |
| `OP_DXX`, `OP_DXY`, `OP_DYY` | second derivatives |
| `OP_LAPLACE` | the Laplacian |

For phi = r^q these are, with d = (xc, yc) - x_k,

```
phi_x   = q r^(q-2) dx
phi_xx  = q r^(q-4) ((q-1) dx^2 + dy^2)
phi_xy  = q (q-2) r^(q-4) dx dy
lap phi = q^2 r^(q-2)
```

all of which vanish at r = 0 for q >= 3.

## Solvers

The saddle-point matrix is symmetric indefinite. `SOLVER_LU` (the
default) factors it with partial pivoting, `dgesv`; `SOLVER_LDLT` with
the Bunch-Kaufman LDL^T of `dsytrf`/`dsytrs`, which does half the
flops. The two agree to rounding.

## Use

The workspace holds the parameters and the work arrays of one solve,
sized once for stencils of up to `nmax` nodes and `nrhs_max` right-hand
sides:

```fortran
use rbf_fd
type(rbf_fd_workspace) :: ws
ws = rbf_fd_workspace(q=3, p=3, nmax=21, nrhs_max=4)
```

One stencil, with `x(1:n)`, `y(1:n)` stencil-local:

```fortran
op = [OP_DX, OP_DY, OP_LAPLACE, OP_VALUE]
xc = [0.0_wp, 0.0_wp, 0.0_wp, 0.3_wp]      ! derivatives at the node,
yc = [0.0_wp, 0.0_wp, 0.0_wp, -0.2_wp]     ! interpolation at an offset
call rbf_fd_weights(ws, n, x, y, 4, op, xc, yc, w, info)
```

fills `w(1:n, 1:4)`; `info` is LAPACK's, positive when the system is
singular (coincident nodes, or fewer than npoly(p) of them).

All stencils, through two callbacks the caller supplies:

```fortran
call rbf_fd_assemble(ws, nstencils, nrhs, op, xc, yc, gather, scatter, info)
```

`gather(s, nmax, n, x, y)` delivers the stencil-local nodes of stencil
`s`, `scatter(s, n, nrhs, w)` receives its weights and stores them in
whatever sparse layout the caller keeps: `w(k, j)` multiplies the value
at the k-th node `gather` delivered, for the j-th operator. Internal
procedures serve as the callbacks and see the caller's arrays by host
association; `test/test_rbf_fd.f90` gathers with the minimum image of
a periodic box and scatters into fixed-row-length CSR values. The loop
is OpenMP-parallel over stencils, each thread with its own copy of the
workspace, so `scatter` must be safe to call from several threads at
once, which it is when stencil `s` writes only its own row. `info` is 0,
or the lowest index of a stencil whose system was singular; `scatter` is
not called for those.

`rbf_fd_basis(ws, n, x, y, op, xc, yc, b)` evaluates the right-hand side
itself, `b(1:n)` the PHS part and `b(n+1:n+np)` the monomials, for
anyone who assembles differently.

## Building

`find_package(LAPACK)` in the top-level `CMakeLists.txt` looks for
LAPACK; the module is the target `rbf_fd`, kept apart from the host
library `rbf` so that a build without LAPACK still gets everything
else. Without LAPACK the module and its test are skipped with a
message at configure time. On Ubuntu the package is `liblapack-dev`,
which CI installs.

The test, `ctest -R rbf_fd`, checks the basis derivatives against
central differences for q = 3, 5, 7, the exactness of one stencil's
weights on every monomial of degree at most p, the unit vector of
interpolation at a node, the agreement of the two factorizations, and
the accuracy of the assembled operators of `data/poisson_32_21` on a
smooth periodic field.
