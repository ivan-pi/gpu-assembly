# RBF-FD weights on the host

The Fortran module `rbf_fd` (`src/rbf_fd.f90`) computes RBF-FD weights
stencil by stencil, with LAPACK or with the small-matrix kernels of
`third_party/reclu`. It is the host counterpart of the CUDA header
`rbf_operators.h`: the same basis, the same matrix, the same monomial
order, so the weights of the two can be compared directly, and it
serves the tests and the cases too small for the GPU.

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
column rank; the constructor and the driver stop otherwise.

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

all of which vanish at r = 0 for q >= 3. The second derivatives divide
by r^2; at r = 0 the divisor is clamped to the smallest normal number
and the zero numerator makes the quotient zero, without a branch. On
the monomials the operators are tables built once by the constructor,
coefficient and exponents per monomial, so their evaluation is
branch-free too; the monomials themselves come from the recurrence
x^i y^j = x * x^(i-1) y^j (or y times the previous power of y), which
also fills the polynomial blocks of the matrix without transposing.

## Solvers

The saddle-point matrix is symmetric indefinite. Four factorizations
are offered, selected by the `solver` argument of the constructor:

| Code | Factorization | From |
|---|---|---|
| `SOLVER_LU` | LU with partial pivoting, `dgetf2` + `dgetrs` | LAPACK |
| `SOLVER_LDLT` | Bunch-Kaufman LDL^T, `dsytrf` + `dsytrs` | LAPACK |
| `SOLVER_RECLU_LU` | the same LU | reclu |
| `SOLVER_RECLU_LDLT` | the same LDL^T, the default | reclu |

LDL^T does half the flops of LU and needs only the lower triangle of
the matrix filled. The LU goes through the unblocked `dgetf2` because
`dgesv` reaches for the recursive `dgetrf2`, which at these sizes is
almost twice as slow. reclu (`third_party/reclu/README.md`,
`PERFORMANCE.md`) is a set of kernels written for exactly these
systems: register-tiled updates, the pivot search fused into the
update, right-hand sides in vector lanes. Its fast paths want the
matrix 32-byte aligned with a leading dimension that is a multiple of
4, which the workspace provides for every solver: the leading dimension
is the largest system padded to a multiple of 8 doubles, and the matrix
sits in a buffer with slack at an offset recomputed on every fill from
the buffer's address (a workspace copy lands elsewhere). Without AVX2
and FMA in the target flags reclu compiles to its plain-loop
references, which are still faster than the netlib LAPACK. All four
agree to rounding.

Which LAPACK matters less than which factorization, but the assembly
loop threads over stencils, so the library must be a sequential one:
with MKL, configure with `-DBLA_VENDOR=Intel10_64lp_seq`; once MKL is
installed it is also what a plain `find_package(LAPACK)` picks, and
`-DBLA_VENDOR=Generic` gets the reference library back.

## Use

The workspace holds the parameters and the work arrays of one solve,
sized once for stencils of up to `nmax` nodes and `nrhs_max` right-hand
sides:

```fortran
use rbf_fd
type(rbf_fd_workspace) :: ws
ws = rbf_fd_workspace(q=3, p=3, nmax=21, nrhs_max=4)                  ! reclu LDL^T
ws = rbf_fd_workspace(q=3, p=3, nmax=21, nrhs_max=4, solver=SOLVER_LU)
```

One stencil, with `x(1:n)`, `y(1:n)` stencil-local:

```fortran
op = [OP_DX, OP_DY, OP_LAPLACE, OP_VALUE]
xc = [0.0_wp, 0.0_wp, 0.0_wp, 0.3_wp]      ! derivatives at the node,
yc = [0.0_wp, 0.0_wp, 0.0_wp, -0.2_wp]     ! interpolation at an offset
call rbf_fd_weights(ws, n, x, y, 4, op, xc, yc, info)
```

leaves the weights in `ws%B(1:n, 1:4)`, of leading dimension `ws%lda`.
`info` is LAPACK's, positive when the factorization met an exact zero
pivot (coincident nodes, or fewer than npoly(p) of them); rounding may
turn such a pivot into a tiny nonzero one, so a singular stencil is not
guaranteed to be reported. `rbf_fd_weights` is `rbf_fd_fill` followed
by `rbf_fd_solve`, both public, which is how the benchmark times the
phases apart. The per-stencil routines check nothing; the driver does.

All stencils, through two callbacks the caller supplies:

```fortran
call rbf_fd_assemble(ws, nstencils, nrhs, op, xc, yc, gather, scatter, info)
```

`gather(s, nmax, n, x, y)` delivers the stencil-local nodes of stencil
`s`, `scatter(s, n, nrhs, w, ldw)` receives its weights, the workspace's
`B` itself with its leading dimension, and stores them in whatever
sparse layout the caller keeps: `w(k, j)` multiplies the value at the
k-th node `gather` delivered, for the j-th operator. Internal procedures
serve as the callbacks and see the caller's arrays by host association;
`test/test_rbf_fd.f90` gathers with the minimum image of a periodic box
and scatters into fixed-row-length CSR values, `examples/rbf_fd_bench.f90`
does the same for rows of any length. The loop is OpenMP-parallel over
stencils, each thread with its own copy of the workspace, so `scatter`
must be safe to call from several threads at once, which it is when
stencil `s` writes only its own row. The callbacks, like the module's
own per-stencil procedures, should be declared `recursive`: the
standard allows concurrent invocation only of recursive procedures, and
gfortran's `-fcheck=all` enforces it. `info` is 0, or the lowest index
of a stencil whose system was singular; `scatter` is not called for
those.

`rbf_fd_basis(ws, n, x, y, op, xc, yc, b)` evaluates the right-hand side
itself, `b(1:n)` the PHS part and `b(n+1:n+np)` the monomials, for
anyone who assembles differently.

## Building

`find_package(LAPACK)` in the top-level `CMakeLists.txt` looks for
LAPACK; the module is the target `rbf_fd`, kept apart from the host
library `rbf` so that a build without LAPACK still gets everything
else. Without LAPACK the module, its test and the benchmark are
skipped with a message at configure time. On Ubuntu the package is
`liblapack-dev`, which CI installs; MKL is `libmkl-dev`.

The build is `Release` unless a build type is given.
`-DGPU_ASSEMBLY_MARCH_NATIVE=ON` adds `-march=native`, which is what
turns on reclu's AVX2 kernels and the vector width of the fill loops;
the gcc job of CI builds with it, the flang jobs without, so both
paths of reclu are tested.

The test, `ctest -R rbf_fd`, checks the basis derivatives against
central differences for q = 3, 5, 7, the exactness of one stencil's
weights on every monomial of degree at most p, the unit vector of
interpolation at a node, the agreement of the four solvers, the
storage contract of the kernels, and the accuracy of the assembled
operators of `data/poisson_32_21` on a smooth periodic field. reclu's
own tests run as `ctest -R reclu`.

## Timings

`examples/rbf_fd_bench.f90` assembles the operators of a case with each
solver through the callbacks and reports the rate, then times the
phases of one stencil on one thread:

```
rbf_fd_bench <case> <Lx> <Ly> [p] [q] [nrhs] [reps]
```

A case of 90000 nodes on a periodic 300 x 300 box, jittered grid,
21-node stencils (`data/gen/perturbed_grid.py -n 300 --knn 21`), p = 3,
q = 3, so 31 x 31 systems; gfortran 13 with `-O3 -march=native`, the
reference LAPACK 3.12, four threads of a Xeon at 2.8 GHz with AVX-512;
one right-hand side, the Laplacian:

| Solver | µs per stencil, one thread (gather / fill / solve) | stencils/s, 4 threads |
|---|---|---|
| LU, LAPACK | 7.7 (0.2 / 0.9 / 6.6) | 497 k |
| LDL^T, LAPACK | 7.1 (0.2 / 0.6 / 6.3) | 549 k |
| LU, reclu | 4.6 (0.2 / 0.9 / 3.5) | 852 k |
| LDL^T, reclu | 4.3 (0.2 / 0.6 / 3.5) | 900 k |

With seven right-hand sides (all six derivatives and an interpolation)
the reclu LDL^T stencil takes 6.3 µs, 608 k stencils/s. MKL's
sequential LAPACK is within ten percent of the reference one either
way: at 31 x 31 the time is not in the flops but in the per-step
pivoting chain and the stores, which is what reclu attacks. The solve
is now four fifths of a stencil; the fill went from 1.9 to 0.6 µs by
evaluating one loop nest per PHS exponent instead of a call per column,
and by filling only the lower triangle for the LDL^T solvers.
