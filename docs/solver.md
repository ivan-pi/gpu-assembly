# Sparse products and iterative solvers

The pieces between an assembled operator and a solved system: the
product of a matrix in ELLPACK storage with a vector, in Fortran and
part of the host library, and a C interface over Eigen's iterative
solvers with its Fortran module, the optional component `rbf_solver`.
Together they are a small solver library, usable from Fortran without
a line of C++.

| Piece | Where | Needs |
|---|---|---|
| ELLPACK products | `src/rbf_ellpack.F90`, module `rbf_ellpack` | the host library |
| CSR product | `rbf_csr_mv_dp`, Fortran `csr_mv` | `rbf_solver` |
| CSR solvers, matrix-free solvers | `rbf_solve_sparse_csr_dp`, `rbf_solve_sparse_mf_dp`, Fortran `solve_sparse` | `rbf_solver` |

## Building the solver component

`rbf_solver` wraps [Eigen](https://eigen.tuxfamily.org) 3.4, a
dependency nothing else in the library has, so it is a target of its
own and off by default:

```
cmake -B build -DGPU_ASSEMBLY_ENABLE_SOLVER=ON
```

CMake finds Eigen through its own config file, as `libeigen3-dev`
(Debian, Ubuntu) and Eigen's install ship it; `Eigen3_ROOT` or
`CMAKE_PREFIX_PATH` point at a copy elsewhere. Eigen is private to the
target: the header `rbf_solver.h` is plain C, the module `rbf_solver`
only declares interfaces, and a consumer of the target needs Eigen
neither to compile nor to link. With OpenMP, Eigen threads its CSR
product over the rows.

The tests `solver` and `solver_fortran` build with the option and
exercise the interface from C++ and from Fortran; CI builds with it
on.

## ELLPACK products

A fixed-width stencil graph is ELLPACK storage already: every row has
`nnzrow` entries, so the column indices and the values are rectangular
arrays and there is no row pointer. Two layouts, differing in which
index runs fastest:

| Layout | Arrays | Contiguous | Padding |
|---|---|---|---|
| row | `a(lda, n)`, `ja(lda, n)`, `lda >= nnzrow` | the entries of a row | entries past `nnzrow` |
| col | `a(lda, nnzrow)`, `ja(lda, nnzrow)`, `lda >= n` | the j-th entries of all rows | rows past `n` |

The row layout is compressed sparse row with a fixed row length and
the row pointer implicit, `ia(i) = i*lda`. It is the layout the library
produces: the stencils `ja(k, n)` of `NodeSet::stencils` are its index
array and the weights the assembly kernels write beside them its
values, so a driver can multiply with what the assembly left without
converting. The col layout is ELLPACK proper, the transposed storage
that lets the SIMD lanes take consecutive rows, and is the one to
convert to where the product dominates the run time.

Both products are the BLAS-shaped update `y := beta*y + alpha*A*x`,
with the BLAS argument order (the dimensions, `alpha`, the matrix with
its leading dimension, `x`, `beta`, `y`) and the BLAS convention that
`y` is not read when `beta` is zero:

```fortran
use rbf_ellpack
call ellpack_mv_row(n, nnzrow, alpha, a, ja, lda, x, beta, y)
call ellpack_mv_col(n, nnzrow, alpha, a, ja, lda, x, beta, y)
```

The two have the same argument list, only the meaning of `lda`
differs. Indices are 0-based, as the stencils come; padding is never
read. `beta = 1` is the accumulating form the matrix-free solver's
callback needs.

### Threads, SIMD, and the block size

Threads take blocks of rows, an OpenMP `parallel do` with a static
schedule, and the SIMD lanes take what is contiguous in the layout: in
the row layout the entries of a row, a gather and a short reduction; in
the col layout the rows of a block, where the j-th entry of consecutive
rows sits in consecutive memory. The two directives are kept apart, one
loop each, since a combined `parallel do simd` would need
`schedule(simd:static)` to split the rows along the vector length, and
not every compiler takes that. The row reduction needs
`simd reduction(+:t)` to let the compiler reassociate the sum, which
flang cannot lower yet, so under flang that one loop runs scalar; the
guard is in the source and goes once flang can.

The col layout kernel sweeps the `nnzrow` entries of a block of rows
with the block's partial sums in a small stack array, and the block
size matters. Measured on a 21-point stencil over a million nodes of
a square grid, numbered along the grid, at `-O2 -march=x86-64-v3`, on
a 4-core Xeon with AVX-512; time per product in milliseconds, best of
20:

| rows per block | gfortran, 1 thread | gfortran, 4 threads | flang, 1 thread | flang, 4 threads |
|---|---|---|---|---|
| 8 | 25.1 | 4.8 | 24.0 | 5.3 |
| 16 | 22.0 - 26.6 | 3.6 - 4.5 | 25.7 | 3.5 |
| 32 | 16.2 - 20.3 | 3.2 - 3.3 | 22.5 | 3.3 |
| 64 | 18.7 - 19.4 | 3.5 - 3.6 | 21.7 | 3.6 |
| 128 | 23.8 - 27.9 | 3.7 - 4.3 | 28.9 | 4.3 |
| 256 | 30.3 - 31.8 | 4.4 - 4.6 | 29.6 | 4.2 |
| 1024 | 31.3 | 4.4 | 30.5 | 4.6 |
| row layout, for comparison | 24.5 - 31.9 | 4.4 - 5.6 | 28.5 - 31.0 | 4.4 - 4.9 |

Ranges are two runs. The optimum is 32 to 64 rows, about 1.5x faster
than the ends, and 32 is the default (`nblock` in the source). The
product is memory-bound, so on another machine the shape of the curve
matters more than the numbers: compiling with `-DRBF_ELLPACK_NBLOCK=`
overrides the default for measuring it again. At 4 threads the col
layout reaches about 80 GB/s counting `a`, `ja`, `x` and `y` once
each, and the row layout, with its scalar reduction per row, about
60 GB/s in either compiler.

## CSR product

```fortran
use rbf_solver
call csr_mv(nr, nc, nnz, val, ia, ja, alpha, x, beta, y)   ! y = beta y + alpha A x
```

```c
rbf_csr_mv_dp(nr, nc, nnz, val, ia, ja, alpha, x, beta, y);
```

Eigen's product of a CSR matrix, 0-based, with a vector, threaded over
the rows when built with OpenMP; `y` is not read when `beta` is zero.
For the residuals and time steps around a solve. `rbf::make_row_ptr`
in [rbf_reorder.h](renumbering.md) gives the row pointer of a
fixed-width graph, which is the bridge from the assembly's storage to
CSR.

## Solvers

```fortran
use rbf_solver
status = solve_sparse(n, nnz, val, ia, ja, b, x)
status = solve_sparse(n, nnz, val, ia, ja, b, x, res_error=err, res_iter=iter, &
                      method=RBF_SOLVER_CONJUGATE_GRADIENT, precond=RBF_PRECOND_INCOMPLETE, &
                      max_iter=500, tolerance=1.0e-10_c_double)
```

```c
#include "rbf_solver.h"
int status = rbf_solve_sparse_csr_dp(n, nnz, val, ia, ja, b, x,
                                     &err, &iter, &method, &precond, &max_iter, &tol);
```

`A x = b` with `A` in CSR, 0-based; `x` is the initial guess on input
and the solution on output. The trailing arguments are optional, in
Fortran by keyword and in C as `NULL`, and default to Eigen's own
choices:

| Argument | Values | Default |
|---|---|---|
| `method` | `RBF_SOLVER_BICGSTAB`, `RBF_SOLVER_CONJUGATE_GRADIENT` | BiCGSTAB |
| `precond` | `RBF_PRECOND_IDENTITY`, `RBF_PRECOND_DIAGONAL`, `RBF_PRECOND_INCOMPLETE` | diagonal (Jacobi) |
| `max_iter` | | `2 n` |
| `tolerance` | relative residual to stop at | machine epsilon |
| `res_error` | out: the relative residual reached | |
| `res_iter` | out: iterations taken | |

Conjugate gradient assumes a symmetric positive definite matrix;
BiCGSTAB takes any nonsingular one. `INCOMPLETE` is ILUT for BiCGSTAB
and incomplete Cholesky for conjugate gradient. Eigen counts completed
passes, so a preconditioner that happens to be exact reports zero
iterations, as does an initial guess that already solves the system.

The status is one of

| Status | Meaning |
|---|---|
| `RBF_SOLVER_SUCCESS` | converged |
| `RBF_SOLVER_NUMERICAL_ISSUE` | a breakdown in the iteration or the preconditioner |
| `RBF_SOLVER_NO_CONVERGENCE` | the iteration limit was reached; `x` holds the last iterate |
| `RBF_SOLVER_INVALID_INPUT` | a `NULL` array, a nonpositive size, `ia[n] != nnz`, an enumerator outside the enums |

The values are Eigen's `ComputationInfo`. The CSR arrays are wrapped
in place; the solver copies nothing but what its preconditioner keeps.

### Matrix-free

The same generic, and the same C function with `_mf_`, take an
operator given by its matrix-vector product instead of a matrix, for an
operator that is never formed, or that is applied on the device:

```fortran
subroutine matvec(nr, nc, alpha, x, y, data) bind(c)   ! y := y + alpha A x
    integer(c_int), value :: nr, nc
    real(c_double), value :: alpha
    real(c_double), intent(in) :: x(nc)
    real(c_double), intent(inout) :: y(nr)
    type(c_ptr), value :: data
    ...
end subroutine

status = solve_sparse(n, n, matvec, c_loc(ctx), b, x, res_error=err)
```

The callback is a `bind(c)` procedure of the interface `rbf_matvec_t`,
a BLAS level-2 update, and receives the context pointer unchanged;
`c_null_ptr` when it needs none. A module procedure, not an internal
one: passing an internal procedure to C needs a trampoline, hence an
executable stack. The matrix-free solvers run unpreconditioned, so
`precond` is not among their arguments. The test `solver_fortran`
solves a system this way with the ELLPACK product above as the
operator.
