# Sparse products and iterative solvers

The pieces between an assembled operator and a solved system: the
products of a matrix in fixed-row-length CSR, ELLPACK or sliced
ELLPACK storage with a vector, in Fortran and part of the host library, and a C
interface over Eigen's iterative solvers with its Fortran module, the
optional component `rbf_solver`.
Together they are a small solver library, usable from Fortran without
a line of C++.

| Piece | Where | Needs |
|---|---|---|
| fixed-row-length CSR product | `src/rbf_csr.F90`, module `rbf_csr` | the host library |
| ELLPACK product | `src/rbf_ellpack.F90`, module `rbf_ellpack` | the host library |
| sliced ELLPACK product | `src/rbf_sell.f90`, module `rbf_sell` | the host library |
| general CSR product | `rbf_csr_mv_dp`, Fortran `csr_mv` | `rbf_solver` |
| CSR solvers, matrix-free solvers | `rbf_solve_csr_dp`, `rbf_solve_mf_dp`, Fortran `solve_sparse` | `rbf_solver` |

Names: the C functions carry the library's `rbf_` prefix because they
are global symbols for the linker, as the other `bind(c)` entry points
of the library are. The enumerators are scoped by their category,
`SOLVER_` and `PRECOND_`, in both languages, and Fortran reaches the
functions through the generics, so a Fortran caller never types the
prefix.

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

## The fixed-row-length CSR product

The pattern of a k-nearest-neighbour graph over a point cloud: every
row holds the same number of entries, `nnzrow`, so the row pointer is
implicit and the values and column indices are the rectangular arrays
`a(lda, n)` and `ja(lda, n)`, `lda >= nnzrow`, with `a(j, i)` the j-th
entry of row `i`, multiplying `x(ja(j, i))`. The entries of a row are
contiguous; the entries past `nnzrow` are padding and never read.
Indices are 0-based. The stencils `ja(k, n)` of `NodeSet::stencils`
are this index array with `lda = k`, and the weights the assembly
kernels write beside them are its values, so a driver multiplies with
what the assembly left:

```fortran
use rbf_csr
call csr_mv(n, nnzrow, alpha, a, ja, lda, x, beta, y)   ! y = beta y + alpha A x
```

The product is the BLAS-shaped update `y := beta*y + alpha*A*x` in the
BLAS argument order (the dimensions, `alpha`, the matrix with its
leading dimension, `x`, `beta`, `y`), with the BLAS convention that `y`
is not read when `beta` is zero. It is the argument list of
`ellpack_mv` below, on the transposed storage; only the meaning of
`lda` differs.

Threads take rows, an OpenMP `parallel do` with a static schedule, and
the SIMD lanes take the entries of a row, a gather and a short
reduction. The reduction needs `simd reduction(+:t)` to let the
compiler reassociate the sum, which flang cannot lower yet, so under
flang that loop runs scalar; the guard is in the source and goes once
flang can. gfortran vectorizes it.

The general CSR product with a row pointer is `csr_mv` in `rbf_solver`
(the [CSR product](#csr-product) below, through Eigen). The two share
the name on purpose: a scope that uses both modules sees one generic
`csr_mv`, as Fortran merges generics of the same name, and the
arguments tell them apart, a rectangular `a` here against `nnz` and a
row pointer there. The test `solver_fortran` calls both through the
one name.

## The ELLPACK product

ELLPACK storage: every row holds the same number of entries, `nnzrow`,
and the values and column indices are the rectangular arrays
`a(lda, nnzrow)` and `ja(lda, nnzrow)`, `lda >= n`, with `a(i, j)` the
j-th entry of row `i`, multiplying `x(ja(i, j))`. The j-th entries of
consecutive rows are consecutive in memory, which is what lets the
SIMD lanes take consecutive rows; the rows past `n` are padding and
never read. Indices are 0-based, as the stencils come.

```fortran
use rbf_ellpack
call ellpack_mv(n, nnzrow, alpha, a, ja, lda, x, beta, y)   ! y = beta y + alpha A x
```

The product is the BLAS-shaped update `y := beta*y + alpha*A*x` in the
BLAS argument order (the dimensions, `alpha`, the matrix with its
leading dimension, `x`, `beta`, `y`), with the BLAS convention that `y`
is not read when `beta` is zero. `beta = 1` is the accumulating form
the matrix-free solver's callback needs; the test `solver_fortran`
solves a system that way with `ellpack_mv` as the operator.

A fixed-width stencil graph and the weights assembled beside it,
`ja(k, n)`, are this storage transposed, the
[fixed-row-length CSR](#the-fixed-row-length-csr-product) above.
Whether the transpose is worth making is measured below: on kNN
matrices it is not, for the product alone.

### Threads, SIMD, and the block size

Threads take blocks of rows, an OpenMP `parallel do` over the blocks
with a static schedule, and within a block the SIMD lanes take the
rows. The two directives sit on separate loops, since a combined
`parallel do simd` would need `schedule(simd:static)` to split the
rows along the vector length, and not every compiler takes that. Both
gfortran and flang vectorize the block loops.

The block's partial sums stay in a stack array while its `nnzrow`
entries are swept, so the sweep is `2*nnzrow` contiguous runs through
`a` and `ja`, one block long each, and one pass over `y`. The block
size matters, and its optimum is not stable across machines. On one
4-core Xeon (2.1 GHz, AVX-512) 32 rows ran about 1.5x faster than
1024 on the grid stencil below; on another (2.8 GHz, AVX-512, 4 MiB
of L2 per core, 33 MiB of L3) 1024 rows ran 3x faster than 32, on
every matrix. Short runs leave the `2*nnzrow` streams to the hardware
prefetcher, which may or may not keep up with that many; long runs
make every stream sequential for kilobytes at a time and cost only 8
bytes of stack per row. 1024 was within 10% of `csr_mv` on both
machines and is the default (`nblock` in the source); compiling with
`-DRBF_ELLPACK_NBLOCK=` overrides it for measuring again.

## The sliced ELLPACK product

Sliced ELLPACK (SELL-C, Kreutzer et al. 2014, without the sorting,
since every row has the same number of entries) cuts the rows into
chunks of `c` and stores each chunk in ELLPACK:
`a(c, nnzrow, nchunks)` and `ja(c, nnzrow, nchunks)`, with
`a(i, j, k)` the j-th entry of the i-th row of chunk `k`. Consecutive
rows are consecutive in memory, so the SIMD lanes take the rows of a
chunk as in ELLPACK, and a whole chunk is one contiguous run of
`c*nnzrow` values and as many indices, so the product streams two
sequential arrays as CSR does, instead of ELLPACK's `2*nnzrow`
strided streams. The rows past `n` in the last chunk are padding,
zero values with in-bounds indices.

```fortran
use rbf_sell
nchunks = sell_nchunks(n, c)
allocate(asell(c, nnzrow, nchunks), jsell(c, nnzrow, nchunks))
call sell_pack(n, nnzrow, a, ja, lda, c, asell, jsell)     ! from the fixed-row-length CSR
call sell_mv(n, nnzrow, c, alpha, asell, jsell, x, beta, y) ! y = beta y + alpha A x
```

`c` is a run-time dimension so it can be measured without rebuilding;
a multiple of the vector length, 8 for AVX-512 doubles, keeps the
inner loops free of remainders. The product has the BLAS shape and
conventions of the other two. Threads take chunks and the lanes the
rows of a chunk, with the chunk's partial sums in a stack array of
length `c`.

## CSR against ELLPACK on a kNN matrix

`examples/bench_spmv.f90` times the three products on one matrix from
`data/tools/knn_stream.py`, the 21 nearest neighbours of a million
random points in the unit square with the node itself first, as
`NodeSet::stencils` orders a row, either in the random order the
points were drawn in or Morton-sorted, as `rbf::morton_order` would
leave them; and, for comparison, the 21-point stencil of a
1000-by-1000 grid numbered along the grid. It reports effective
bandwidth in GB/s, best of 20 products, counting the bytes of `a` and
`ja`: the reads one product cannot avoid, 252 MB here. `x` is
gathered `nnzrow` times and `y` written once, from and to cache when
the ordering is good, so the true traffic is higher. Built with
`-O2 -march=x86-64-v3 -fopenmp`, on the second machine above, where a
STREAM-style copy runs at 10.5 GB/s on one thread and 38.5 GB/s on
four:

| matrix | threads | `csr_mv` | `ellpack_mv`, 32 rows | 256 rows | 1024 rows |
|---|---|---|---|---|---|
| grid stencil | 1 | 7.8 - 8.8 | 2.8 - 3.3 | 5.6 - 6.2 | 8.1 - 8.4 |
| grid stencil | 4 | 28.0 - 30.7 | 8.7 - 13.2 | 17.9 - 22.7 | 26.5 - 31.5 |
| kNN, Morton order | 1 | 7.8 - 8.1 | 2.8 - 3.1 | 4.9 - 5.2 | 6.6 - 7.7 |
| kNN, Morton order | 4 | 26.8 - 29.6 | 8.5 - 9.4 | 15.6 - 16.2 | 25.0 |
| kNN, random order | 1 | 1.8 - 2.1 | 1.8 - 2.0 | 2.0 - 2.1 | 1.9 - 2.0 |
| kNN, random order | 4 | 7.2 - 8.1 | 6.8 - 7.2 | 7.6 - 7.9 | 7.6 - 7.9 |

Ranges span gfortran 13 and flang 20, which agree within them. What
the numbers say:

- **The fixed-row-length CSR product is the one to use for a kNN
  matrix.** It is as fast as ELLPACK at ELLPACK's best block size,
  faster at every other, needs no transpose of the assembled arrays,
  and has no parameter to tune. ELLPACK's SIMD across rows does not
  pay here because the product is bound by memory, not by
  arithmetic, and the gather of `x` costs the same in both layouts.
  The module is kept for data that is in ELLPACK already.
- **Ordering is worth more than either kernel.** Morton order runs
  the kNN matrix at the speed of a grid stencil; the random order is
  4x slower on one thread, and the two kernels tie, because every
  `x` access then misses to memory and the run time is latency. That
  is what the renumbering of [renumbering.md](renumbering.md) buys.
- **Effective bandwidth.** The ordered matrices reach 8 to 9 GB/s on
  one thread and 27 to 31 GB/s on four, that is 75 to 85% of the copy
  bandwidth, before the traffic of `x` and `y`. Random order lands at
  2 GB/s and 7 to 8 GB/s. A matrix of a hundred thousand points, 25
  MB, fits the L3 and runs at 40 to 50 GB/s on four threads.

### Sliced ELLPACK against both

The same matrices and machine, GB/s of `a` and `ja`, both compilers,
chunk lengths of 8, 32 and 1024 rows for the sliced format and the
default 1024-row blocks for ELLPACK:

| matrix | threads | `csr_mv` | `ellpack_mv` | `sell_mv`, c = 8 | c = 32 | c = 1024 |
|---|---|---|---|---|---|---|
| grid stencil | 1 | 7.3 - 9.3 | 6.4 - 8.7 | 8.9 - 9.6 | 8.2 - 10.4 | 9.4 - 9.6 |
| grid stencil | 4 | 26 - 35 | 24 - 32 | 30 - 34 | 29 - 33 | 31 - 32 |
| kNN, Morton order | 1 | 7.1 - 9.0 | 6.3 - 8.2 | 8.7 - 9.2 | 8.2 - 9.4 | 8.6 - 8.9 |
| kNN, Morton order | 4 | 25 - 35 | 23 - 31 | 31 - 32 | 34 | 33 - 34 |
| kNN, random order | 1 | 1.4 - 2.2 | 1.5 - 2.3 | 2.1 - 2.2 | 1.7 - 2.4 | 1.7 - 1.8 |
| kNN, random order | 4 | 6 - 9 | 6 - 9 | 7 - 8 | 7 - 8 | 7 |

On one thread the sliced format is the fastest of the three, 5 to
15% ahead of CSR: there the core is the limit, and lanes across rows
with two sequential streams beat a gather-and-reduce per row. On four
threads the two tie, at 80 to 90% of the copy bandwidth, which is the
limit that counts for a run that uses the machine; ELLPACK stays 5 to
10% behind both. The chunk length hardly matters between 8 and 1024,
which is the point of the format: the streams are sequential at any
chunk length, unlike ELLPACK's blocks. Random order flattens
everything to the latency of the `x` gather. So the sliced format is
not faster than CSR where it matters, and CSR is the storage the
assembly already produces; `rbf_sell` earns its place where the
product runs on one core, or on a GPU, which is the format's home.

### Why the ELLPACK kernel keeps a stack array

The blocked kernel above was measured against three other shapes of
the same product on the same matrices, in GB/s of `a` and `ja` at 1
and 4 threads, gfortran and flang:

| shape | grid, 1 | grid, 4 | kNN Morton, 1 | kNN Morton, 4 |
|---|---|---|---|---|
| blocked, partial sums in a stack array (the kernel) | 7 - 8 | 25 - 28 | 8 | 28 - 30 |
| a scalar accumulator per row, entries loop inside | 3 - 4 | 11 - 12 | 3 - 4 | 11 - 16 |
| the same with `simd` on the row loop | 3 | 11 - 12 | 3 - 4 | 11 - 16 |
| accumulating into `y`, no temporary | 5 - 6 | 22 - 25 | 5 - 6 | 24 |

The scalar accumulator halves the bandwidth: the entries loop is the
inner one, so no compiler vectorizes across rows, `simd` on the row
loop or not, and each row touches `2*nnzrow` cache lines a stride of
`lda` apart. Accumulating straight into `y` sweeps `y` `nnzrow` times
and lands 10 to 20% below the stack array, which keeps the partial
sums of a block in L1 instead. In random order all shapes tie at the
latency of the `x` gather.

The numbers are the machine's, and a shared one at that: the same
executable measured 79 GB/s with 32-row blocks on the first machine
and 13 on the second. Run the benchmark rather than reading the
table:

```
python data/tools/knn_stream.py 1000000 21 morton knn.bin
cmake -B build -DCMAKE_Fortran_FLAGS="-O2 -march=x86-64-v3" && cmake --build build --target bench_spmv
OMP_NUM_THREADS=4 OMP_PROC_BIND=true ./build/examples/bench_spmv knn.bin
```

## CSR product

```fortran
use rbf_solver
call csr_mv(nr, nc, nnz, val, ia, ja, alpha, x, beta, y)   ! y = beta y + alpha A x
```

The general form, with a row pointer; the fixed-row-length form of
`rbf_csr` above needs no Eigen and shares the generic name.

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
                      method=SOLVER_CG, precond=PRECOND_ILU, &
                      max_iter=500, tolerance=1.0e-10_c_double)
```

```c
#include "rbf_solver.h"
int status = rbf_solve_csr_dp(n, nnz, val, ia, ja, b, x,
                                     &err, &iter, &method, &precond, &max_iter, &tol);
```

`A x = b` with `A` in CSR, 0-based; `x` is the initial guess on input
and the solution on output. The trailing arguments are optional, in
Fortran by keyword and in C as `NULL`, and default to Eigen's own
choices:

| Argument | Values | Default |
|---|---|---|
| `method` | `SOLVER_BICGSTAB`, `SOLVER_CG` | BiCGSTAB |
| `precond` | `PRECOND_NONE`, `PRECOND_JACOBI`, `PRECOND_ILU` | diagonal (Jacobi) |
| `max_iter` | | `2 n` |
| `tolerance` | relative residual to stop at | machine epsilon |
| `res_error` | out: the relative residual reached | |
| `res_iter` | out: iterations taken | |

Conjugate gradient assumes a symmetric positive definite matrix;
BiCGSTAB takes any nonsingular one. `PRECOND_ILU` is ILUT for BiCGSTAB
and incomplete Cholesky for conjugate gradient. Eigen counts completed
passes, so a preconditioner that happens to be exact reports zero
iterations, as does an initial guess that already solves the system.

The status is one of

| Status | Meaning |
|---|---|
| `SOLVER_SUCCESS` | converged |
| `SOLVER_NUMERICAL_ISSUE` | a breakdown in the iteration or the preconditioner |
| `SOLVER_NO_CONVERGENCE` | the iteration limit was reached; `x` holds the last iterate |
| `SOLVER_INVALID_INPUT` | a `NULL` array, a nonpositive size, `ia[n] != nnz`, an enumerator outside the enums |

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

The callback is a `bind(c)` procedure of the interface `matvec_t`
(`rbf_matvec` in C),
a BLAS level-2 update, and receives the context pointer unchanged;
`c_null_ptr` when it needs none. A module procedure, not an internal
one: passing an internal procedure to C needs a trampoline, hence an
executable stack. The matrix-free solvers run unpreconditioned, so
`precond` is not among their arguments. The test `solver_fortran`
solves a system this way with the ELLPACK product above as the
operator.
