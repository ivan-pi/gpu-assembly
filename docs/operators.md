# The operators of the right-hand sides, for the device kernels

`src/rbf_ops.h` carries the RBF-FD operators of `rbf_fd.f90` (the value,
the first and second derivatives, the Laplacian, on the PHS r^q and on
the monomials) into C++ that compiles for the host and, under nvcc,
for the device, and prototypes three ways for a caller to say which
operators the right-hand sides of a stencil carry. `rbf_operators.h`
today fills interpolation columns only; whichever of the three is
adopted replaces its `fill_rhs`. `test/test_rbf_ops.cpp` checks that
the three produce the same columns, that the derivatives match central
differences of the value, and that the composed operators are what
they claim, on the ring stencil of `test/test_rbf_fd.f90` for
q = 3, 5, 7.

## The formulas, in one place

`PHS<Q>::apply<op>(dx, dy)` and `Poly<P>::term<op>(i, j, X, Y)` are
templates on the operator code, `Op`, an enum whose values are those of
`OP_*` in the Fortran module. Each has a run-time twin, `apply(op, ...)`,
that is one `switch` calling the template. So the formulas exist once,
and whether the operator is resolved at compile time or at run time is
the caller's choice per call site, not a property of the kernel.

## 1. Codes at run time

```cpp
Op ops[] = {Op::laplace, Op::dx, Op::value};
fill_rhs_dynamic<N, P, Q>(B, xs, ys, ops, xc, yc, 3, tid, nthreads);
```

The Fortran API as is: an array of codes and evaluation points, `nrhs`
a run-time number. The switch sits inside the fill loop; it is uniform
across the threads of a block, so there is no divergence, and the
compiler may unswitch it. The cost is the `switch` per element, which
is nothing next to the solve, and the loss of `NRHS` as a compile-time
constant, which `interp_config` needs for the solver type and the
shared-memory layout. In a kernel the codes would live in constant
memory or in the config as a run-time array up to a compile-time
maximum.

## 2. Codes at compile time

```cpp
fill_rhs_static<N, P, Q>(ops_list<Op::laplace, Op::dx, Op::value>{},
                         B, xs, ys, xc, yc, tid, nthreads);
```

The same codes as a non-type template pack. `ops_list<...>::size` is
`NRHS`, so the list can be the parameter of `interp_config` from which
everything else follows, and a fold expression over the pack gives
every column its own loop with the operator resolved at compile time,
no dispatch anywhere. This is the natural fit for the kernels, whose
`N`, `P`, `Q` and `NRHS` are template parameters already. The
evaluation points stay run-time data, as now.

## 3. Operator objects

```cpp
auto ops = pack(2.0 * Dxx{} + 0.5 * Dyy{},   // anisotropic Laplacian
                Dx{} * 0.6 + Dy{} * 0.8,      // directional derivative
                Value{});
fill_rhs_objects<N, P, Q>(ops, B, xs, ys, xc, yc, tid, nthreads);
```

A tag type per code (`Dx`, `Laplace`, `Value`, ...) answering
`phs<Q>(dx, dy)` and `poly<P>(i, j, X, Y)`, and `Scaled` and `Sum`
combining them, built by `*` and `+`. The result of `2.0 * Dxx{} +
0.5 * Dyy{}` is an object of type `Sum<Scaled<Basic<dxx>>, Basic<dyy>>`
holding the two coefficients: trivially copyable, so a kernel argument,
with the structure of the operator resolved at compile time and its
coefficients at run time. `Pack` is the tuple device code can hold, a
head-and-tail struct; `std::tuple` would do on the host and
`cuda::std::tuple` on the device. This is the one mechanism that names
operators the codes do not: the combinations a discretization actually
wants (an advection-diffusion operator, a boundary normal derivative)
without adding a code for each, and it composes with mechanism 2, since
a `Basic<op>` is just a code.

## What to adopt

Mechanism 2 for the kernels' configuration, since it gives `NRHS` and
specializes each column for free, with mechanism 3's objects as the
elements of the list where a combination is needed: an `ops_list` of
codes is the common case and a `Pack` of objects the general one, and
the two share the formulas. Mechanism 1 stays the host API in Fortran,
where the operator is data because the caller chains operators at run
time; on the host the switch costs nothing measurable
(`docs/rbf_fd.md`). One more thing carries over from the Fortran side:
the derivatives at the node are always taken at (0, 0), where the
monomial column collapses to a single nonzero, so a kernel may skip the
power tables for those columns.
