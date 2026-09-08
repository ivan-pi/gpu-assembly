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

An operator is the partial derivative `D<A, B>` = d^A/dx^A d^B/dy^B of
order A + B <= 2, or a linear combination of those: `Scaled` and `Sum`,
built by `*` and `+`. The value is `D<0, 0>`, the Laplacian is
`D<2, 0> + D<0, 2>`, derived rather than special-cased; `Dx`, `Dxy`
and the rest are names for the instances. Every operator answers two
questions: on the PHS phi = r^q at a displacement (dx, dy), and on a
monomial x^i y^j from the power tables of the evaluation point,

```
phi          = p r^2                                p = r^(q-2), s = r^(q-4)
D_a phi      = q p d_a
D_a D_b phi  = q s ((q-2) d_a d_b + delta_ab r^2)

D<A,B> x^i y^j = i (i-1) ... (i-A+1)  j (j-1) ... (j-B+1)  x^(i-A) y^(j-B)
```

three lines by order for the PHS and one for the monomials, where a
case per operator would be seven of each. The `Op` enum, with the
values of `OP_*` in the Fortran module, maps to the tags through
`tag_of<op>`, so a code and a tag are the same operator by two names.
A column of a right-hand side is `fill_column`: the operator on the
PHS of every node, then on the monomials.

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
                0.6 * Dx{} + 0.8 * Dy{},      // directional derivative
                Value{});
fill_rhs_objects<N, P, Q>(ops, B, xs, ys, xc, yc, tid, nthreads);
```

The tags and their combinations. `2.0 * Dxx{} + 0.5 * Dyy{}` is an
object of type `Sum<Scaled<D<2,0>>, D<0,2>>` holding the two
coefficients: trivially copyable, so a kernel argument, with the
structure of the operator resolved at compile time and its
coefficients at run time. `Pack` is the tuple device code can hold, a
head-and-tail struct; `std::tuple` would do on the host and
`cuda::std::tuple` on the device. This is the one mechanism that names
operators the codes do not, without adding a code for each, and it is
what the other two reduce to: mechanism 2 packs the tags of its codes,
mechanism 1 is one switch per column onto the tag.

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
