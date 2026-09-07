# Spatial search

`rbf::spatial` holds the point-cloud search structures the rest of the
library builds on, and the periodic box they search in:

```cpp
#include "rbf_spatial.h"
```

Today that is one k-d tree; a quadtree or a ball tree would go in the
same namespace.

## The periodic box

`rbf::spatial::PeriodicBox<T, D>` is the box
`[origin[d], origin[d] + period[d])` along each of `D` axes, with
opposite faces identified:

```cpp
rbf::spatial::PeriodicBox<double, 2> box{{0.0, 0.0}, {64.0, 64.0}};
```

Two operations, the two ways periodicity enters an RBF-FD run:

- `wrap(d, x)` maps a coordinate into the box. Used on the cloud and on
  query points, so a departure point that has drifted out of the box
  needs no special handling.
- `minimum_image(d, dx)` shortens a displacement to the nearest of its
  periodic images. Used in the assembly, where a stencil node across the
  boundary must enter the RBF matrix at its short displacement from the
  centre, not its long one.

Both have whole-point forms taking and returning a `std::array<T, D>`.

The Fortran type `periodic_box` in `src/rbf_periodic_box.f90` is the 2-d
case with its origin at zero, and is what the
[periodic benchmarks](periodic_benchmarks.md) use. The rounding matches:
a displacement of exactly half a period maps to `-L/2` in both.

## The k-d tree

`rbf::spatial::KdTree` is a k-d tree over a fixed cloud. It is
[SciPy's ckdtree](../third_party/ckdtree/README.md), vendored under
`third_party/` and wrapped; the tree owns a copy of the cloud, so the
caller's arrays need not outlive it.

Points come in ckdtree's own layout, interleaved: coordinate `d` of
point `i` at `points[i*ndim + d]`. `interleave` builds that from
per-axis arrays, widening to `double` on the way, since ckdtree works in
`double`:

```cpp
auto points = rbf::spatial::interleave(x, y);        // 2-d
auto points = rbf::spatial::interleave(x, y, z);     // 3-d

rbf::spatial::KdTree tree(points, box);              // dimension from the box
rbf::spatial::KdTree tree(points, 2);                // open plane, dimension given
auto ja = tree.stencils(k);                          // k-nearest-neighbour stencils
```

The dimension is a run-time value, as it is in ckdtree, and `ndim()`
reports it. Nothing in the tree or the box is 2-d; a 3-d run costs a
different `ndim` and nothing else.

`stencils` returns `ja(k, nq)` in Fortran order: the `k` neighbours of
query `s` are contiguous at `ja[s*k]`, sorted by distance. Indices are
0-based and refer to the cloud in the order it was passed. The index
type is a template parameter, chosen to match the `CsrMatrix<T, I>` the
stencils will feed (`int32_t` by default). With no query points the
stencils are centred on the cloud's own nodes, so `ja[s*k] == s` --
unless two nodes coincide, in which case they tie at distance zero and
either may lead the row.

Queries may be arbitrary points, which is what departure-point-centred
stencils need:

```cpp
auto ja = tree.stencils(q, k);      // q interleaved, nq*ndim
```

`knn` is the same search with the distances kept:

```cpp
std::vector<std::intptr_t> idx(nq * k);
std::vector<double> d(nq * k);
tree.knn(q, k, idx, d);             // d may be left empty
```

Distances are true Euclidean distances, minimum-image in a periodic box.
Both entry points are OpenMP-parallel over the query points; the tree is
immutable once built, so const queries are thread-safe.

`KdTreeParams` (leaf size, median vs. midpoint splitting, whether node
boxes are shrunk onto their points) tunes the build. It trades build
time against query time and never changes the answer.

## Relation to SciPy's cKDTree

The vendored `third_party/ckdtree/src` is only SciPy's C++ kernel: eight
free functions over a `ckdtree` struct that owns nothing and validates
nothing. The class around it lives in `_ckdtree.pyx`, which is not part
of the vendored code, so `KdTree` is that layer rather than a re-wrapping
of an existing C++ interface.

Where the two line up:

| `cKDTree` | `KdTree` |
| --- | --- |
| `cKDTree(data, leafsize, compact_nodes, balanced_tree, boxsize)` | `KdTree(points, box, params)` |
| `boxsize=None` | the `(points, ndim)` constructor |
| `.query(x, k)` -> `(d, i)` | `.knn(q, k, idx, dist)` |
| `.n`, `.m` | `.size()`, `.ndim()` |

Two deliberate differences. `boxsize` is lengths only, with the box
implicitly at the origin and the data required to already lie in
`[0, L)` -- SciPy raises otherwise; `PeriodicBox` carries an origin and
wraps for you. And `boxsize` may zero out a single axis to leave it
aperiodic, which `PeriodicBox` does not expose.

`query`'s `eps`, `p` and `distance_upper_bound` are fixed here at the
exact Euclidean search: `0`, `2` and infinity. The vendored ckdtree also
carries radius queries (`query_ball_point`, `query_ball_tree`), all-pairs
searches (`query_pairs`), neighbour counts and sparse distance matrices,
compiled into the `ckdtree` target but not wrapped.

## Why periodicity is in the metric

A periodic search is often done by tiling the cloud with its images and
searching the `3^d` copies in an ordinary tree. ckdtree instead makes the
box part of the distance function, so the search runs on `n` points and
the returned indices need no folding back. The cost of a periodic query
is that of an ordinary one, and correctness does not depend on a stencil
staying inside one tile.

The tree works in box-relative coordinates internally, because ckdtree
assumes a box anchored at the origin. That is a rigid translation:
nothing but distances and indices leaves the tree, and both are
invariant under it.
