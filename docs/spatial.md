# Spatial search

`rbf::spatial` holds the point-cloud search structures the rest of the
library builds on, and the periodic box they search in:

```cpp
#include "rbf_spatial.h"
```

Today that is one k-d tree; a quadtree or a ball tree would go in the
same namespace.

## The periodic box

`rbf::spatial::PeriodicBox<T>` is the rectangle
`[x0, x0+Lx) x [y0, y0+Ly)` with opposite sides identified, periodic in
both directions:

```cpp
rbf::spatial::PeriodicBox<double> box{0.0, 0.0, 64.0, 64.0};  // x0, y0, Lx, Ly
```

Two operations, the two ways periodicity enters an RBF-FD run:

- `wrap_x(x)`, `wrap_y(y)` map a point into the box. Used on the cloud
  and on query points, so a departure point that has drifted out of the
  box needs no special handling.
- `minimum_image_x(dx)`, `minimum_image_y(dy)` shorten a displacement to
  the nearest of its periodic images. Used in the assembly, where a
  stencil node across the boundary must enter the RBF matrix at its
  short displacement from the centre, not its long one.

This is the C++ twin of the Fortran type `periodic_box` in
`src/rbf_periodic_box.f90`, which has its origin fixed at zero and is
what the [periodic benchmarks](periodic_benchmarks.md) use. The rounding
matches: a displacement of exactly half a period maps to `-L/2` in both.

## The k-d tree

`rbf::spatial::KdTree` is a 2-d k-d tree over a fixed cloud. It is
[SciPy's ckdtree](../third_party/ckdtree/README.md), vendored under
`third_party/` and wrapped; the tree owns a copy of the cloud, so the
caller's arrays need not outlive it.

```cpp
rbf::spatial::KdTree tree(x, y, box);   // without a box, the open plane
auto ja = tree.stencils(k);             // k-nearest-neighbour stencils
```

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
auto ja = tree.stencils(qx, qy, k);
```

`knn` is the same search with the distances kept:

```cpp
std::vector<std::intptr_t> idx(nq * k);
std::vector<double> d(nq * k);
tree.knn(qx, qy, k, idx, d);            // d may be left empty
```

Distances are true Euclidean distances, minimum-image in a periodic box.
Both entry points are OpenMP-parallel over the query points; the tree is
immutable once built, so const queries are thread-safe.

`KdTreeParams` (leaf size, median vs. midpoint splitting, whether node
boxes are shrunk onto their points) tunes the build. It trades build
time against query time and never changes the answer.

For a cloud queried once, the free functions build and drop the tree in
one call:

```cpp
auto ja = rbf::spatial::knn_stencils<double>(x, y, k, box);
auto jq = rbf::spatial::knn_stencils<double>(x, y, qx, qy, k, box);
```

These take any coordinate type `T` and widen it, since ckdtree works in
`double`. The box is optional in both, as it is in the constructor.

## Why periodicity is in the metric

A periodic search is often done by tiling the cloud with its 8 images
and searching the 9n points in an ordinary tree. ckdtree instead makes
the box part of the distance function, so the search runs on n points
and the returned indices need no folding back. The cost of a periodic
query is that of an ordinary one, and correctness does not depend on a
stencil staying inside one tile.

The tree works in box-relative coordinates internally, because ckdtree
assumes a box anchored at the origin. That is a rigid translation:
nothing but distances and indices leaves the tree, and both are
invariant under it.

ckdtree can also make a single direction periodic, which would suit a
channel. `PeriodicBox` does not expose it; the flow benchmarks are
periodic in both.
