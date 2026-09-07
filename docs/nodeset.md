# Node sets

`rbf::NodeSet<T, I>` is a point cloud read from a
[node file](file_formats.md#node-file), with the renumbering it has
undergone and a lazily-built index for the stencil search:

```cpp
#include "rbf_nodeset.h"
```

`T` is the coordinate type, `I` the index type of the stencils, chosen to
match the `CsrMatrix<T, I>` they will feed (`double` and `int32_t` by
default).

The whole flow:

```cpp
rbf::NodeSet ns("case.node");
ns.renumber(rbf::morton_order(ns.x, ns.y))
  .renumber(rbf::boundary_last_by_flag(ns.flag));
auto ja = ns.stencils(k);
ns.file_order().write("case.iperm");
```

The index keeps a reference to the node set, so it is built on first use
and dropped by `renumber`. Renumbering before the first `stencils` call
therefore builds it exactly once, in the final numbering. The class is
non-copyable.

## Data

`x`, `y` and `flag` are public and always in the current numbering;
`flag[i]` is the node's boundary marker from the file, `0` for an interior
node. `bnd` lists the indices of the nonzero-flag nodes and is rebuilt on
every renumbering, so after a `boundary_last_*` ordering it is the
contiguous tail block. `num_points()`, `num_boundary()` and
`num_interior()` are the sizes. `indices_with(value)` selects a single
marker value, for problems with several kinds of boundary.

Attributes in the node file are skipped: read them with
`rbf::io::read_nodes` directly and bring them into the current numbering
with `file_order()`.

## Stencils

```cpp
auto ja = ns.stencils(k);
```

The `k` nearest neighbours of every node as `ja(k, ntot)` in Fortran
order: the neighbours of node `s` are contiguous at `ja[s*k]`, sorted by
distance, so `ja[s*k] == s`. Indices are 0-based. The search is
OpenMP-parallel. An optional second argument tunes the build of the
index; it matters only for performance, and only on the call that
actually builds it, that is the first after construction or after a
`renumber`.

Stencils extracted before a renumbering stay in the old numbering; use
`renumber_stencils` from [rbf_reorder.h](renumbering.md#renumbering-a-graph)
to bring them along.

## Renumbering and file order

`renumber(p)` permutes `x`, `y` and `flag`, rebuilds `bnd`, invalidates
the index, and returns `*this` so orderings can be chained.

`file_order()` maps the current numbering back to the order of the file
the set was read from, and is extended by every `renumber`, never reset:

```cpp
ns.file_order().permute(std::span{u});    // file order -> current
ns.file_order().unpermute(std::span{u});  // current -> file order
ns.file_order().write("case.iperm");      // Permutation::read gets it back
```

`write(fname)` saves the nodes in the current numbering as a node file,
with the flag as boundary marker, so a reordered set can be read back as
is. Only `x`, `y` and `flag` go to the file: the re-read set starts from
the identity order, so save `file_order()` alongside it if the link to the
original file matters.
