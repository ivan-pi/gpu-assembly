# Renumbering

`rbf::Permutation` and the ordering builders come from one header:

```cpp
#include "rbf_reorder.h"
```

The curve keys themselves are computed by the Fortran module
`src/rbf_ordering.F90`.

Renumbering nodes along a space-filling curve puts neighbours close
together in memory, and moving the boundary nodes to the end makes the
interior a contiguous block. The header is independent of `NodeSet`, so it
can also drive offline tools that reorder node and graph files.

## Permutation

A permutation of `n` items, stored in gather form:

```cpp
p.map()[i]  // OLD index of the item that lands at NEW position i
p.inv()[j]  // NEW index of the item that sat at OLD position j
```

so `new[i] = old[p.map()[i]]`. The inverse is computed once at
construction and cached; both directions are needed, one to move data, the
other to relabel indices that point at it.

| Call | Meaning |
|---|---|
| `Permutation(v)` | construct from a new-to-old map |
| `Permutation::from_inverse(v)` | construct from an old-to-new map |
| `Permutation::identity(n)` | the identity on `n` items |
| `Permutation::read(f)` / `p.write(f)` | round trip through an [ordering file](file_formats.md#ordering-file) (`.iperm`) |
| `p.size()` | the number of items |
| `p.map()` / `p.inv()` | the two maps, as spans |
| `p.permute(a)` | `a_new[i] = a_old[map()[i]]`, in place |
| `p.unpermute(a)` | the reverse |
| `p.then(q)` | apply `p` first, then `q` |
| `p.inverse()` | the inverse permutation |

`permute` and `unpermute` work through a temporary copy of `a` and take a
`std::span`, so they apply to any per-node array, not just the coordinates.
A debug build checks that a map really is a permutation.

## Orderings

Every builder returns a `Permutation<I>` and none of them touches data:

```cpp
auto p = rbf::morton_order(x, y);            // Z-curve
auto q = rbf::hilbert_order(x, y);           // Hilbert curve
auto r = rbf::boundary_last_by_flag(flag);   // nonzero flag goes last
auto s = rbf::boundary_last_by_index(n, bnd);// the listed nodes go last
auto t = rbf::partition_last(n, pred);       // pred(i) == true goes last
```

The curve orders subdivide a bounding box `ndiv` times (default 16, at
most 31, since a level costs 2 bits of an `int64` key) and sort the points
by key. `bbox` is computed from the points unless given; supplying one
keeps the numbering comparable across point sets that share a domain.
Points must lie inside it. All the builders are stable: ties -- points in
the same cell, nodes on the same side of the partition -- keep their
relative order. `morton_keys` and `hilbert_keys` expose the raw keys for
inspection.

Chain them with `then`, coarsest criterion first, since the last one wins:

```cpp
auto p = rbf::morton_order(x, y).then(rbf::boundary_last_by_flag(flag));
```

## Renumbering a graph

Both routines apply the symmetric renumbering `A' = P A P^T`, that is
`A'(i,j) = A(p(i), p(j))` with `p = map()`, in place. Rows are repacked
(the `P` factor) and every stored column index is relabelled through
`inv()` (the `P^T` factor); within-row order is preserved.

```cpp
rbf::renumber_stencils(std::span{ja}, k, p);        // fixed width k
rbf::renumber_csr(std::span{ia}, std::span{ja}, p); // general CSR
```

Both rewrite the arrays they are given, so copy first to keep the
original graph:

```cpp
auto ja_p = ja;                                     // fixed width k
rbf::renumber_stencils(std::span{ja_p}, k, p);

auto ia_q = ia, ja_q = ja;                          // general CSR
rbf::renumber_csr(std::span{ia_q}, std::span{ja_q}, p);
```

For a fixed-width graph the `k` neighbours of node `s` are contiguous at
`ja[s*k]`, and the invariant `ja[s*k] == s` survives the renumbering.
`make_row_ptr(n, k)` builds the explicit row pointer `{0, k, 2k, ...}` for
consumers that want honest CSR.
