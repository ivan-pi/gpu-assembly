# File formats

The formats read and written by `rbf::io`. Two of them are our own
plain-text formats, two are borrowed from Triangle and METIS, and the rest
are standards or conventions of which only the parts we use are described
here.

```cpp
#include "rbf_io.h"          // points, graph, node, ordering, Matrix Market
#include "rbf_io_vtk.h"      // write_vtk_polydata
#include "rbf_io_gnuplot.h"  // write_columns
```

All three are header-only, and the two writers pull in `rbf_io.h`
themselves, as do `rbf_reorder.h` and `rbf_nodeset.h`.

| Format | Extension | Read | Write |
|---|---|---|---|
| [Points file](#points-file) | `.points` | `read_points`, `read_points_aos` | |
| [Graph file](#graph-file) | `.graph` | `read_graph_csr` | |
| [Node file](#node-file) | `.node` | `read_nodes`, `NodeSet(fname)` | `write_nodes`, `NodeSet::write` |
| [Ordering file](#ordering-file) | `.iperm` | `Permutation::read`, `read_ordering` | `Permutation::write`, `write_ordering` |
| [Matrix Market](#matrix-market) | `.mtx` | | `write_matrix_market`, `write_matrix_market_pattern` |
| [VTK legacy](#vtk) | `.vtk` | | `write_vtk_polydata` |
| [Columns](#columns) | `.dat` | | `write_columns` |

The points file and the graph file go together: one gives the point cloud,
the other the stencil of every node, in the same numbering. A case is
typically a pair such as `poisson_32_21.points` and `poisson_32_21.graph`.

Every reader exits with a message naming the file if it cannot be opened,
and readers whose format announces a count check that they got that many.
Every writer emits numbers as the shortest text that reads back to the same
value, so a file written here reproduces the values it was given.

## Points file

The point count, then one coordinate pair per line:

```
n
x0 y0
x1 y1
...
```

The order of the lines is the node numbering, which the graph file refers
to. `read_points` returns separate `x` and `y` arrays (SoA, the layout the
assembly kernels take); `read_points_aos` fills a container of two-element
structs or arrays. Anything after the `n`-th pair is ignored. A short file
or a coordinate that is not finite is an error.

## Graph file

The stencils of all nodes in compressed sparse row form, one line per node:

```
n nnz
j00 j01 j02 ...
j10 j11 ...
...
```

The header gives the node count `n` and `nnz`, the number of edges of the
stencil graph, which is also the number of nonzeros of the assembled
matrix. Line `i` lists the stencil of node `i`: the indices of the nodes it
depends on, including `i` itself, which comes first in the files produced
so far. Rows may have different lengths, but every row must have at least
one entry, since a node with no neighbours gives a singular system.

Indices are 0-based. An index outside `[0, n)` is an error, which is also
how a 1-based file is caught: its largest index equals `n`. An index listed
twice in the same row is an error too, since it would put two entries at
one matrix position.

`read_graph_csr(fname, k)` with `k > 0` declares that every row has exactly
`k` entries, as for k-nearest-neighbour stencils. The file format is the
same; the reader additionally requires `nnz == n * k` and rejects a row of
any other length.

The layout follows the METIS graph file (METIS manual 5.1.0, section
4.1.1, <https://github.com/KarypisLab/METIS>), but the two are not
interchangeable:

- METIS numbers vertices from 1; we number from 0.
- The METIS header is `n m`, with `m` the number of undirected edges, each
  counted once; our `nnz` counts every directed edge, that is every entry.
- METIS lines list the vertices adjacent to a vertex, in an undirected
  graph; our lines are stencils, which include the node itself and need not
  be symmetric: node `j` may be in the stencil of `i` without `i` being in
  the stencil of `j`.
- METIS skips lines starting with `%` as comments; we have no comments.

To hand a stencil graph to METIS or KaHIP it has to be symmetrised, which
amounts to forming the pattern of `A + A^T`, with the diagonal dropped and
the indices shifted by one.

## Node file

Nodes with a boundary marker, in the format of Triangle's `.node` file
(<https://www.cs.cmu.edu/~quake/triangle.node.html>). A header line, then
one vertex per line:

```
n 2 nattr nmark
i x y a0 ... a(nattr-1) [marker]
...
```

The header gives the vertex count, the dimension (always 2), the number of
per-vertex attributes, and whether a boundary-marker column is present
(`nmark` is 0 or 1). `#` starts a comment and blank lines may appear
anywhere. Vertices are numbered consecutively from 0 or from 1; both are
accepted, and the line order is the node numbering. We write from 0, to
match the graph file.

The marker is an integer: 0 marks an interior node, any nonzero value a
boundary node. Distinct nonzero values can tag distinct boundary segments;
`NodeSet::indices_with(value)` selects by them. A bitmask encoding several
properties works too, since only zero versus nonzero matters to the
library. Without a marker column every node is interior.

Attributes are per-vertex reals, such as physical quantities attached to the
nodes. `read_nodes` returns them row-major when asked for and skips them
otherwise; `write_nodes` takes them the same way. The library itself does
not use them.

This is the file `NodeSet` is constructed from, with the marker as its
flag, and the file `NodeSet::write` produces. NodeSet keeps coordinates and
flags only: attributes in the file are skipped on reading and the file it
writes has none. Renumbering (`NodeSet::renumber`) permutes the arrays in
memory, and `NodeSet::write` saves the result in the new order, so a
reordered file can be read back as is, minus any attributes it had.

## Ordering file

A permutation of the nodes, in the format of the METIS ordering file
(METIS manual 5.1.0, section 4.2.2; `ndmetis` writes it as
`graphfile.iperm`). One integer per line, no header:

```
iperm0
iperm1
...
```

Line `i` holds the new index of node `i`, 0-based, so the file is the
inverse permutation `iperm`: old index to new index. The values must be a
permutation of `0 .. n-1`, which the reader checks.

`Permutation::write` stores a permutation this way and `Permutation::read`
gets it back. `NodeSet::file_order()` is the renumbering a NodeSet has
applied since it was read, with `i` the position in the file it was read
from, so writing it lets per-node data kept in file order be brought into
the same numbering elsewhere:

```cpp
ns.file_order().write("case.iperm");
...
auto p = rbf::Permutation<int>::read("case.iperm");
p.permute(std::span{u});   // u: file order -> current numbering
```

`read_ordering` and `write_ordering` in `rbf::io` handle the raw vector,
which is `Permutation::inv()`, the old-to-new direction.

## Matrix Market

The coordinate format from <https://math.nist.gov/MatrixMarket/formats.html>,
written for inspection in MATLAB, SciPy or Octave, or as input to a
reference solver. We write

```
%%MatrixMarket matrix coordinate real general
rows cols nnz
i j value
...
```

with 1-based indices, as the format requires, whatever the base of the CSR
arrays passed in. `write_matrix_market_pattern` writes the `pattern`
variant, with `i j` lines and no values, to compare sparsity structures
without the weights. The format does not prescribe a precision for the
values; we write each as the shortest text that reads back exactly.

## VTK

Point clouds with per-node fields for ParaView, in the legacy ASCII format
(<https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html>).
The dataset is `POLYDATA` with one vertex cell per point, which is what
makes the cloud renderable, and every field goes in `POINT_DATA` under the
name given at the call site:

```cpp
rbf::io::write_vtk_polydata("u.vtk", n, x, y, {{"u", u}, {"residual", r}});
rbf::io::write_vtk_polydata("flow.vtk", n, x, y, {{"p", p}}, {{"U", ux, uy}});
```

Scalars are written as `SCALARS name type 1`, 2-d vectors as `VECTORS name
type` with a zero third component. Names are single tokens: legacy VTK
reads them up to the next whitespace.

The second line of a legacy file is a free-text header of at most 255
characters, which the format requires to be present but allows to be empty;
it defaults to `rbf point cloud` and can be given as the last argument.


## Columns

Whitespace-separated columns, one node per line, for gnuplot and anything
else that reads plain numeric text (`numpy.loadtxt`, pandas). The first line
is a `#` comment naming the columns, which gnuplot skips:

```
# x y rho ux uy
0.5 0.5 1 0 0
...
```

```cpp
rbf::io::write_columns("macros.dat", n, x, y, {{"rho", rho}, {"ux", ux}, {"uy", uy}});
rbf::io::write_columns("rcond.dat", n, x, y, {{"rcond", rc}});
```

```gnuplot
plot 'macros.dat' using 1:2:3 with points palette     # rho over (x, y)
plot 'macros.dat' using 1:2:4:5 with vectors          # velocity
```

Values read back exactly. The points file and the
node file can be plotted the same way by skipping their header line:
`plot 'case.points' skip 1 using 1:2`, and `plot 'case.node' skip 1 using
2:3:4 with points palette` to colour by marker.
