# File formats

The formats read and written by `rbf::io` (`src/rbf_io.h`, `src/rbf_io_vtk.h`).
Three of them are our own plain-text formats; the other two are standards,
and only the parts we use are described here.

| Format | Extension | Read | Write |
|---|---|---|---|
| [Node file](#node-file) | `.nodes` | `read_nodes`, `read_nodes_aos` | |
| [Graph file](#graph-file) | `.graph` | `read_graph_csr` | |
| [NodeSet file](#nodeset-file) | | `read_nodeset`, `NodeSet(fname)` | `write_nodeset`, `NodeSet::write` |
| [Matrix Market](#matrix-market) | `.mtx` | | `write_matrix_market`, `write_matrix_market_pattern` |
| [VTK legacy](#vtk) | `.vtk` | | `write_vtk_polydata`, `write_lbm_vtk_polydata` |

The node file and the graph file go together: one gives the point cloud, the
other the stencil of every node, in the same numbering. A case is typically
a pair such as `poisson_32_21.nodes` and `poisson_32_21.graph`.

Every reader exits with a message naming the file if it cannot be opened,
and readers whose format announces a count check that they got that many.
Every writer emits floating point with enough digits to round-trip exactly.

## Node file

The node count, then one coordinate pair per line:

```
n
x0 y0
x1 y1
...
```

The order of the lines is the node numbering, which the graph file refers
to. `read_nodes` returns separate `x` and `y` arrays (SoA, the layout the
assembly kernels take); `read_nodes_aos` fills a container of two-element
structs or arrays. Anything after the `n`-th pair is ignored. A short file
is an error.

## Graph file

The stencils of all nodes in compressed sparse row form, one line per node:

```
n nnz
j00 j01 j02 ...
j10 j11 ...
...
```

The header gives the node count and the total number of entries. Line `i`
lists the stencil of node `i`: the indices of the nodes it depends on,
including `i` itself, which comes first in the files produced so far. Rows
may have different lengths, but every row must have at least one entry,
since a node with no neighbours gives a singular system.

Indices are 0-based. An index outside `[0, n)` is an error, which is also
how a 1-based file is caught: its largest index equals `n`.

`read_graph_csr(fname, k)` with `k > 0` declares that every row has exactly
`k` entries, as for k-nearest-neighbour stencils. The header must then
satisfy `nnz == n * k`, and the body is read as a flat list of `n * k`
indices: line breaks carry no meaning on that path.

The layout follows the METIS graph file (METIS manual 5.1.0, section
4.1.1, <https://github.com/KarypisLab/METIS>), but the two are not
interchangeable:

- METIS numbers vertices from 1; we number from 0.
- The METIS header is `n m`, with `m` the number of undirected edges, each
  counted once; our second number is the total count of entries.
- METIS lines list the vertices adjacent to a vertex, in an undirected
  graph; our lines are stencils, which include the node itself and need not
  be symmetric: node `j` may be in the stencil of `i` without `i` being in
  the stencil of `j`.
- METIS skips lines starting with `%` as comments; we have no comments.

To hand a stencil graph to METIS or KaHIP it has to be symmetrised, which
amounts to forming the pattern of `A + A^T`, with the diagonal dropped and
the indices shifted by one.

## NodeSet file

The file `NodeSet` is constructed from and that `NodeSet::write` produces.
One node per line, no header, read to end of file:

```
x0 y0 flag0
x1 y1 flag1
...
```

`flag` is an integer per node: 0 marks an interior node, any nonzero value a
boundary node. Distinct nonzero values can tag distinct boundary segments;
`NodeSet::indices_with(value)` selects by them. A bitmask encoding several
properties works too, since only zero versus nonzero matters to the library.

The order of the lines is the node numbering. Renumbering (`NodeSet::renumber`)
permutes the arrays in memory, and `NodeSet::write` saves the result in the
new order, so a reordered file can be read back as is.

The file is taken to be complete. Reading stops at the first line that does
not parse as two reals and an integer, without complaint.

This is a stripped-down relative of the Triangle `.node` format
(<https://www.cs.cmu.edu/~quake/triangle.node.html>): no header line, no
vertex number column, always two dimensions, and the boundary marker is
mandatory. A Triangle file needs the header and the first column removed to
be read here.

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
values; we write 17 significant digits for `double` and 9 for `float`, so
the file reproduces the matrix exactly.

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
reads them up to the next whitespace. `write_lbm_vtk_polydata` is the
lattice-Boltzmann special case, with the fields named `Density` and
`Velocity`.
