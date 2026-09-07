# File formats

The formats read and written by `rbf::io` (`src/rbf_io.h`, `src/rbf_io_vtk.h`).
Three of them are our own plain-text formats; the other two are standards,
and only the parts we use are described here.

| Format | Extension | Read | Write |
|---|---|---|---|
| [Node file](#node-file) | `.nodes` | `read_nodes`, `NodeSet(fname)` | `write_nodes`, `NodeSet::write` |
| [Point file](#point-file) | `.txt` | `read_points`, `read_points_aos` | |
| [Graph file](#graph-file) | `.graph` | `read_graph_csr` | |
| [Matrix Market](#matrix-market) | `.mtx` | | `write_matrix_market`, `write_matrix_market_pattern` |
| [VTK legacy](#vtk) | `.vtk` | | `write_vtk_polydata`, `write_lbm_vtk_polydata` |

All of these are ASCII: convenient and diffable, but slow. If reading ever
dominates a run, the answer is a binary format, not a faster parser.

Every reader exits with a message naming the file if it cannot be opened,
and readers whose format announces a count check that they got that many.
Every writer emits floating point with enough digits to round-trip exactly.

## Node file

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

## Point file

A count followed by that many coordinate pairs:

```
n
x0 y0
x1 y1
...
```

Used for point clouds that carry no per-node tag, such as the output of a
node generator before boundary nodes have been flagged. `read_points`
returns separate `x` and `y` arrays (SoA, the layout the assembly kernels
take); `read_points_aos` fills a container of two-element structs or arrays.
Anything after the `n`-th pair is ignored. A short file is an error.

## Graph file

The adjacency structure in compressed sparse row form, one line per node:

```
n nnz
j00 j01 j02 ...
j10 j11 ...
...
```

The header gives the node count and the total number of entries. Line `i`
lists the stencil of node `i`: the indices of the nodes it depends on,
including `i` itself for the stencils `NodeSet::stencils` produces. Rows may
have different lengths, but every row must have at least one entry, since a
node with no neighbours gives a singular system.

Index values are passed through unchanged. A 0-based file gives 0-based
`ja`, a 1-based file gives 1-based `ja`; the row pointer `ia` is always
0-based. Nothing in the reader checks which convention the file uses.

`read_graph_csr(fname, k)` with `k > 0` declares that every row has exactly
`k` entries, as for k-nearest-neighbour stencils. The header must then
satisfy `nnz == n * k`, and the body is read as a flat list of `n * k`
indices: line breaks carry no meaning on that path.

The layout is that of the METIS graph format
(<https://github.com/KarypisLab/METIS>, manual section 4.5), with two
differences. METIS numbers vertices from 1 and counts each undirected edge
once in the header, where we count entries. And METIS requires a symmetric
graph, while stencil graphs generally are not: node `j` may be in the
stencil of `i` without `i` being in the stencil of `j`. To hand a stencil
graph to METIS or KaHIP it has to be symmetrised first, which amounts to
forming the pattern of `A + A^T`.

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
without the weights. Values carry 17 significant digits for `double` and 9
for `float`, so the file reproduces the matrix exactly.

## VTK

Point clouds with per-node fields for ParaView, in the legacy ASCII format
(<https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html>,
section "Simple Legacy Formats"). The dataset is `POLYDATA` with one vertex
cell per point, which is what makes the cloud renderable, and every field
goes in `POINT_DATA` under the name given at the call site:

```cpp
rbf::io::write_vtk_polydata("u.vtk", n, x, y, {{"u", u}, {"residual", r}});
rbf::io::write_vtk_polydata("flow.vtk", n, x, y, {{"p", p}}, {{"U", ux, uy}});
```

Scalars are written as `SCALARS name type 1`, 2-d vectors as `VECTORS name
type` with a zero third component. Names are single tokens: legacy VTK
reads them up to the next whitespace. `write_lbm_vtk_polydata` is the
lattice-Boltzmann special case, with the fields named `Density` and
`Velocity`.

The XML formats (`.vtu`, `.vtp`, and `.pvd` time-series collections) are not
read or written yet. Point-cloud exports from other tools typically arrive
as `.vtu` files with one `VTK_VERTEX` cell per point and zlib-compressed
appended data, so a reader for that subset is the planned next step.
