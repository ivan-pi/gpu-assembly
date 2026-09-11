# File formats

The formats read and written by `rbf::io`. Where an existing format
serves, we build on it rather than invent our own — each section names
its source — and only the parts we use are described here.

`write_vtk_polydata` is declared in `rbf_io_vtk.h` and `write_columns`
in `rbf_io_gnuplot.h`; everything else in `rbf_io.h`. Include the
headers whose formats you use.

| Format | Extension | Read | Write |
|---|---|---|---|
| [Points file](#points-file) | `.points` | `read_points`, `read_points_aos` | |
| [Graph file](#graph-file) | `.graph` | `read_graph_csr` | |
| [Node file](#node-file) | `.node` | `read_nodes`, `NodeSet(fname)` | `write_nodes`, `NodeSet::write` |
| [Grid file](#grid-file) | `.grid` | `read_grid` | `write_grid` |
| [Boundary-condition files](#boundary-condition-files) | `.bcmap`, `.mapbc` | `read_bcmap`, `read_mapbc` | |
| [Ordering file](#ordering-file) | `.iperm` | `Permutation::read`, `read_ordering` | `Permutation::write`, `write_ordering` |
| [Matrix Market](#matrix-market) | `.mtx` | | `write_matrix_market`, `write_matrix_market_pattern` |
| [VTK legacy](#vtk-legacy) | `.vtk` | | `write_vtk_polydata` |
| [Columns](#columns) | `.dat` | | `write_columns` |

The points file and the graph file go together: one gives the point cloud,
the other the stencil of every node, in the same numbering. A case is
typically a pair such as `poisson_32_21.points` and `poisson_32_21.graph`.
The ones used by the tests are in [data/](../data/README.md).

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

## Grid file

A 2D unstructured grid — nodal coordinates, triangle and quadrilateral
connectivity, and the boundary as lists of nodes — in the custom
`.grid` format of Hiroaki Nishikawa's grid-generation and EDU2D solver
codes. The three counts share the header line, the element sections
follow the coordinates directly, and the boundary section gives the
node count of every part before the first node list:

```
nnodes ntria nquad
x y            one node per line, nnodes lines
a b c          one triangle per line, ntria lines
a b c d        one quad per line, nquad lines
nbound
nb             the node count of every part, one per line
b1             then the node lists, part after part,
...            one node index per line
```

This is the layout of Nishikawa, "Making Your Own Mesh: A List of
Custom Grid Generation Codes", joint NIA & SU2 Foundation user
workshop, August 2019. Cite the format as:

> Nishikawa, Hiroaki. (2018). Unstructured grid file format (2D, 3D).
> <https://www.researchgate.net/publication/356915452_Unstructured_grid_file_format_2D_3D>
> (accessed Sep 11, 2026)

The 2018 reference presents an older sectioned variant, each count on
a line of its own before its section; that layout is not read. The
reference also describes the boundary-condition file of the
[next section](#boundary-condition-files).

Node indices in the file are 1-based, as the format prescribes. A
count that is not met, an index out of range, a repeated node within
an element, a boundary part of fewer than two nodes, and a boundary
node repeated twice in a row are all errors. The format itself has no
blank lines; the reader skips any it meets, so a file spaced apart for
readability reads the same. A grid may have no triangles, no quads, or
no boundary parts.

A boundary part lists its nodes in order along the boundary, each pair
of consecutive nodes a boundary edge. A part marks itself closed by
repeating the node where it closes — both parts of the reference's
example end on a repeat of their first node — and a part that does not
is an open polyline, ending where the next part begins;
`UnstructuredGrid::closed(b)` tells the two apart.

The reference sets three orientation conventions: element nodes are
ordered counterclockwise, the boundary node ordering is induced by the
element node ordering, and the domain is always on your left while
walking along a boundary — so the outer boundary runs counterclockwise
and holes clockwise. Parsing accepts any orientation;
`UnstructuredGrid::check_orientation()` verifies the conventions on
demand and returns `true` when the grid follows them. It checks that
every element's signed area is positive and that every part edge is an
element edge no element uses in reverse — a mesh-boundary edge, walked
in the element-induced direction with the domain on its left — and
flags elements that are not counterclockwise, part edges that are
reversed, interior, or absent from the elements, directed element
edges used twice, and mesh-boundary edges no part walks. The check is
silent by default; given a stream, it prints one message per
violation:

```cpp
if (!g.check_orientation(&std::cerr)) { /* the messages name each fault */ }
```

`read_grid` returns an `rbf::UnstructuredGrid<T, I>`, the class in
`rbf_grid.h` (included by `rbf_io.h`) that owns the grid: `x()`,
`y()`, the connectivity `tri()` and `quad()` row-major, and the node
list of each boundary part in `bound()`. The required second argument
of `read_grid`, an `rbf::IndexBase`, chooses which base the indices
are kept in once read: `IndexBase::one` keeps them native,
`IndexBase::zero` shifts them to 0-based, the numbering of the graph
file and the rest of the library. The grid records the choice in
`base()`, which `markers()` and `write_grid` consult, so a grid cannot
be handed on in the wrong base; the file `write_grid` writes is
1-based either way. The class's constructor takes the arrays directly,
moved in, and asserts the same structural rules the reader enforces on
a file, so a grid that constructs also round-trips through
`write_grid` and back.

`UnstructuredGrid::markers()` bridges to the [node file](#node-file)
convention: a node gets the number of the first boundary part that
lists it (from 1), or 0 if no part does, in either base. `NodeSet` has
a constructor taking an `UnstructuredGrid`: the nodes with the markers
as the flag, the connectivity dropped. An lvalue grid stays usable and
only its coordinates are copied; pass it with `std::move` to move the
coordinate arrays in instead (`NodeSet` owns its geometry, since
`renumber` permutes it in place, so a non-owning view is not an
option):

```cpp
auto g = rbf::io::read_grid("case.grid", rbf::IndexBase::one);
rbf::NodeSet<double> ns(std::move(g));   // flag b: node on boundary part b
ns.write("case.node");                   // the same nodes as a node file
```

## Boundary-condition files

The condition to apply on each boundary part of a
[grid file](#grid-file), by the part's tag — the number
`UnstructuredGrid::markers()` assigns. Two dialects, one reader each;
both return `std::vector<BoundaryCondition>` in file order, treat `!`
as starting a comment, skip blank lines, and reject a tag listed
twice.

`read_bcmap` reads the `.bcmap` of Nishikawa's EDU2D/3D solvers (the
[grid file](#grid-file) reference describes it): one part per line,
its tag and the name of its condition, read to end of file:

```
! Boundary tag  BC name
1 freestream
2 subsonic_outflow
3 viscous_wall
```

The record's `name` holds the condition's name; `bc` stays 0.

`read_mapbc` reads FUN3D's `.mapbc` (the FUN3D manual, appendix B,
<https://fun3d.larc.nasa.gov/>): the number of boundary groups on the
first line, then one line per part with its tag, the FUN3D
boundary-condition number, and optionally a family name:

```
13
1 6662 box_ymin
2 5025 box_zmax
...
13 3000 wing_tip
```

The number goes to `bc` and the family to `name`, empty when absent. A
count that is not met, or anything after the last group, is an error.

Names are single tokens, read up to the next whitespace. Neither
reader checks the tags against a grid, since the two files stand
alone; a solver would look each `markers()` value up among the tags.

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

## VTK legacy

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
