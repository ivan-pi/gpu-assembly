# Test cases

Small point clouds and their stencil graphs, in the formats of
[docs/file_formats.md](../docs/file_formats.md), for the functionality
tests. They are read by the tests and the examples, so they live here
rather than under `test/`.

Keep the cases small: they exist to exercise the code, not to measure it.
Anything larger than a few hundred kilobytes should be generated when
needed instead of committed.

A case is a pair `<name>.points` and `<name>.graph` in the same
numbering. The name is free; what it stands for is recorded in the table
below. Node sets, which are a single `.node` file and carry no graph, are
listed under [Extracted](#extracted).

## Cases

| Case | Points | k | Origin |
|---|---|---|---|

## Extracted

Node sets read off the figures of published papers with WebPlotDigitizer 4,
as [node files](../docs/file_formats.md#node-file). They carry no stencil
graph; `NodeSet::stencils` builds one from the coordinates. Digitizing
leaves a node up to a pixel from where it belongs, so the nodes on a
boundary were put back onto it exactly and their outward normals are exact.

| Case | Nodes | Boundary | Geometry | Markers | From |
|---|---|---|---|---|---|
| `wright_disk_200.node` | 200 | 56 | the disk `r <= 1` about the origin | 1 circle | [1] |
| `wright_square_hole_892.node` | 892 | 160 | `[-1, 1]^2` less the disk `r <= 0.4` about the origin | 1 south, 2 east, 3 north, 4 west, 5 corner, 6 hole | [1] |
| `musavi_disk_108.node` | 108 | 32 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_409.node` | 409 | 64 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_1501.node` | 1501 | 128 | the disk `r <= 1` about the origin | 1 circle | [2] |

Marker 0 is an interior node in all of them.

1. Wright and Fornberg (2006), J. Comput. Phys. 212, 99-123,
   [doi:10.1016/j.jcp.2005.05.030](https://doi.org/10.1016/j.jcp.2005.05.030).
   The square with the hole is a Cartesian grid at spacing `h = 1/15` in
   which the box `[-0.6, 0.6]^2` around the hole is replaced by 220
   scattered nodes.
2. Musavi and Ashrafizaadeh (2015), Phys. Rev. E 91, 023310,
   [doi:10.1103/PhysRevE.91.023310](https://doi.org/10.1103/PhysRevE.91.023310).
   Three quasi-uniform clouds of the same disk, each spacing half the last:
   0.196, 0.098, 0.049. Only the interior nodes are in the figures as
   coordinates; the boundary nodes are equally spaced on the circle, one at
   `(1, 0)`.

## Generating

Scripts that produced a case go in `gen/`.

- `gen/cavity_refined.py`: the lid-driven cavity `[0, Lx] x [0, Ly]`,
  refined towards the walls in three levels, as a `.node` file with
  markers for the four walls and the corners. Needs numpy; `--plot`
  needs matplotlib.
