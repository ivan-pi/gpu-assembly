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

The Python here -- the generators in `gen/`, the tools in `tools/` -- may
use numpy, scipy and matplotlib.

## Cases

| Case | Points | k | Origin |
|---|---|---|---|

## Extracted

Node sets read off the figures of published papers with WebPlotDigitizer 4,
as [node files](../docs/file_formats.md#node-file). A digitized coordinate
is imprecise, limited by the pixel resolution of the figure, so wherever the
shape of the boundary is known its nodes were reconstructed onto it to
machine precision, which is what makes the normal taken from a boundary node
exact.

| Case | Nodes | Boundary | Geometry | Markers | From |
|---|---|---|---|---|---|
| `wright_disk_200.node` | 200 | 56 | the disk `r <= 1` about the origin | 1 circle | [1] |
| `wright_square_hole_892.node` | 892 | 160 | `[-1, 1]^2` less the disk `r <= 0.4` about the origin | 1 south, 2 east, 3 north, 4 west, 5 corner, 6 hole | [1] |
| `musavi_disk_108.node` | 108 | 32 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_409.node` | 409 | 64 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_1501.node` | 1501 | 128 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `shu_square_4786.node` | 4786 | 264 | the square `[0, 1]^2` | 1 south, 2 east, 3 north, 4 west, 5 corner | [3] |

Marker 0 is an interior node in all of them.

1. Wright and Fornberg (2006), J. Comput. Phys. 212, 99-123,
   [doi:10.1016/j.jcp.2005.05.030](https://doi.org/10.1016/j.jcp.2005.05.030).
   The square with the hole is a Cartesian grid at spacing `h = 1/15` in
   which the box `[-0.6, 0.6]^2` around the hole is replaced by 220
   scattered nodes.
2. Musavi and Ashrafizaadeh (2015), Phys. Rev. E 91, 023310,
   [doi:10.1103/PhysRevE.91.023310](https://doi.org/10.1103/PhysRevE.91.023310).
   Three quasi-uniform clouds of the same disk at node spacings of about
   0.2, 0.1 and 0.05.
3. Shu, Ding and Zhao (2006), Comput. Math. Appl. 51, 1297-1310,
   [doi:10.1016/j.camwa.2006.04.015](https://doi.org/10.1016/j.camwa.2006.04.015).

## Generating

Scripts that produced a case go in `gen/`, and what they share -- the
writers for the formats above, the boundary-marker convention and the
command-line conventions -- in the `gen/pointclouds` package beside them.
The generators are:

- `gen/cavity_refined.py`: the lid-driven cavity `[0, Lx] x [0, Ly]`,
  refined towards the walls in three levels, as a `.node` file with
  markers for the four walls and the corners.
- `gen/perturbed_grid.py`: a Cartesian grid with every node displaced by
  a small random amount, periodic on both sides or a channel with walls
  at the top and bottom, as a `.points` or a `.node` file together with
  the `.graph` of its stencils, whose search wraps around the periodic
  sides.

They work in lattice units, in which the spacing is 1; scaling a case to
other units is left to whoever needs it. `--help` describes the rest.

## Inspecting

`tools/inspect_points.py` reads and checks the files of a case and
reports on them.

```
usage: inspect_points.py [-h] [--plot] [--labels] [--stencil I [I ...]]
                         [--k K] [--periodic LX LY] [--spy] [--save FILE]
                         FILE [FILE ...]

Check and describe the files of a case (docs/file_formats.md): node and marker
counts, nearest-neighbour statistics, graph statistics. Never writes a file.

positional arguments:
  FILE                 one each of .points or .node, .graph and .iperm; an
                       ordering is applied to the others

options:
  -h, --help           show this help message and exit
  --plot               draw the nodes, coloured by marker
  --labels             write the index next to every node
  --stencil I [I ...]  draw the stencils of these nodes, from the graph or the
                       --k nearest neighbours
  --k K                stencil size for --stencil without a graph file
  --periodic LX LY     the periodic box [0, LX) x [0, LY): minimum-image
                       distances
  --spy                draw the sparsity pattern of the graph, before and
                       after an ordering file
  --save FILE          write the figure to FILE instead of showing it
```
