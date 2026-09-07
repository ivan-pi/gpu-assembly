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
listed under [Node sets](#node-sets).

## Cases

| Case | Points | k | Origin |
|---|---|---|---|

## Node sets

Point clouds with a boundary marker on every node, as
[node files](../docs/file_formats.md#node-file). They have no stencil
graph: the stencils are built from the coordinates by `NodeSet::stencils`.

| Case | Nodes | Boundary | Domain |
|---|---|---|---|
| `wright_disk_200.node` | 200 | 56 | the unit disk |
| `wright_cylinder_892.node` | 892 | 160 | the square with a circular hole |

Both are node layouts of Wright and Fornberg (2006), "Scattered node
compact finite difference-type formulas generated from radial basis
functions", J. Comput. Phys. 212, 99-123,
[doi:10.1016/j.jcp.2005.05.030](https://doi.org/10.1016/j.jcp.2005.05.030).
The paper gives them as figures only, so they were read off the figures
with WebPlotDigitizer 4 into `gen/wright2006_disk.csv` and
`gen/wright2006_cylinder.csv`, and `gen/wright2006_nodes.py` turns those
into the two files above. Digitizing quantizes a coordinate to a pixel,
about 0.0026 of the plot width, and leaves the boundary nodes scattered
about the boundary rather than on it; what the script does about that is
described in its docstring, and how far it moved which nodes it prints as
it runs.

### The unit disk

`x^2 + y^2 <= 1`, centred at the origin, so both coordinates run over
`[-1, 1]`. 200 quasi-uniform nodes, mean spacing 0.126, of which 56 lie on
the circle and carry marker 1; the interior nodes carry 0. This is the
paper's 200-node unstructured discretization of the disk.

The 56 are on the circle to machine precision, so the outward unit normal
at boundary node `i` is `(x[i], y[i])` as it stands.

### The square with a circular hole

`[-1, 1] x [-1, 1]` with a hole of radius 0.4 removed, the hole concentric
with the square at the origin. 892 nodes in two parts:

- 672 Cartesian nodes at spacing `h = 1/15`, that is the 31 x 31 tensor
  grid on `[-1, 1]`, with the box `[-0.6, 0.6] x [-0.6, 0.6]` around the
  hole taken out of it;
- the 220 scattered nodes that fill that box, mean spacing 0.060: 40 on
  the hole, 180 between the hole and the box.

The markers are those of `gen/cavity_refined.py` for the four walls, plus
one for the hole: 0 interior, 1 south (`y = -1`), 2 east (`x = 1`),
3 north (`y = 1`), 4 west (`x = -1`), 5 corner, 6 the hole. Wall nodes
have the exact wall coordinate and the corners are exactly `(+-1, +-1)`;
the hole nodes are on `r = 0.4` to machine precision, so the outward unit
normal of the domain at hole node `i` is `-(x[i], y[i]) / 0.4`, pointing
into the hole.

Only the 220 scattered nodes are digitized. The Cartesian nodes are
regenerated, which is what puts them on the outer walls exactly, and three
numbers behind that are read off the figure rather than stated in the
paper:

| Quantity | Value | How it is known |
|---|---|---|
| hole radius | 0.4 | the 40 digitized nodes on it give 0.39988 +- 0.00032, well within a pixel of 0.4 |
| box the scattered nodes fill | `[-0.6, 0.6]^2` | measured off the figure; the script's `--box` |
| Cartesian spacing | `1/15` | measured off the figure; the script's `--steps` |

The scattered nodes turn out to be symmetric under the eight symmetries of
the square to within one pixel, which the script uses to average the
digitizing error away; the cloud it writes has that symmetry exactly.

## Generating

Scripts that produced a case go in `gen/`.

- `gen/cavity_refined.py`: the lid-driven cavity `[0, Lx] x [0, Ly]`,
  refined towards the walls in three levels, as a `.node` file with
  markers for the four walls and the corners. Needs numpy; `--plot`
  needs matplotlib.
- `gen/wright2006_nodes.py`: the two domains above, from the digitized
  figures beside it, as `.node` files with every boundary node on the
  boundary. Needs numpy; `--plot` needs matplotlib.
