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

Node sets read off the figures of published papers, as
[node files](../docs/file_formats.md#node-file).

| Case | Nodes | Boundary | Geometry | Markers | From |
|---|---|---|---|---|---|
| `wright_disk_200.node` | 200 | 56 | the disk `r <= 1` about the origin | 1 circle | [1] |
| `wright_square_hole_892.node` | 892 | 160 | `[-1, 1]^2` less the disk `r <= 0.4` about the origin | 1 south, 2 east, 3 north, 4 west, 5 corner, 6 hole | [1] |
| `musavi_disk_108.node` | 108 | 32 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_409.node` | 409 | 64 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `musavi_disk_1501.node` | 1501 | 128 | the disk `r <= 1` about the origin | 1 circle | [2] |
| `shu_square_4786.node` | 4786 | 264 | the square `[0, 1]^2` | 1 south, 2 east, 3 north, 4 west, 5 corner | [3] |
| `barnett_disk_817.node` | 817 | 96 | the disk `r <= 1` about the origin | 1 circle | [4] |

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
4. Barnett (2015), doctoral dissertation, University of Colorado Boulder,
   [scholar.colorado.edu/.../8623hx72q](https://scholar.colorado.edu/concern/graduate_thesis_or_dissertations/8623hx72q),
   figure 4.12.

## Generating

Scripts that produced a case go in `gen/`.

- `gen/cavity_refined.py`: the lid-driven cavity `[0, Lx] x [0, Ly]`,
  refined towards the walls in three levels, as a `.node` file with
  markers for the four walls and the corners. Needs numpy; `--plot`
  needs matplotlib.
