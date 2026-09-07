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

The Python here -- the generators in `gen/`, the tools in `tools/` --
shares the `pointclouds` package beside them, which has to be installed
for the scripts to find it. From the repository root:

```
pip install -e data                # pointclouds and numpy, in place
pip install scipy matplotlib       # neighbour search, rcm, and the figures
pip install numba                  # the Poisson disk sampler
pip install pymetis                # nested dissection
pip install scikit-sparse          # minimum degree, and the fill-in count
```

scikit-sparse compiles against SuiteSparse, which it does not bring
along, so install that first. On Debian and Ubuntu:

```
sudo apt install libsuitesparse-dev
```

On macOS:

```
brew install suite-sparse
```

The extras `tools`, `reorder` and `generate` of the package pull in
the same set in one go:

```
pip install -e "data[tools,reorder,generate]"
```

The Python is formatted with [Black](https://black.readthedocs.io/) and
its imports sorted and checked with [Ruff](https://docs.astral.sh/ruff/),
with the settings in `pyproject.toml`; CI checks both. Before
committing, from this directory:

```
pip install black ruff
black .                            # reformat
ruff check --fix .                 # sort imports, drop unused ones
```

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

Scripts that produced a case go in `gen/`, and what they share with the
tools in the `pointclouds` package, laid out like the C++ library:
`pointclouds.io` reads and writes the formats above, `pointclouds.markers`
fixes the boundary-marker convention, `pointclouds.cli` the command-line
conventions. `pointclouds.poisson` fills a rectangle with points no two
of which are closer than a radius, periodic in either axis or neither,
from seed points if given, for a generator to scale to lattice units;
its interface follows `scipy.stats.qmc.PoissonDisk`, and Bridson's loop
is compiled by numba, so a million points take a few seconds. The
generators are:

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

## Reordering

`tools/reorder_graph.py` computes a renumbering of the nodes of a graph
file and writes it as an [ordering file](../docs/file_formats.md#ordering-file):

- `rcm`, reverse Cuthill-McKee from scipy, reduces the bandwidth;
- `nd`, the nested dissection of METIS through pymetis, reduces the
  fill-in of a direct factorisation;
- `amd`, the approximate minimum degree of SuiteSparse through
  scikit-sparse, reduces the fill-in too, by greedy elimination.

The report on standard error gives the bandwidth before and after and,
with scikit-sparse installed, the nonzeros of the Cholesky factor.
`inspect_points.py case.graph case.iperm --spy` draws the two patterns.

```
usage: reorder_graph.py [-h] [-m {rcm,nd,amd}] [-o FILE] [--seed SEED] GRAPH

Renumber the nodes of a graph file to reduce bandwidth (rcm) or fill-in (nd,
amd) and write the ordering file (docs/file_formats.md).

positional arguments:
  GRAPH                 the .graph file to order

options:
  -h, --help            show this help message and exit
  -m {rcm,nd,amd}, --method {rcm,nd,amd}
                        rcm: reverse Cuthill-McKee (default); nd: METIS nested
                        dissection; amd: SuiteSparse approximate minimum
                        degree
  -o FILE, --output FILE
                        the ordering file, default GRAPH with the extension
                        .iperm; - writes to standard output
  --seed SEED           the random seed of METIS, for nd
```
