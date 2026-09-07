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

The scripts in `tools/` generate, inspect and reorder cases, and share
the `pointclouds` package beside them, which has to be installed for
them to find it. From the repository root:

```
pip install -e data                # pointclouds, numpy and scipy, in place
pip install matplotlib             # the figures
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
with the settings in `pyproject.toml`; CI checks both, through the
pre-commit configuration at the repository root (see
[docs/developers.md](../docs/developers.md)). By hand, from this
directory:

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

The scripts share the `pointclouds` package, laid out like the C++
library:

- `pointclouds.nodeset`: `NodeSet`, a cloud with a marker per node and
  the box it lives in, which writes, selects its stencils and draws
  itself; `TiledNodeSet`, copies of it side by side; `MARKERS`.
- `pointclouds.generators`: the clouds the generators make, as children
  of `NodeSet`, so that a script is its command line and a constructor.
- `pointclouds.stencils`: the stencil selection, by nearest neighbours,
  radius or range, wrapping around the periodic sides of the box.
- `pointclouds.poisson`: a Poisson disk sampler of a rectangle, periodic
  or not, after `scipy.stats.qmc.PoissonDisk` and compiled by numba;
  `tools/poisson_demo.py` shows a sample of it.
- `pointclouds.periodic`, `pointclouds.io`, `pointclouds.cli`: the box
  arithmetic, the file formats above, the command-line conventions.

The generators, in lattice units with the spacing 1; `--help` has the
rest:

- `tools/refined_cavity.py`: the lid-driven cavity, refined towards the
  walls in three bands, with markers for the walls and the corners.
- `tools/perturbed_grid.py`: a Cartesian grid with every node displaced
  by a small random amount, periodic or a channel with walls.
- `tools/poisson_box.py`: a Poisson disk sample of a periodic box, alone
  or around a circular hole (an array of cylinders), tiled if asked.

## Inspecting

`tools/inspect_points.py` reads and checks the files of a case and
reports on them.

```
usage: inspect_points.py [-h] [--plot] [--labels] [--stencil I [I ...]] [-K K]
                         [--periodic LX LY] [--spy] [--save FILE]
                         FILE [FILE ...]

Check and describe the files of a case, and draw them.

positional arguments:
  FILE                 one each of .points or .node, .graph and .iperm; an
                       ordering is applied to the others

options:
  -h, --help           show this help message and exit
  --plot               draw the nodes, coloured by marker
  --labels             write the index next to every node
  --stencil I [I ...]  draw the stencils of these nodes, from the graph or as
                       the K nearest
  -K K, --knn K        the stencil of a node is its K nearest nodes, without a
                       graph file
  --periodic LX LY     the periodic box [0, LX) x [0, LY): minimum-image
                       distances
  --spy                draw the sparsity pattern of the graph, before and
                       after an ordering file
  --save FILE          write the figure to FILE instead of showing it

examples:
  inspect_points.py case.node --plot --labels
  inspect_points.py case.points case.graph --stencil 0 17
  inspect_points.py case.points case.graph case.iperm --spy
  inspect_points.py case.points --periodic 32 32

One file of each kind, told apart by extension. Each is checked against
its header and the files against each other; the problems found are
listed at the end and make the exit status 1. An ordering file is
applied to the nodes and the graph before they are drawn, so --labels
shows the new indices; only the spy plot also shows the file order.
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

Renumber the nodes of a graph file and write the ordering file.

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

examples:
  reorder_graph.py case.graph                # case.iperm, by rcm
  reorder_graph.py case.graph --method nd    # nested dissection
  reorder_graph.py case.graph -o -           # to standard output

rcm, from scipy, reduces the bandwidth by numbering neighbouring nodes
close together. nd, METIS through pymetis, and amd, SuiteSparse through
scikit-sparse, reduce the fill-in of a sparse direct factorisation, by
recursive bisection and by greedy elimination. All three order the
undirected graph of the stencils, the pattern of A + A^T without the
diagonal. The report, on standard error so that `-o -` leaves the file
alone, gives the bandwidth before and after and, with scikit-sparse, the
nonzeros of the Cholesky factor; `inspect_points.py case.graph
case.iperm --spy` shows the effect.
```
