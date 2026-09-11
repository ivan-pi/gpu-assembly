#!/usr/bin/env python3
"""Generate a randomly perturbed Cartesian grid with its stencil graph.

The cloud is PerturbedGrid in pointclouds.generators, the node layout of
Strzelczyk and Matyka (2022).
"""

from pointclouds import cli
from pointclouds.generators import PerturbedGrid

EPILOG = """\
examples:
  perturbed_grid.py -n 40 tg_40
  perturbed_grid.py -n 40 --sigma 0.02 -K 21 tg_40
  perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
  perturbed_grid.py -n 40 --geometry box --sigma 0 square_40.node
  perturbed_grid.py -n 40 --seed 1234 tg_40

Lengths are in lattice units: the box is N by N and the spacing 1. The
periodic box is the Taylor-Green test; the channel, periodic in x with
walls at y = 0 and y = N, the Poiseuille test, its wall nodes with the
markers 1 (bottom) and 3 (top); the box, walled on all four sides as
in the Poisson tests on the square, marks its walls 1 (south), 2
(east), 3 (north), 4 (west) and 5 (corner). A wall node slides along
its wall, a corner stays put, and --sigma 0 is the exact Cartesian
grid. The output is a points file and a graph file in the same
numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "-n",
        "--nodes",
        type=cli.number(int, least=2),
        required=True,
        metavar="N",
        help="nodes across the box",
    )
    ap.add_argument(
        "-s",
        "--sigma",
        type=cli.number(float, least=0.0),
        default=0.2,
        metavar="SIGMA",
        help="displacement of a node, uniform on [-SIGMA, SIGMA] spacings; "
        "a node stays in its cell below 0.5 (default: 0.2)",
    )
    ap.add_argument(
        "-g",
        "--geometry",
        choices=("periodic", "channel", "box"),
        default="periodic",
        help="periodic on both sides, walls at y = 0 and y = N and "
        "periodic in x, or walls on all four sides (default: periodic)",
    )
    cli.add_graph_options(ap)
    ap.add_argument(
        "--seed", type=int, help="seed of the displacements (default: random)"
    )
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    if args.sigma >= 0.5:
        print(f"warning: --sigma {args.sigma:g} may put nodes on top of each other")
    cloud = PerturbedGrid(
        args.nodes, args.sigma, geometry=args.geometry, seed=args.seed
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
