#!/usr/bin/env python3
"""Generate the equilateral triangle with its stencil graph.

The cloud is EquilateralTriangle in pointclouds.generators: the
triangle on its own lattice, perturbed if asked, the domain of
McCartin's closed-form Robin eigenproblem.
"""

from pointclouds import cli
from pointclouds.generators import EquilateralTriangle

EPILOG = """\
examples:
  triangle.py -n 24 triangle_24
  triangle.py -n 24 --sigma 0.2 --seed 7 triangle_24.node
  triangle.py -n 40 -K 21 triangle_40

Lengths are in lattice units: the base runs from the origin to (N, 0),
the apex is at (N/2, N sqrt(3)/2), and the triangular lattice of side 1
fits the triangle exactly. The sides are marked by the x sign of their
outward normals, 1 (south) the base, 2 (east) the right, 4 (west) the
left, and 5 the three vertices; a perturbed side node slides along its
side and a vertex stays put. The output is a points file and a graph
file in the same numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "-n",
        "--side",
        type=cli.number(int, least=2),
        required=True,
        metavar="N",
        help="lattice spacings along a side",
    )
    ap.add_argument(
        "-s",
        "--sigma",
        type=cli.number(float, least=0.0),
        default=0.0,
        metavar="SIGMA",
        help="displacement of a node, uniform on [-SIGMA, SIGMA] spacings; "
        "a node stays in its cell below 0.5 (default: 0, the exact lattice)",
    )
    cli.add_graph_options(ap)
    ap.add_argument(
        "--seed", type=int, help="seed of the displacements (default: random)"
    )
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    if args.sigma >= 0.5:
        print(f"warning: --sigma {args.sigma:g} may put nodes on top of each other")
    cloud = EquilateralTriangle(args.side, args.sigma, seed=args.seed)
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
