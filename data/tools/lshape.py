#!/usr/bin/env python3
"""Generate the L-shaped domain, graded toward its re-entrant corner.

The cloud is LShape in pointclouds.generators: arcs about the corner,
graded inside a radius for the corner singularity, uniform and clipped
to the domain beyond it.
"""

from pointclouds import cli
from pointclouds.generators import LShape

EPILOG = """\
examples:
  lshape.py --size 24 lshape_24
  lshape.py --size 24 --radius 8 --exponent 1 lshape_24_uniform.node
  lshape.py --size 1 --spacing 0.05 unit_lshape.node

Lengths are in lattice units. The domain is the square [-L, L]^2 less
the quadrant x > 0, y < 0, with the re-entrant corner at the origin;
dividing the coordinates by L gives the unit L-shape. Inside the graded
radius the spacing falls toward the corner as r^(1 - 1/beta), matched
by default to the r^(2/3) singularity of the Laplace solution there; an
exponent of 1 keeps it even, as a control case. The markers name the
outward normal of a wall (1 south, 2 east, 3 north, 4 west, 5 corner),
so the two walls of the corner share theirs with the outer walls of the
same normal. The output is a points file and a graph file in the same
numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "--size",
        type=cli.number(float, above=0.0),
        required=True,
        metavar="L",
        help="leg length of the L: each leg is L wide and 2 L long",
    )
    ap.add_argument(
        "-d",
        "--spacing",
        type=cli.number(float, above=0.0),
        default=1.0,
        metavar="H",
        help="distance between neighbouring nodes away from the corner " "(default: 1)",
    )
    ap.add_argument(
        "--radius",
        type=cli.number(float, above=0.0),
        default=None,
        metavar="R",
        help="radius of the graded region about the corner " "(default: half the size)",
    )
    ap.add_argument(
        "--exponent",
        type=cli.number(float, least=1.0),
        default=1.5,
        metavar="B",
        help="grading exponent: ring k of n sits at R (k/n)^B, so the "
        "spacing falls toward the corner as r^(1 - 1/B); 1 is even "
        "(default: 1.5, matching the r^(2/3) corner singularity)",
    )
    cli.add_graph_options(ap)
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    cloud = LShape(
        args.size,
        args.spacing,
        radius=args.radius,
        exponent=args.exponent,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
