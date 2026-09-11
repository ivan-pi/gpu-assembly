#!/usr/bin/env python3
"""Generate a reentrant-corner domain, graded toward the corner.

The cloud is ReentrantCorner in pointclouds.generators: a square less
a wedge at the corner angle, its arcs graded toward the corner for the
corner singularity, uniform and clipped to the domain beyond.
"""

from math import pi

from pointclouds import cli
from pointclouds.generators import ReentrantCorner

EPILOG = """\
examples:
  reentrant_corner.py --size 24 lshape_24
  reentrant_corner.py --size 24 --omega 1.75 corner_775.node
  reentrant_corner.py --size 24 --exponent 1 lshape_uniform.node
  reentrant_corner.py --size 1 --spacing 0.05 unit_lshape.node

Lengths are in lattice units and the angle in units of pi. The domain
is the sector 0 <= theta <= W pi of the square [-L, L]^2, the corner
at the origin and one edge of it along the positive x axis, the family
of Mitchell (2013): the Laplace solution r^(1/W) sin(theta / W) is
singular at the corner for W > 1, and the spacing falls toward the
corner as r^(1 - 1/W) to match, unless --exponent overrides it. An
angle of 1.5 is the L-shaped domain; 2, the square with a slit, is
excluded. The edge on the x axis is marked 1 and the edge at the angle
2, the walls of the square 1-4 by outward normal and the corners 5.
The output is a points file and a graph file in the same numbering, or
a node file with the markers if named so (docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "--size",
        type=cli.number(float, above=0.0),
        required=True,
        metavar="L",
        help="half-side of the square the sector is cut from",
    )
    ap.add_argument(
        "-w",
        "--omega",
        type=cli.number(float, above=1.0, below=2.0),
        default=1.5,
        metavar="W",
        help="angle of the corner, in units of pi "
        "(default: 1.5, the L-shaped domain)",
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
        default=None,
        metavar="B",
        help="grading exponent: arc k of n sits at R (k/n)^B, so the "
        "spacing falls toward the corner as r^(1 - 1/B); 1 is even "
        "(default: W, matching the corner singularity)",
    )
    cli.add_graph_options(ap)
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    cloud = ReentrantCorner(
        args.size,
        args.spacing,
        omega=args.omega * pi,
        radius=args.radius,
        exponent=args.exponent,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
