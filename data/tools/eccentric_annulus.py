#!/usr/bin/env python3
"""Generate the annulus between two eccentric circles with its graph.

The cloud is EccentricAnnulus in pointclouds.generators: the domain of
the Wannier flow benchmark, its circles laid first and the interior a
Poisson disk sample.
"""

from math import pi

from pointclouds import cli
from pointclouds.generators import EccentricAnnulus

EPILOG = """\
examples:
  eccentric_annulus.py wannier.node
  eccentric_annulus.py -d 0.05 wannier_005.node
  eccentric_annulus.py --radius 20 --hole 6 -e 8 -d 1 bearing_20.node

The outer circle of radius R is about the origin, the inner one of
radius R0 is moved down to (0, -E), the narrow gap at the bottom as
the reference draws it; its nodes come after the outer circle's,
marked 6 against 1. The defaults are the cylinders of the Wannier flow
benchmark of Trask, Maxey and Hu (2016), R0 = pi/10 and R = pi/2 at
the eccentricity pi/5 (rotating at 1 and 1/2), which are of unit size,
so the default spacing is 0.1 rather than the lattice unit. The output
is a points file and a graph file in the same numbering, or a node
file with the markers if named so (docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "--radius",
        type=cli.number(float, above=0.0),
        default=pi / 2,
        metavar="R",
        help="radius of the outer circle, about the origin "
        "(default: pi/2, the Wannier benchmark)",
    )
    ap.add_argument(
        "--hole",
        type=cli.number(float, above=0.0),
        default=pi / 10,
        metavar="R0",
        help="radius of the inner circle (default: pi/10)",
    )
    ap.add_argument(
        "-e",
        "--eccentricity",
        type=cli.number(float, least=0.0),
        default=pi / 5,
        metavar="E",
        help="distance of the inner centre below the origin, along -y; "
        "0 is the concentric annulus (default: pi/5)",
    )
    ap.add_argument(
        "-d",
        "--spacing",
        type=cli.number(float, above=0.0),
        default=0.1,
        metavar="H",
        help="distance between neighbouring nodes "
        "(default: 0.1, for the unit-size default cylinders)",
    )
    ap.add_argument(
        "--candidates",
        type=cli.number(int, least=1),
        default=100,
        metavar="K",
        help="candidates a node throws before it is retired: more of them "
        "pack the nodes tighter, up to a point (default: 100)",
    )
    cli.add_graph_options(ap)
    ap.add_argument("--seed", type=int, help="seed of the sample (default: random)")
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    cloud = EccentricAnnulus(
        args.radius,
        args.spacing,
        hole=args.hole,
        eccentricity=args.eccentricity,
        ncandidates=args.candidates,
        seed=args.seed,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
