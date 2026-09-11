#!/usr/bin/env python3
"""Generate the region between two polar curves with its stencil graph.

The cloud is PolarRegion in pointclouds.generators: nodes along the
curves evenly in arc length, the interior a Poisson disk sample.
"""

from pointclouds import cli
from pointclouds.generators import PolarRegion

# the domain of Bayona, Flyer, Fornberg and Barnett (2017):
# inner 3/10 + sin(t)/10 + 3 sin(5t)/20, outer 1 + cos(t)/5 + 3 sin(4t)/20
BAYONA_OUTER = ((1.0, 0.2), (0.0, 0.0, 0.0, 0.15))
BAYONA_INNER = ((0.3,), (0.1, 0.0, 0.0, 0.0, 0.15))

EPILOG = """\
examples:
  polar_region.py bayona.node
  polar_region.py -d 0.02 --no-inner starfish.node
  polar_region.py --outer-cos 20 4 --outer-sin 0 0 0 3 \\
      --inner-cos 6 --inner-sin 2 0 0 0 3 -d 1 bayona_20.node

A curve is r(t) = C0 + C1 cos(t) + C2 cos(2t) + ... + S1 sin(t) + ...
about the origin, star-shaped, given by its cosine and sine
coefficients; a single cosine coefficient is a circle. The defaults
are the inner and outer boundaries of the variable-coefficient
elliptic test of Bayona, Flyer, Fornberg and Barnett (2017), which are
about unit size, so the default spacing is 0.05 rather than the
lattice unit; the last example is the same domain scaled by 20 at
spacing 1. The nodes of the outer curve come first, marked 1, then
those of the inner curve, marked 6. The output is a points file and a
graph file in the same numbering, or a node file with the markers if
named so (docs/file_formats.md)."""


def curve(cos, sin, default):
    """Builds a curve from the coefficient options, or the default."""
    if cos is None and sin is None:
        return default
    return (cos or [0.0], sin or [])


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "-d",
        "--spacing",
        type=cli.number(float, above=0.0),
        default=0.05,
        metavar="H",
        help="distance between neighbouring nodes; give it well below the "
        "least width of the region (default: 0.05, for the unit-size "
        "default curves)",
    )
    ap.add_argument(
        "--outer-cos",
        nargs="+",
        type=float,
        metavar="C",
        help="cosine coefficients C0 C1 ... of the outer curve "
        "(default: the Bayona outer curve)",
    )
    ap.add_argument(
        "--outer-sin",
        nargs="+",
        type=float,
        metavar="S",
        help="sine coefficients S1 S2 ... of the outer curve",
    )
    inner = ap.add_argument_group("inner curve")
    inner.add_argument(
        "--inner-cos",
        nargs="+",
        type=float,
        metavar="C",
        help="cosine coefficients of the inner curve "
        "(default: the Bayona inner curve)",
    )
    inner.add_argument(
        "--inner-sin",
        nargs="+",
        type=float,
        metavar="S",
        help="sine coefficients of the inner curve",
    )
    inner.add_argument(
        "--no-inner",
        action="store_true",
        help="no inner curve: the whole region inside the outer one",
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

    if args.no_inner and (args.inner_cos or args.inner_sin):
        ap.error("--no-inner contradicts the inner coefficients")
    cloud = PolarRegion(
        curve(args.outer_cos, args.outer_sin, BAYONA_OUTER),
        args.spacing,
        inner=(
            None
            if args.no_inner
            else curve(args.inner_cos, args.inner_sin, BAYONA_INNER)
        ),
        ncandidates=args.candidates,
        seed=args.seed,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
