#!/usr/bin/env python3
"""Point clouds for the lid-driven cavity [0, Lx] x [0, Ly], refined
towards the walls in three levels, written as a node file with boundary
markers.

    python3 cavity_refined.py cavity.node
    python3 cavity_refined.py --steps 20 cavity_fine.node
    python3 cavity_refined.py --distribution grid --size 100 130 cavity_tall.node

Everything is in lattice units: the spacing is h = 1 at the wall, 1.5 in
the second band and 2.5 in the middle, so a band from each wall is N,
1.5 N and 2.5 N wide and the three of them from both walls fill a cavity
of 10 N, which is the size a bare `--steps N` gives. The proportions are
those of the figure in Lin, Wu and Zhang (2019), "A mesh-free radial
basis function-based semi-Lagrangian lattice Boltzmann method for
incompressible flows", Int. J. Numer. Meth. Fluids 91, 198-211, which
shows the point distribution without specifying it.

`--size LX LY` gives the cavity a size of its own, in lattice units and
so at least the 10 N the bands need; a side longer than that gets its
middle at the coarsest spacing. Scaling a case to other units is left to
whoever needs it. `--steps N` refines or coarsens all three bands
together, which widens the cavity, since the spacing at the wall stays
1.

Two distributions realise the spacings and give different clouds:

  rings  Concentric rectangles inset from the walls, with the points h
         apart along a ring and consecutive rings h apart. The spacing is
         the same along and across a ring, but where it changes the
         points of neighbouring rings do not line up. The rings end in
         the centre point of a square; in a rectangle they go on at the
         coarsest spacing until they degenerate to a segment. Each side
         of a ring holds a whole number of spacings, so the spacing
         along a ring can differ from h by a fraction of a percent.
  grid   The tensor product of 1-d coordinates graded the same way. Rows
         and columns line up everywhere, but a point near the middle of
         a wall sits in a cell of h by 2.5 h.

Both put the wall nodes first in the file, then the interior: from the
wall inwards for the rings, row by row for the grid.

Boundary markers (docs/file_formats.md, node file):

    0  interior
    1  south wall, y = 0
    2  east wall,  x = Lx
    3  north wall, y = Ly, the lid
    4  west wall,  x = 0
    5  corner, where two walls meet

The walls are numbered counter-clockwise from the bottom, the numbering
the generators share (pointclouds/formats.py). The corners get
their own marker because a corner node belongs to two walls with, in the
cavity, different boundary data: the lid velocity meets the wall's no-slip.
Whoever assembles the boundary conditions decides what a corner gets.

The output file is a node file, named after the argument with `.node`
appended if it is not there already. Its first line is a comment with
the command that produced it, and every node carries its marker. A name
ending in `.points` gives a points file instead, a format with no
comment line and no markers: only the coordinates are written. The name
`-` writes to standard output, `-- -.points` the points file there,
after the option separator since the name starts with a dash. The report
goes to standard error, so the stream carries the file alone.
"""

import argparse
import sys

import numpy as np

from pointclouds import (CORNER, EAST, INTERIOR, MARKER_STYLE, NORTH, SOUTH,
                         WEST, number, output_stem, write_node, write_points)

SPACINGS = (1.0, 1.5, 2.5)     # at the wall, in the second band, in the middle

TOL = 1e-9


def levels(N, for_rings):
    """(spacing, count) per band for N spacings across each band: the number
    of rings, or of steps of the 1-d grid coordinate. Consecutive rings are
    one spacing of the outer ring apart, so a band holds N + 1 rings and
    the first ring of the next band lies one spacing of this band inside
    its last. The last band holds N rings and ends in the centre point,
    which works out because the first two bands together are as wide as
    the third."""
    counts = (N + 1, N + 1, N) if for_rings else (N, N, N)
    return list(zip(SPACINGS, counts))


def side(a, b, h):
    """From a to b in a whole number of steps as close to h as possible;
    the end b is left out."""
    n = max(round(abs(b - a) / h), 1)
    return np.linspace(a, b, n + 1)[:-1]


def ring(x0, x1, y0, y1, h):
    """Points about h apart on the boundary of [x0, x1] x [y0, y1], counter-
    clockwise from (x0, y0); a segment or a point when a side is zero."""
    if x1 - x0 < TOL and y1 - y0 < TOL:
        return np.array([[x0, y0]])
    if x1 - x0 < TOL:
        return np.column_stack((np.full(len(side(y0, y1, h)) + 1, x0),
                                np.append(side(y0, y1, h), y1)))
    if y1 - y0 < TOL:
        return np.column_stack((np.append(side(x0, x1, h), x1),
                                np.full(len(side(x0, x1, h)) + 1, y0)))
    sx, sy = side(x0, x1, h), side(y0, y1, h)
    south = np.column_stack((sx, np.full(len(sx), y0)))
    east = np.column_stack((np.full(len(sy), x1), sy))
    north = np.column_stack((x1 + x0 - sx, np.full(len(sx), y1)))
    west = np.column_stack((np.full(len(sy), x0), y1 + y0 - sy))
    return np.vstack((south, east, north, west))


def rings(Lx, Ly, levels):
    spacings = [h for h, count in levels for _ in range(count)]
    pts = []
    r = 0.0
    k = 0
    while Lx - 2 * r > -TOL and Ly - 2 * r > -TOL:
        h = spacings[min(k, len(spacings) - 1)]
        pts.append(ring(r, Lx - r, r, Ly - r, h))
        r += h
        k += 1
    return np.vstack(pts)


def graded_coordinate(L, levels):
    """From the wall at 0 to the far wall at L: the levels inwards from
    both walls, and the middle at the coarsest spacing."""
    z = [0.0]
    for h, count in levels:
        z.extend(z[-1] + h * np.arange(1, count + 1))
    z = np.array(z)
    if L - 2 * z[-1] > TOL:
        middle = side(z[-1], L - z[-1], levels[-1][0])
    else:
        middle = np.empty(0)                 # the unit side: the bands meet
    return np.concatenate((z[:-1], middle, L - z[::-1]))


def grid(Lx, Ly, levels):
    x, y = np.meshgrid(graded_coordinate(Lx, levels), graded_coordinate(Ly, levels))
    pts = np.column_stack((x.ravel(), y.ravel()))
    on_wall = markers(pts, Lx, Ly) != INTERIOR
    return np.vstack((pts[on_wall], pts[~on_wall]))


def markers(pts, Lx, Ly):
    x, y = pts[:, 0], pts[:, 1]
    s, e, n, w = y < TOL, x > Lx - TOL, y > Ly - TOL, x < TOL
    m = np.full(len(pts), INTERIOR)
    m[s], m[e], m[n], m[w] = SOUTH, EAST, NORTH, WEST
    m[(s | n) & (e | w)] = CORNER
    return m


HELP = __doc__.split("\n\n")[0] + """

Lengths are in lattice units, in which the spacing is 1 at the wall and
the cavity is 10 N by 10 N unless --size makes it bigger. Boundary
markers: 0 interior, 1 south wall, 2 east, 3 north (the lid), 4 west,
5 corner. The docstring at the top of the script describes the two point
distributions and the three bands of refinement."""


def main():
    ap = argparse.ArgumentParser(description=HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="output file: a node file with the markers, or a "
                    "points file without them if the name ends in .points; "
                    "`-` is standard output")
    ap.add_argument("--distribution", choices=("rings", "grid"), default="rings",
                    help="concentric rectangles, or a tensor-product grid (default: rings)")
    ap.add_argument("--size", nargs=2, type=number(float, above=0.0),
                    metavar=("LX", "LY"),
                    help="sides of the cavity in lattice units, at least the "
                         "10 N the bands need (default: 10 N by 10 N, the bands "
                         "alone)")
    ap.add_argument("-n", "--steps", type=number(int, least=1), default=10, metavar="N",
                    help="spacings across each band (default: 10)")
    ap.add_argument("--plot", action="store_true", help="show the cloud, coloured by marker")
    args = ap.parse_args()
    span = 2 * args.steps * sum(SPACINGS)        # the bands from both walls
    Lx, Ly = (span, span) if args.size is None else args.size
    if min(Lx, Ly) < span - TOL:
        ap.error(f"the cavity must be at least {span:g} by {span:g}, the width "
                 f"of the three bands from both walls at --steps {args.steps}")

    stem, suffix = output_stem(args.output, default=".node")

    for_rings = args.distribution == "rings"
    lv = levels(args.steps, for_rings)
    pts = rings(Lx, Ly, lv) if for_rings else grid(Lx, Ly, lv)
    pts = np.round(pts, 12) + 0.0            # 0.1 + 0.2 style noise, and no -0.0
    m = markers(pts, Lx, Ly)

    if suffix == ".points":
        write_points(stem, pts)                  # the format has no comments or markers
    else:
        write_node(stem, pts, m,
                   f"produced by cavity_refined.py --distribution {args.distribution} "
                   f"--size {Lx:g} {Ly:g} --steps {args.steps}")

    counts = np.bincount(m, minlength=6)
    written = "standard output" if stem == "-" else stem + suffix
    print(f"{written}: {len(pts)} nodes, "
          f"{counts[0]} interior, S {counts[1]} E {counts[2]} N {counts[3]} "
          f"W {counts[4]}, {counts[5]} corners", file=sys.stderr)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter(pts[:, 0], pts[:, 1], c=m, s=4, **MARKER_STYLE)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()
