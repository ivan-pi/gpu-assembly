#!/usr/bin/env python3
"""Node files for the two domains of Wright and Fornberg (2006), "Scattered
node compact finite difference-type formulas generated from radial basis
functions", J. Comput. Phys. 212, 99-123, doi:10.1016/j.jcp.2005.05.030.

    python3 wright2006_nodes.py disk     ../wright_disk_200.node
    python3 wright2006_nodes.py cylinder ../wright_cylinder_892.node

The node coordinates in the paper are published as figures only, so they
were read off the figures with WebPlotDigitizer 4 into the two CSV files
next to this script. Digitizing quantizes a coordinate to a pixel, about
0.0026 of the plot width here, and the boundary nodes come out scattered
about the boundary instead of on it. This script turns the two CSVs into
node files with the nodes on the boundary exactly, so that the outward
normal of a boundary node is exact too.

  disk      The unit disk, centre at the origin: 200 nodes, of which 56
            lie on the circle r = 1. Quasi-uniform, mean spacing 0.126,
            and with no symmetry to exploit, so the interior nodes are the
            digitized ones rounded to six decimals and only the 56
            boundary nodes are moved, radially out to r = 1. This is the
            paper's 200-node unstructured discretization of the disk.

  cylinder  The square [-1, 1] x [-1, 1] with a circular hole of radius
            0.4 at the origin. Cartesian nodes at spacing h = 1/15 fill
            the square, except in the box [-0.6, 0.6] x [-0.6, 0.6] around
            the hole, where 220 scattered nodes (mean spacing 0.060) take
            over: 40 on the hole and 180 between the hole and the box.
            Only those 220 are digitized; the Cartesian nodes are
            regenerated here, which puts them on the outer boundary
            exactly. 892 nodes in all.

            The 220 scattered nodes are symmetric under the eight
            reflections and rotations of the square to within one pixel,
            so each node is replaced by the average of its orbit, which
            makes the cloud exactly symmetric and averages the digitizing
            error of up to eight copies of the node. The 40 nodes of the
            hole are then projected radially onto r = 0.4.

Read off the figures rather than stated in the paper, and so the least
certain of the numbers above: the hole radius (the digitized nodes give
0.39988 +- 0.00032), the box the scattered nodes occupy, and the Cartesian
spacing. --hole, --box and --steps override them.

Boundary markers (docs/file_formats.md, node file), sharing the numbering
of cavity_refined.py as far as the four walls go:

    0  interior
    1  south wall, y = -1
    2  east wall,  x = 1
    3  north wall, y = 1
    4  west wall,  x = -1
    5  corner, where two walls meet
    6  the hole, r = 0.4  (the disk's boundary r = 1 also gets 1)

Boundary nodes come first in the file, walls before the hole, then the
interior nodes.
"""

import argparse
import csv
import sys

import numpy as np

INTERIOR, SOUTH, EAST, NORTH, WEST, CORNER, HOLE = 0, 1, 2, 3, 4, 5, 6

DIGITIZED = 6      # decimals kept of a digitized coordinate; a pixel is 0.0026
GENERATED = 12     # decimals kept of a coordinate we compute, to shed 0.1 + 0.2 noise

# The eight symmetries of the square, as matrices acting on a row vector.
SQUARE_GROUP = [np.array(m, dtype=float) for m in
                (((1, 0), (0, 1)), ((-1, 0), (0, 1)), ((1, 0), (0, -1)), ((-1, 0), (0, -1)),
                 ((0, 1), (1, 0)), ((0, -1), (1, 0)), ((0, 1), (-1, 0)), ((0, -1), (-1, 0)))]


def read_digitized(fname, column):
    """The named pair of columns of a WebPlotDigitizer CSV: a row of dataset
    names, a row of `X,Y` headings, then the values, with the shorter
    datasets padded with empty fields."""
    with open(fname) as f:
        rows = list(csv.reader(f))
    names, pts = rows[0], []
    if column not in names:
        sys.exit(f"{fname}: no dataset '{column}'; it has {[n for n in names if n]}")
    c = names.index(column)
    for row in rows[2:]:
        if len(row) > c + 1 and row[c].strip() and row[c + 1].strip():
            pts.append((float(row[c]), float(row[c + 1])))
    return np.array(pts)


def symmetrize(pts, tol):
    """The point cloud made symmetric under the eight symmetries of the
    square, by replacing every node with the average of its orbit.

    Node `i` maps under symmetry `g` to some node `j` of the same cloud, up
    to the digitizing error; `g` carried back over that `j` is another
    reading of node `i`, and the eight readings are averaged. A node on a
    symmetry axis is its own image under the symmetries that fix the axis,
    so it is read fewer than eight times but averaged the same way.

    `tol` is how far a node may sit from the image it matches. A symmetry
    that leaves some node further away than that is not a symmetry of the
    cloud, and the caller is told rather than handed a silent average over
    an incomplete orbit."""
    out, worst = np.empty_like(pts), 0.0
    for i, p in enumerate(pts):
        readings = []
        for g in SQUARE_GROUP:
            image = p @ g
            j = np.argmin(np.hypot(pts[:, 0] - image[0], pts[:, 1] - image[1]))
            missed = float(np.hypot(*(pts[j] - image)))
            if missed > tol:
                sys.exit(f"node {i} at {p} misses its image under a symmetry of "
                         f"the square by {missed:.5f}, more than --pixel {tol}")
            worst = max(worst, missed)
            readings.append(pts[j] @ g.T)          # the image carried back
        out[i] = np.mean(readings, axis=0)
    print(f"patch: symmetric to within {worst:.5f}")
    return out


def to_circle(pts, radius):
    """Radially onto the circle of that radius about the origin. The angle of
    a node is left alone: it carries whatever the node generator put there,
    which the digitized nodes show is not a uniform spacing."""
    return pts * (radius / np.hypot(pts[:, 0], pts[:, 1]))[:, None]


def counter_clockwise(pts):
    return pts[np.argsort(np.arctan2(pts[:, 1], pts[:, 0]))]


def tidy(pts, decimals):
    return np.round(pts, decimals) + 0.0          # + 0.0: no -0.0 in the file


def disk(args):
    """The unit disk. The digitized `All_Disk` holds all 200 nodes and
    `Inner_Disk` the 144 interior ones; the 56 on the circle are the ones
    the radius tells apart, which we check against that second dataset."""
    pts = tidy(read_digitized(args.csv, "All_Disk"), DIGITIZED)
    interior = read_digitized(args.csv, "Inner_Disk")
    on_circle = np.hypot(pts[:, 0], pts[:, 1]) > 0.5 * (
        np.hypot(*interior.T).max() + args.radius)      # midway between the two
    if on_circle.sum() != len(pts) - len(interior):
        sys.exit(f"{args.csv}: {on_circle.sum()} nodes near r = {args.radius}, "
                 f"but Inner_Disk leaves {len(pts) - len(interior)} outside itself")
    ring = counter_clockwise(pts[on_circle])
    circle = to_circle(ring, args.radius)
    report("disk", moved=np.hypot(*(circle - ring).T))
    return (np.vstack((circle, pts[~on_circle])),
            np.concatenate((np.full(len(circle), 1),
                            np.full(len(pts) - len(circle), INTERIOR))))


def cylinder(args):
    """The square with a circular hole: the digitized scattered patch, made
    symmetric and projected onto the hole, inside a regenerated Cartesian
    grid with the box around the hole cut out of it."""
    tol = 1e-9
    patch = read_digitized(args.csv, "Irregular")
    clean = tidy(symmetrize(patch, tol=args.pixel), DIGITIZED)
    report("patch", moved=np.hypot(*(clean - patch).T))

    r = np.hypot(clean[:, 0], clean[:, 1])
    order = np.argsort(r)                        # the nodes on the hole are the
    ring = order[:np.argmax(np.diff(r[order])) + 1]   # ones inside the widest gap
    measured = r[ring].mean()
    print(f"hole: {len(ring)} nodes at r = {measured:.5f} +- {r[ring].std():.5f}")
    if abs(measured - args.hole) > args.pixel:
        sys.exit(f"the nodes on the hole sit at r = {measured:.5f}, further than a "
                 f"pixel from --hole {args.hole}")
    on_hole = np.zeros(len(clean), dtype=bool)
    on_hole[ring] = True
    around = counter_clockwise(clean[on_hole])
    hole = to_circle(around, args.hole)
    report("hole", moved=np.hypot(*(hole - around).T))
    patch_interior = clean[~on_hole]

    z = tidy(np.linspace(-1.0, 1.0, 2 * args.steps + 1), GENERATED)
    grid = np.array([(x, y) for y in z for x in z])
    outside_box = (np.abs(grid[:, 0]) > args.box - tol) | (np.abs(grid[:, 1]) > args.box - tol)
    grid = grid[outside_box]                     # the box is where the patch goes
    on_wall = walls(grid, tol) != INTERIOR

    wall = counter_clockwise(grid[on_wall])
    pts = np.vstack((wall, hole, grid[~on_wall], patch_interior))
    marker = np.concatenate((walls(wall, tol),
                             np.full(len(hole), HOLE),
                             np.full(len(grid) - len(wall) + len(patch_interior), INTERIOR)))
    return pts, marker


def walls(pts, tol):
    x, y = pts[:, 0], pts[:, 1]
    s, e, n, w = y < -1 + tol, x > 1 - tol, y > 1 - tol, x < -1 + tol
    m = np.full(len(pts), INTERIOR)
    m[s], m[e], m[n], m[w] = SOUTH, EAST, NORTH, WEST
    m[(s | n) & (e | w)] = CORNER
    return m


def report(what, moved):
    print(f"{what}: {len(moved)} nodes moved by at most {moved.max():.5f}, "
          f"{moved.mean():.5f} on average")


def write_node(fname, pts, marker, provenance):
    with open(fname, "w") as f:
        f.write(f"# {provenance}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), m) in enumerate(zip(pts.tolist(), marker.tolist())):
            f.write(f"{i} {x!r} {y!r} {m}\n")      # repr: shortest round-trip text


HELP = __doc__.split("\n\n")[0] + """

The nodes are read off the paper's figures, so this needs the CSV next to
the script; --csv points elsewhere. See the docstring at the top of the
script for the two domains, what is measured and what is assumed, and the
boundary markers."""


def main():
    ap = argparse.ArgumentParser(description=HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain", choices=("disk", "cylinder"),
                    help="the unit disk, or the square with a circular hole")
    ap.add_argument("output", help="output node file")
    ap.add_argument("--csv", help="digitized nodes (default: wright2006_<domain>.csv "
                    "next to this script)")
    ap.add_argument("--radius", type=float, default=1.0, metavar="R",
                    help="disk: the radius of the disk (default: 1)")
    ap.add_argument("--hole", type=float, default=0.4, metavar="R",
                    help="cylinder: the radius of the hole (default: 0.4)")
    ap.add_argument("--box", type=float, default=0.6, metavar="B",
                    help="cylinder: half-width of the box the scattered nodes fill, "
                    "cut out of the Cartesian grid (default: 0.6)")
    ap.add_argument("--steps", type=int, default=15, metavar="N",
                    help="cylinder: Cartesian spacings per unit, so the spacing is "
                    "1/N and a side of the square holds 2N + 1 nodes (default: 15)")
    ap.add_argument("--pixel", type=float, default=0.004, metavar="D",
                    help="cylinder: the digitizing error, so how far a node may sit "
                    "from the image it is averaged with, and from --hole (default: 0.004)")
    ap.add_argument("--plot", action="store_true", help="show the cloud, coloured by marker")
    args = ap.parse_args()
    if args.csv is None:
        from pathlib import Path
        args.csv = str(Path(__file__).with_name(f"wright2006_{args.domain}.csv"))

    if args.domain == "disk":
        pts, marker = disk(args)
        provenance = f"produced by wright2006_nodes.py disk --radius {args.radius:g}"
    else:
        pts, marker = cylinder(args)
        provenance = (f"produced by wright2006_nodes.py cylinder --hole {args.hole:g} "
                      f"--box {args.box:g} --steps {args.steps}")
    write_node(args.output, pts, marker,
               provenance + "; nodes of Wright and Fornberg (2006), "
               "doi:10.1016/j.jcp.2005.05.030, read off the paper's figures")

    counts = np.bincount(marker, minlength=7)
    if args.domain == "disk":
        tally = f"{counts[1]} on the circle"
    else:
        tally = (f"S {counts[1]} E {counts[2]} N {counts[3]} W {counts[4]}, "
                 f"{counts[5]} corners, {counts[6]} on the hole")
    print(f"{args.output}: {len(pts)} nodes, {counts[0]} interior, {tally}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter(pts[:, 0], pts[:, 1], c=marker, s=6, cmap="tab10", vmin=0, vmax=9)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()
