#!/usr/bin/env python3
"""Point clouds for the lid-driven cavity [0, Lx] x [0, Ly], refined towards
the walls in three levels, written as a node file with boundary markers.

    python3 cavity_refined.py cavity_refined.node
    python3 cavity_refined.py --layout grid cavity_grid.node
    python3 cavity_refined.py --size 1 1.3 --plot cavity_tall.node

Two layouts share the wall spacings h = 0.01, 0.015, 0.025:

  rings  Concentric rectangles inset from the walls, each with its points
         h apart. The layout of Lin, Wu and Zhang (2019), "A mesh-free
         radial basis function-based semi-Lagrangian lattice Boltzmann
         method for incompressible flows", Int. J. Numer. Meth. Fluids 91,
         198-211. Each ring is one spacing of its own level inside the
         previous ring, so the first ring of a level sits one spacing of
         the previous level inside that level's last ring. The rings end
         in the centre point of a square; a rectangle keeps going at the
         coarsest spacing until the rings degenerate to a segment.
  grid   The tensor product of 1-d coordinates graded the same way:
         h = 0.01 on [0, 0.1], 0.015 on [0.1, 0.25], 0.025 on [0.25, 0.5],
         mirrored from the far wall, with the middle at 0.025 when the
         side is longer than 1.

On the unit square the two give 6169 and 3721 nodes. Both list the wall
nodes first, then the interior from the wall inwards (rings) or row by
row (grid).

Boundary markers (docs/file_formats.md, node file):

    0  interior
    1  south wall, y = 0
    2  east wall,  x = Lx
    3  north wall, y = Ly, the lid
    4  west wall,  x = 0
    5  corner, where two walls meet

The walls are numbered counter-clockwise from the bottom. The corners get
their own marker because a corner node belongs to two walls with, in the
cavity, different boundary data: the lid velocity meets the wall's no-slip.
Whoever assembles the boundary conditions decides what a corner gets;
`NodeSet::indices_with(5)` picks them out.

The output name selects the format: `.node` writes Triangle's node file
with the markers, `.points` writes the points file, which has none.
"""

import argparse
import sys

import numpy as np

# Rings: (spacing, number of rings) from the wall inwards. Grid:
# (spacing, number of steps) of the 1-d coordinate from the wall inwards.
# Both tables reach 0.5 from the wall, the centre of the unit square.
RING_LEVELS = [(0.01, 11), (0.015, 11), (0.025, 10)]
GRID_LEVELS = [(0.01, 10), (0.015, 10), (0.025, 10)]

INTERIOR, SOUTH, EAST, NORTH, WEST, CORNER = 0, 1, 2, 3, 4, 5

TOL = 1e-9


def steps(length, h, what):
    n = round(length / h)
    if abs(length / h - n) > 1e-6:
        sys.exit(f"{what}: {length:g} is not a multiple of the spacing {h:g}")
    return n


def ring(x0, x1, y0, y1, h):
    """Points h apart on the boundary of [x0, x1] x [y0, y1], counter-
    clockwise from (x0, y0); a segment or a point when a side is zero."""
    nx = steps(x1 - x0, h, f"ring at inset {x0:g}, width")
    ny = steps(y1 - y0, h, f"ring at inset {y0:g}, height")
    if nx == 0 and ny == 0:
        return np.array([[x0, y0]])
    if nx == 0:
        return np.column_stack((np.full(ny + 1, x0), y0 + h * np.arange(ny + 1)))
    if ny == 0:
        return np.column_stack((x0 + h * np.arange(nx + 1), np.full(nx + 1, y0)))
    ix, iy = h * np.arange(nx), h * np.arange(ny)
    south = np.column_stack((x0 + ix, np.full(nx, y0)))
    east = np.column_stack((np.full(ny, x1), y0 + iy))
    north = np.column_stack((x1 - ix, np.full(nx, y1)))
    west = np.column_stack((np.full(ny, x0), y1 - iy))
    return np.vstack((south, east, north, west))


def rings_layout(Lx, Ly, levels):
    spacings = [h for h, count in levels for _ in range(count)]
    pts = []
    r = 0.0
    k = 0
    while Lx - 2 * r > -TOL and Ly - 2 * r > -TOL:
        h = spacings[min(k, len(spacings) - 1)]
        pts.append(ring(r, Lx - r, r, Ly - r, h))
        r += h
        k += 1
    if k < len(spacings):
        sys.exit(f"cavity {Lx:g} x {Ly:g} is too small for the refinement levels, "
                 f"which reach 0.5 from the wall")
    return np.vstack(pts)


def graded_coordinate(L, levels):
    """From the wall at 0 to the far wall at L: the levels inwards from
    both walls, and the middle at the coarsest spacing."""
    z = [0.0]
    for h, count in levels:
        z.extend(z[-1] + h * np.arange(1, count + 1))
    z = np.array(z)
    h = levels[-1][0]
    n = steps(L - 2 * z[-1], h, f"side {L:g} minus the graded ends {2 * z[-1]:g}")
    if n < 0:
        sys.exit(f"side {L:g} is shorter than the two graded ends, {2 * z[-1]:g}")
    middle = z[-1] + h * np.arange(1, n + 1)   # ends at L - z[-1], or empty
    return np.concatenate((z, middle, L - z[-2::-1]))


def grid_layout(Lx, Ly, levels):
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


def write_node(fname, pts, m):
    with open(fname, "w") as f:
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")     # repr: shortest round-trip text


def write_points(fname, pts):
    with open(fname, "w") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("output", help="output file, .node (with markers) or .points")
    ap.add_argument("--layout", choices=("rings", "grid"), default="rings")
    ap.add_argument("--size", nargs=2, type=float, default=(1.0, 1.0),
                    metavar=("LX", "LY"), help="side lengths (default: the unit square)")
    ap.add_argument("--plot", action="store_true", help="show the cloud, coloured by marker")
    args = ap.parse_args()
    Lx, Ly = args.size

    if args.layout == "rings":
        pts = rings_layout(Lx, Ly, RING_LEVELS)
    else:
        pts = grid_layout(Lx, Ly, GRID_LEVELS)
    pts = np.round(pts, 12) + 0.0            # 0.1 + 0.2 style noise, and no -0.0
    m = markers(pts, Lx, Ly)

    if args.output.endswith(".points"):
        write_points(args.output, pts)
    else:
        write_node(args.output, pts, m)

    counts = np.bincount(m, minlength=6)
    print(f"{args.output}: {len(pts)} nodes, {counts[0]} interior, "
          f"S {counts[1]} E {counts[2]} N {counts[3]} W {counts[4]}, "
          f"{counts[5]} corners")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.scatter(pts[:, 0], pts[:, 1], c=m, s=4, cmap="tab10", vmin=0, vmax=9)
        plt.axis("equal")
        plt.show()


if __name__ == "__main__":
    main()
