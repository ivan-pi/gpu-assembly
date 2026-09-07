"""The boundary markers of the node files: the convention the generators
share, numbered counter-clockwise from the bottom wall, then the corners,
then a hole in the interior. A node file carries one per node, zero for
an interior node and any nonzero value for a boundary one; which nonzero
value means what is ours to fix, and this is where it is fixed. A
generator uses the ones its geometry has: a periodic side has no wall
and so no marker, only a cavity has corners, and only a cylinder case a
hole.
"""

INTERIOR, SOUTH, EAST, NORTH, WEST, CORNER, HOLE = 0, 1, 2, 3, 4, 5, 6

# Marker colours for --plot: fixed here so that a marker means the same
# colour whichever generator drew the cloud.
MARKER_STYLE = dict(cmap="tab10", vmin=0, vmax=9)
