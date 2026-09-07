"""What the point-cloud generators in data/gen share: the formats they
write and the command-line conventions they follow.

The generators are standalone scripts, run by hand from anywhere, and
Python puts the directory of the script it runs first on the import path,
so they find this package as a sibling without being installed.
"""

from .cli import number, output_stem
from .formats import (CORNER, EAST, INTERIOR, MARKER_STYLE, NORTH, SOUTH,
                      WEST, open_out, write_graph, write_node, write_points)

__all__ = ["INTERIOR", "SOUTH", "EAST", "NORTH", "WEST", "CORNER",
           "MARKER_STYLE", "number", "open_out", "output_stem",
           "write_graph", "write_node", "write_points"]
