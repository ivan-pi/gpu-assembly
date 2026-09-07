"""What the Python under data/ shares: the formats it reads and writes
and the command-line conventions it follows. The generators in data/gen
write the files; the tools in data/tools read them, and one writes an
ordering.

The scripts are standalone, run by hand from anywhere, and not installed,
so each puts data/, the directory above its own, on the import path
before importing this package:

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""

from .cli import number, output_stem
from .formats import (CORNER, EAST, INTERIOR, MARKER_STYLE, NORTH, SOUTH,
                      WEST, open_out, write_graph, write_node, write_ordering,
                      write_points)
from .readers import (Report, plural, read_graph, read_node, read_nodes,
                      read_ordering, read_points, stencils_to_csr)

__all__ = ["INTERIOR", "SOUTH", "EAST", "NORTH", "WEST", "CORNER",
           "MARKER_STYLE", "Report", "number", "open_out", "output_stem",
           "plural", "read_graph", "read_node", "read_nodes", "read_ordering",
           "read_points", "stencils_to_csr", "write_graph", "write_node",
           "write_ordering", "write_points"]
