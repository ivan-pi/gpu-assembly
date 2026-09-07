"""Readers and writers for the files of docs/file_formats.md. A reader
checks what it reads and records the problems in a Report; a writer takes
a stem and appends its own extension."""

from .readers import (Report, plural, read_graph, read_node, read_nodes,
                      read_ordering, read_points, stencils_to_csr)
from .writers import (open_out, write_graph, write_node, write_ordering,
                      write_points)

__all__ = ["Report", "open_out", "plural", "read_graph", "read_node",
           "read_nodes", "read_ordering", "read_points", "stencils_to_csr",
           "write_graph", "write_node", "write_ordering", "write_points"]
