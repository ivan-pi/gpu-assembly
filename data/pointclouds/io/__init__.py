"""Readers and writers for the files of docs/file_formats.md.

A reader raises FormatError on a file that does not follow its format and
logs what is worth a note; a writer takes a stem and appends its own
extension.
"""

from .readers import FormatError, plural, read_graph, read_nodes, read_ordering
from .writers import write_graph, write_node, write_ordering, write_points

__all__ = [
    "FormatError",
    "plural",
    "read_graph",
    "read_nodes",
    "read_ordering",
    "write_graph",
    "write_node",
    "write_ordering",
    "write_points",
]
