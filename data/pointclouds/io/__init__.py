"""Readers and writers for the files of docs/file_formats.md. A reader
raises FormatError on a file that does not follow its format and logs
what is worth a note; a writer takes a stem and appends its own
extension."""

from .readers import (
    FormatError,
    plural,
    read_graph,
    read_node,
    read_nodes,
    read_ordering,
    read_points,
    stencils_to_csr,
)
from .writers import (
    open_out,
    write_graph,
    write_node,
    write_nodes,
    write_ordering,
    write_points,
)

__all__ = [
    "FormatError",
    "open_out",
    "plural",
    "read_graph",
    "read_node",
    "read_nodes",
    "read_ordering",
    "read_points",
    "stencils_to_csr",
    "write_graph",
    "write_node",
    "write_nodes",
    "write_ordering",
    "write_points",
]
