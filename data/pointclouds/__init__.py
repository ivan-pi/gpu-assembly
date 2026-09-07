"""What the Python under data/ shares, laid out like the C++ library:

    pointclouds.io       readers and writers for the files of docs/file_formats.md
    pointclouds.markers  the boundary-marker convention of the node files
    pointclouds.cli      the command-line conventions of the scripts

The generators in data/gen and the tools in data/tools import it, so it
has to be installed; from the repository root, `pip install -e data`
does so in place.
"""
