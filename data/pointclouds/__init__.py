"""What the Python under data/ shares, laid out like the C++ library:

    pointclouds.io          readers and writers for the files of docs/file_formats.md
    pointclouds.nodeset     NodeSet, a cloud with markers and a box, and the tiling
    pointclouds.generators  the clouds the generators make, as children of NodeSet
    pointclouds.stencils    stencil selection, and the --graph option that picks it
    pointclouds.poisson     a Poisson disk sampler of a rectangle, periodic or not
    pointclouds.cli         the command-line conventions of the scripts

The generators in data/gen and the tools in data/tools import it, so it
has to be installed; from the repository root, `pip install -e data`
does so in place.
"""
