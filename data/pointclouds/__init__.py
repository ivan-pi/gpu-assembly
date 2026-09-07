"""What the Python under data/ shares, laid out like the C++ library:

    pointclouds.io       readers and writers for the files of docs/file_formats.md
    pointclouds.cli      the command-line conventions of the scripts
    pointclouds.stencils stencil selection for the generators, and the marker convention
    pointclouds.poisson  a Poisson disk sampler of a rectangle, periodic or not

The generators in data/gen and the tools in data/tools import it, so it
has to be installed; from the repository root, `pip install -e data`
does so in place.
"""
