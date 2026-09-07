"""What the Python under data/ shares, laid out like the C++ library.

The scripts in data/tools import it, so it has to be installed; from
the repository root, ``pip install -e data`` does so in place.

.. autosummary::

   nodeset     NodeSet, a cloud with markers and a box, and the tiling
   generators  the clouds the generators make, as children of NodeSet
   stencils    stencil selection
   periodic    the periodic box: wrap, and the minimum image
   poisson     a Poisson disk sampler of a rectangle, periodic or not
   io          readers and writers for the files of docs/file_formats.md
   cli         the command-line conventions of the scripts
"""
