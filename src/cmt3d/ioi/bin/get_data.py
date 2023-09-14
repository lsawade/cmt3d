"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.07 9.00

"""

from sys import exit, argv
from ..functions.utils import downloaddir
from ..functions.get_data import get_data

def bin():
    """

    Usage:

        gcmt3d-get-data eventfile input.yml

    This script calls a python function that takes in an eventfile and
    an input.yml, and creates a download directory and downloads the data
    into the directory.

    """

    # Get args or print usage statement
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventfile, inputfile = argv[1:]

    # Create download directory
    outdir, _, _ = downloaddir(inputfile, eventfile)

    # Download the data
    get_data(outdir)
