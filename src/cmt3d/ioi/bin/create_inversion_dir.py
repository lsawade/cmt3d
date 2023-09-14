"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import argv, exit
from ..functions.utils import create_forward_dirs


def bin():
    """

    Usage:

        gcmt3d-create-inv-dir /path/to/eventfile <path/to/input.yml>

    This script calls a python function that creates an event inversion overview 
    directory. The structure basic structure is as follows, details in
    ``optim_dir()``

    invdir/
    |---- opt/
    |---- modl/
    |     ...
    |---- hess/
    |---- input.yml

    """

    # Get args or print usage statement 
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventfile, inputfile = argv[1:]

    # Run the initializer
    create_forward_dirs(eventfile, inputfile)


