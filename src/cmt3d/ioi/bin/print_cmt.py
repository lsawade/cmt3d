"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import argv, exit
from ..functions.model import get_cmt


def bin():
    """

    Usage:
    ------

        gcmt3d-cmt eventdir [it [ls]]

    where:
        eventdir   - directory containing the optimization parameters
        it       - iteration number
        ls       - linesearch number

    This script calls a python function that if none of the other options are
    given prints the final cmt solution otherwise a specified cmt.

    """

    # Get args or print usage statement
    if (len(argv) <= 1) or (len(argv) > 4):
        print(bin.__doc__)
        exit()
    elif len(argv) == 2:
        outdir = argv[1]
        it, ls = None, 0
    elif len(argv) == 3:
        outdir = argv[1]
        it, ls = int(argv[2]), 0
    elif len(argv) == 4:
        outdir = argv[1]
        it, ls = int(argv[2]), int(argv[3])

    # Print cmtsolution
    print(get_cmt(outdir, it=it, ls=ls))
