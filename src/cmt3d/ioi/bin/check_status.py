"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import exit, argv
from ..functions.opt import check_status


def bin():
    """

    Exectuable that checks the status of the inversion after linesearch.

    Usage:
        gcmt3d-check-status <statusdir>

    where:
    statusdir  -  directory 'STATUS.txt' is written to

    """

    if len(argv) != 1+1:
        print("Note enough or too few input parameters.")
        print(__doc__)
        exit()

    # Get command line arguments
    statusdir = argv[1]

    # Check Status
    print(check_status(statusdir))
