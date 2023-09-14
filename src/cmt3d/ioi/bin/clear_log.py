"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import exit, argv
from ..functions.log import clear_log


def bin_clear_log():
    """

    Executable that clears the log in a given log directory.

    Usage:
        gcmt3d-clear-log optdir

    where:
        logdir   - directory containing the log(s)
    
    """

    if len(argv) != 1+1:
        print("Note enough or too few input parameters.")
        print(bin.__doc__)
        exit()

    # Get log dir from command line arguments
    optdirdir = argv[1]

    # Clearlog
    clear_log(optdir)
