"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.02.28 16.50

"""

from sys import exit, argv
from ..functions.linesearch import check_optvals


def bin():
    """
    Exectuable that checks teh optimization parameters on whether further iteration
    or linesearch is necessary.

    Usage:
        ioi-check-optvals <optdir> <it> <ls>
                    
    where:
        optdir   - directory containing the optimization parameters
        it       - iteration number
        ls       - linesearch number

    """

    if len(argv) != 1+3:
        print("Note enough or too few input parameters.")
        print(bin.__doc__)
        exit()

    # Get command line arguments
    optdir, it, ls, nls_max = argv[1:4]

    # Clearlog
    check_optvals(optdir, statdir, costdir, it, ls, nls_max)
