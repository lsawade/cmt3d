"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import exit, argv
from ..functions.opt import check_done


def bin():
    """

    Exectuable that checks whether an iteration should be added.

    Usage:
        ioi-check-done outdir

    where:
        outdir   - event directory

    """

    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    else:
        outdir = argv[1]

    # Clearlog
    check_done(outdir)
