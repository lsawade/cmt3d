"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.02.28 16.50

"""

from sys import exit, argv
from ..functions.linesearch import linesearch


def bin():
    """

    Exectuable that checks and writes linesearch parameters for on linesearch
    iteration.

    Usage:
        gcmt3d-linesearch <eventdir>

    """
    
    # Get args or print usage statement 
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        outdir = argv[1]

    # Clearlog
    linesearch(outdir)
