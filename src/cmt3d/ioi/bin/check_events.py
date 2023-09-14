"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import argv, exit
from ..functions.events import check_events


def bin():
    """

    Usage:

        gcmt3d-check-events-todownload <path/to/input.yml>

    Note that the input.yml contains the grander event status directory. The 
    check-events-todownload will go through all of them to see whats missing.

    """
    # Get args or print usage statement
    if (len(argv) == 1) or (len(argv) > 4):
        print(bin.__doc__)
        exit()
    elif (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    elif len(argv) == 2:
        inputfile = argv[1]
        # Check event statuses
        check_events(inputfile)
    elif len(argv) == 3:
        inputfile, resetflag = argv[1:]
        check_events(inputfile, resetopt=resetflag)
    else:
        print(__doc__)
        exit()
