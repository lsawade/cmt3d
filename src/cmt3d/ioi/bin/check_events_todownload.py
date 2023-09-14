"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

import os
from sys import argv, exit
from ..functions.events import check_events_todownload


def bin():
    """

    Usage:

        gcmt3d-check-events-todownload <path/to/input.yml>

    Note that the input.yml contains the grander event status directory. The 
    check-events-todownload will go through all of them to see whats missing.

    """

    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(__doc__)
        exit()
    else:
        inputfile = argv[1]

    # Run the initializer
    TODOWNLOAD = check_events_todownload(inputfile)

    for _todo in TODOWNLOAD:
        print(os.path.basename(_todo))
    