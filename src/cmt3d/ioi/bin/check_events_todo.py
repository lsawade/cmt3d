"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

import os
from sys import argv, exit
from ..functions.events import check_events_todo


def bin():

    """

    Usage:

        gcmt3d-check-events-todo <path/to/input.yml>

    This script calls a python function that adds events to an existing event status
    directory. It checks whether events have already been added before,

    where path to 'event_status' and 'label' are found in the 'input.yml'.

    After initialization.

    """

    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        inputfile = argv[1]

    # Run the initializer
    TODO = check_events_todo(inputfile)

    for _todo in TODO:
        print(os.path.basename(_todo))
