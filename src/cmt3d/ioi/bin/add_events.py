"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import argv, exit
from ..functions.events import add_events


def bin():
    """

    Usage:
    ------

        gcmt3d-add-events {<path/to/eventdir> | <path/to/event>} <path/to/input.yml>

    where:
        path/to/event(dir) - can point to a specific CMT solution or a directory
                             of CMTSOLUTIONs
        input.yml - is the gcmt3d input file.

    This script calls a python function that adds events to an existing event status
    directory. It checks whether events have already been added before,

    where path to 'event_status' and 'label' are found in the 'input.yml'.

    After initialization.

    """

    # Get args or print usage statement
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir, inputfile = argv[1:]

    # Run the initializer
    add_events(eventdir, inputfile)
