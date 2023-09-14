"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

from sys import argv, exit
from ..functions.events import create_event_status_dir


def bin():
    """
    
    Usage: 

        gcmt3d-create-eventdir <path/to/eventdir> <path/to/input.yml>

    This script calls a python function that creates an event inversion overview 
    directory. The structure is as follows

    event_status/
    |---- label/
        |---- DOWNLOADED/
        |---- STATUS/
        |---- EVENTS_INIT/
        |---- EVENTS_FINAL/
        |---- input.yml

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
    create_event_status_dir(eventdir, inputfile)
