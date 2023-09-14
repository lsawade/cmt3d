"""

:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00

"""

import os
from sys import argv, exit


def bin():
    """

    Usage:

        gcmt3d-print-status <directory>

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
        directory = argv[1]

    # list files
    for _file in os.listdir(directory):

        # Path to file
        path = os.path.join(directory, _file)

        # Open file
        with open(path, 'r') as f:
            message = f.read()

        print(f"{_file + ':':.<30} {message}")
    
