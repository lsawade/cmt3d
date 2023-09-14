"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.06 16.00

"""

from sys import argv, exit
from os import path
from lwsspy.gcmt3d.ioi.functions.model import print_model_names


def bin():
    """

    Usage: 

        gcmt3d-print-model-names eventdir [anything]

    This script calls a python function that prints the model_names. If any 
    command line arguments are included the function includes the linesearches.

    """

    # Get args or print usage statement
    if (len(argv) < 2) or (len(argv) > 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir = path.abspath(argv[1])

    print_model_names(eventdir)
