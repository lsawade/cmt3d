"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.06 16.00

"""

from sys import argv, exit
from os import path, listdir
from numpy import load
from ..functions.descent import descent


def bin():
    """

    Usage: 

        gcmt3d-compute-descent eventdir

    This script calls a python function that prints computes the descent direction.

    """

    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir = path.abspath(argv[1])

    descent(eventdir)