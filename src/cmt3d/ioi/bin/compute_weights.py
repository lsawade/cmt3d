"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.06 16.00

"""

from sys import argv, exit
from os import path, listdir
from numpy import load
from ..functions.weighting import compute_weights


def bin():
    """

    Usage: 

        gcmt3d-compute-weights eventdir

    This script calls a python function that prints computes the weights.

    """

    # Get args or print usage statement
    if (len(argv) != 2) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir = path.abspath(argv[1])

    compute_weights(eventdir)