"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.06 16.00

"""

from sys import argv, exit
from os import path, listdir
from numpy import load, array2string


def bin():
    """

    Usage: 

        gcmt3d-print-model eventdir [anything]

    This script calls a python function that prints the models. If any 
    command line arguments are included the function includes the linesearches.

    """

    # Get args or print usage statement
    if (len(argv) < 2) or (len(argv) > 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        eventdir = path.abspath(argv[1])
        lsflag = True if len(argv) == 3 else False

    # Print costs
    modldir = path.join(eventdir, 'modl')

    for file in sorted(listdir(modldir)):

        # Combine the filename
        filename = path.join(modldir, file)

        # Get iteration and step from filename
        _, it, ls = file.split("_")
        it = int(it[2:])
        ls = int(ls[2:-4])

        # Check filename for iteration and linesearch number
        if lsflag:
            print(f"Iter/Step: {it:0>5}/{ls:0>5} -> Model: {array2string(load(filename), max_line_width=1e10)}")
        else:
            if ls == 0:
                print(f"Iter: {it:0>5} -> Model: {array2string(load(filename), max_line_width=1e10)}")
