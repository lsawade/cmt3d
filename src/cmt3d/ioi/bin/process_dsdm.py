"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from sys import argv, exit
from ..functions.processing import process_dsdm


def bin():
    """

    Usage:

        gcmt3d-process-dsdm eventdir param

    This script calls a python function that process the frechet derivatives and
    saves them in the frec/ folder

    IMPORTANT FOR PARALLEL COMPUTATION
    ----------------------------------
        Before execution, make sure that you

        export OMP_NUM_THREADS=1

        This is sooo important for parallel processing
        If you don't set this numpy, mkl, etc. will try to use threads
        for processing, but you do not want that, because you want to 
        distribute work to the different cores manually. If this is not 
        set, the different cores will fight for threads!!!!
        
        You cannot prepend this to the executable presumably because the entry 
        point preloads numpy which again causes trouble.
    
    """

    # Get args or print usage statement 
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        outdir, nm = argv[1:]

    # Run the initializer
    process_dsdm(outdir, nm)
