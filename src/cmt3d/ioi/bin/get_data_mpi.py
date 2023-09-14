"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.07 9.00

"""

from mpi4py import MPI
from sys import exit, argv
from ..functions.get_data_mpi import get_data_mpi
from ..functions.events import check_events_todownload


def bin():
    """

    Usage:

        mpiexec -n <procs> gcmt3d-get-data-mpi input.yml

    This script calls a python function that takes in an eventfile and
    an input.yml, and creates a download directory and downloads the data
    into the directory.

    """

    # Get MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get args or print usage statement
    if (len(argv) < 2) or (len(argv) > 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        if rank == 0:
            print(bin.__doc__)
        exit()
    else:

        inputfile = argv[1]

        if len(argv) == 3:
            max_downloads = int(argv[2])
        else:
            max_downloads = int(1e6)

    if rank == 0:
        # Get files to dowload
        eventfiles = check_events_todownload(inputfile)[:max_downloads]

        # Number of events to download
        ND = len(eventfiles)

        # # If the number of events is smaller than the number of available 
        # # workers. Stop
        # if size > ND:
        #     errorflag = True
        # else:
        #     errorflag = False
    else:
        eventfiles = None
        # errorflag = False

    # # Broadcast the error flag to all workers to stop
    # errorflag = comm.bcast(errorflag, root=0)

    # # On worker 0 raise Error, exit on the rest.
    # if errorflag:
    #     if rank == 0:
    #         raise ValueError(f'You asked for {size} workers to download {ND} events...')
    #     else:
    #         exit()

    # Feed into function
    get_data_mpi((eventfiles, inputfile))
