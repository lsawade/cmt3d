from mpi4py import MPI
from .utils import downloaddir
from .get_data import get_data
from lwsspy.utils.io import read_yaml_file
import typing as tp


def split(container, count) -> tp.List[list]:
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


def get_data_mpi(args):

    # Get args
    eventfiles, inputfile = args

    # Get MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Pre read the yaml file and scatter across cores
    if rank == 0:
        inputparams = read_yaml_file(inputfile)
        eventlist = split(eventfiles, size)
    else:
        eventlist = None
        inputparams = None

    # Eventfiles
    eventlist = comm.scatter(eventlist, root=0)

    # Broadcast process dictionary
    inputparams = comm.bcast(inputparams, root=0)

    # Create downloaddir
    for eventfile in eventlist:

        # Create download directory
        outdir, _, _ = downloaddir(inputparams, eventfile)

        # Download data
        get_data(outdir)
