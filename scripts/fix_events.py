# %%
import os
import cmt3d
import cmt3d.ioi as ioi
from cmt3d.ioi.M0 import fix_synthetics

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split(container, count):
    """
    Simple function splitting a container into equal length chunks.
    Order is not preserved but this is potentially an advantage depending on
    the use case.
    """
    return [container[_i::count] for _i in range(count)]


# %%

if rank == 0:

    dbdir = "/lustre/orion/geo111/scratch/lsawade/gcmt/nnodes"
    outdirs = os.listdir(dbdir)
    outdirs.sort()

    outdirs = outdirs[4750:5700]

    N = 0
    events = []
    for _i, _eventid in enumerate(outdirs):

        # Get absolute directory
        od = os.path.join(dbdir, _eventid)

        # Check if status exists. If not, skip.
        if not os.path.exists(os.path.join(od, "STATUS.txt")):
            continue

        # Only read event if you status is finished
        status = ioi.read_status(od)
        if "FINISHED" not in status:
            continue

        N += 1
        events.append(od)
        print(_eventid)

        chunks = split(events, size)

else:
    chunks = None

chunks = comm.scatter(chunks, root=0)


for outdir in chunks:
    fix_synthetics(outdir, label='GCMT3D')
