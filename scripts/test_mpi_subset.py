from mpi4py import MPI
from gf3d.mpi_subset import MPISubset
from gf3d.source import CMTSOLUTION


cmt_file = "/lustre/orion/geo111/scratch/lsawade/gcmt/nnodes/C201003151108A/meta/init_model.cmt"
subset = "/lustre/orion/geo111/scratch/lsawade/temp.h5"

cmt = CMTSOLUTION.read(cmt_file)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Subset: ", subset)
    print("CMT: ", cmt)

subrange = range(0, size, 2)
subgroup = comm.Get_group().Incl(subrange)
subcomm = comm.Create(subgroup)

if rank in subrange:
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    print(f"---Rank/Size: {subrank}/{subsize}", flush=True)

    MS = MPISubset(subset, comm=subcomm)
    MS.get_seismograms(cmt)

    subcomm.Barrier()
comm.Barrier()