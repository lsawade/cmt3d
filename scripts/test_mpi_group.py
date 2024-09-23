from mpi4py import MPI
import typing as tp

comm = MPI.COMM_WORLD

# Use dill instead of pickle for transport
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    print("Hello from rank 0")
    print("Size is ", size)

comm.Barrier()


def chunkfunc(sequence: tp.Sequence, n: int):
    n = max(1, n)
    return [sequence[i : i + n] for i in range(0, len(sequence), n)]


ranks = chunkfunc(range(size), 4)

for _subranks in ranks:
    if rank in _subranks:
        subgroup = comm.Get_group().Incl(_subranks)
        subcomm = comm.Create(subgroup)
        subrank = subcomm.Get_rank()
        subsize = subcomm.Get_size()


print(f"Rank{rank:02d}/{size:02d} -> {subrank:02d}/{subsize:02d}")

comm.barrier()
