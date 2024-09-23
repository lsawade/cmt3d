import cmt3d.utils as utils
from math import isclose
from pprint import pprint
from collections import OrderedDict


def setup_mpi_groups(mpisetup: dict, comm=None):
    """Splits the MPI communicator into subcommunicators based on the setup dictionary."""

    # If comm is not provided, get it using mpi4py
    if comm is None:
        comm, rank, size = utils.get_comm_rank_size()
    else:
        rank = comm.Get_rank()
        size = comm.Get_size()

    if rank == 0:
        # Print the setup
        absolute_cores = 0
        ratio_check = 0

        # Intilize the dictionaries to divide the cores
        ratio_dict = {}
        absolute_dict = {}

        for key, setupdict in mpisetup.items():
            if "ratio" in setupdict:
                ratio_check += setupdict["ratio"]
                ratio_dict[key] = setupdict["ratio"]
            elif "size" in setupdict:
                absolute_cores += setupdict["size"]
                absolute_dict[key] = setupdict["size"]
            else:
                print(key, setupdict)
                raise ValueError("Invalid setup dictionary")

        # Checks making sure the ratios add up to 1.0
        if not isclose(ratio_check, 1.0):
            raise ValueError("Ratios do not add up to 1.0. ratio_check = ", ratio_check)

        # Checks making sure the absolute cores add up to the total number of cores
        if absolute_cores > size:
            raise ValueError("Absolute cores exceed the total number of cores")

        # Check whether the absolute cores and number of ratios add up to the total number of cores
        if (absolute_cores + len(ratio_dict)) > size:
            raise ValueError(
                "Absolute cores and ratio cores do not add up to the total number of cores"
            )

        # Get number of cores used for ratios
        ratio_cores = size - absolute_cores

        finald = OrderedDict()

        # Calculate the number of cores for each ratio
        counter = 0
        for key, setupdict in mpisetup.items():
            # Get the number of cores from ratio
            if "ratio" in setupdict:
                cores = int(setupdict["ratio"] * ratio_cores)

            # Get the number of cores from absolute size
            elif "size" in setupdict:
                cores = setupdict["size"]

            # Compute subrange
            subrange = list(range(counter, counter + cores))

            # Add the subrange to the final dictionary
            finald[key] = subrange

            # Increment core counter
            counter += cores

        print(finald, flush=True)
        subname = []
        subrange = []

        for _key, _subrange in finald.items():
            print(_key, _subrange, flush=True)
            for _ in _subrange:
                subname.append(_key)
                subrange.append(_subrange)

        print(subname, subrange, flush=True)

        # Add extra entries to make sure the subname and subrange are the same size
        if len(subname) < size:
            subname += [None] * (size - len(subname))
            subrange += [None] * (size - len(subrange))
    else:
        subname = None
        subrange = None

    # Broadcast the subname and subrange to all ranks
    subname = comm.scatter(subname, root=0)
    subrange = comm.scatter(subrange, root=0)

    subgroup = comm.Get_group().Incl(subrange)
    subcomm = comm.Create(subgroup)

    return subname, subcomm


class CMT3D(object):
    def __init__(self, cmtdir, comm=None):
        """MPI Inversion Class taking care MPI communication and data handling."""

        self.cmtdir = cmtdir

        # If comm is not provided, get it using mpi4py
        if comm is None:
            self.comm, self.rank, self.size = utils.get_comm_rank_size()

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        if self.rank == 0:
            print("CMT3D: ", self.cmtdir)
            print(f"---Rank/Size: {self.subrank}/{self.subsize}", flush=True)

        if self.rank in self.subrange:
            self.subcomm.Barrier()

    def setup_subcomms(self):
        if self.rank == 0:
            print("Setting up subcomms")

            # Define the ratios of how many processors should be used for what
            self.ratios = [0.5, 0.25, 0.25]
            self.names = ["A", "B", "C"]

            # Dynamically divide the number of processors
            nprocs_per_sub = [int(r * self.size) for r in self.ratios]
            procs_per_sub = [list(range(0, n)) for n in nprocs_per_sub]

            for i, procs in enumerate(procs_per_sub):
                pass

        # Create the subgroups
        self.subgroups = [comm.Get_group().Incl(subrange) for subrange in procs_per_sub]

        # Create the subcomms
        self.subcomms = [comm.Create(subgroup) for subgroup in self.subgroups]

        # Get the ranks and sizes of the subcomms
        self.subranks = [subcomm.Get_rank() for subcomm in self.subcomms]
        self.subsizes = [subcomm.Get_size() for subcomm in self.subcomms]

        # Print the results
        for i, subcomm in enumerate(self.subcomms):
            print(f"Subcomm {self.names[i]}: {self.subranks[i]}/{self.subsizes[i]}")

        # Barrier
        comm.Barrier()

        # self.subrange = range(0, self.size, 2)

        # if self.rank in self.subrange:
        #     self.subgroup = comm.Get_group().Incl(self.subrange)
        #     self.subcomm = comm.Create(self.subgroup)
        #     self.subrank = self.subcomm.Get_rank()
        #     self.subsize = self.subcomm.Get_size()
        #     self.subcomm.Barrier()


if __name__ == "__main__":
    # General setup for MPI process
    mpisetup = OrderedDict(
        A=dict(ratio=0.5),
        B=dict(size=1),
        C=dict(size=1),
        D=dict(size=2),
        E=dict(ratio=0.5),
    )

    subname, subcomm = setup_mpi_groups(mpisetup=mpisetup)

    comm, rank, size = utils.get_comm_rank_size()
    subrank = subcomm.Get_rank()
    subsize = subcomm.Get_size()

    print(f"R/S: {rank+1}/{size} -> {subname} SR/SS: {subrank+1}/{subsize}", flush=True)
