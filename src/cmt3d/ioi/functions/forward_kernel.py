# External
import os
import sys
import dill
import time
import cmt3d
from copy import deepcopy
import numpy as np

# Internal
from .model import read_model, read_model_names, read_perturbation
from .kernel import write_dsdm_raw
from .forward import write_synt_raw
from .constants import Constants
from .log import get_iter, get_step
from .utils import cmt3d2gf3d
from gf3d.mpi_subset import MPISubset
import _pickle


def mpiabort_excepthook(type, value, traceback):

    if hasattr(traceback, "print_exc"):
        traceback.print_exc()

    print("", flush=True)
    time.sleep(1.0)
    import mpi4py.MPI
    mpi_comm = mpi4py.MPI.COMM_WORLD
    mpi_comm.Abort()


def forward_kernel(outdir):

    # Set abort hook on all ranks.
    sys.excepthook = mpiabort_excepthook

    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    # Use dill instead of pickle for transport
    rank = comm.Get_rank()
    size = comm.Get_size()

    # This first block is just to check how many rocess we have vs. need
    if rank == 0:
        # Read metadata and mode
        model_names = read_model_names(outdir)

        # Check whether the number of needed MPI is equal to the one in size
        # Otherwise scatter does not work properly. This is also because we are
        # Directly perscribing whuich rank sends to which for finite differences
        needed_size = 1  # One for synthetics
        for _mname in model_names:
            if _mname in Constants.nosimpars or _mname in Constants.mt_params:
                needed_size += 1
            else:
                needed_size += 2
    else:
        needed_size = None

    needed_size = comm.bcast(needed_size)

    if needed_size != size:
        raise ValueError('The number of MPI processes MUST be equal the number'
                         f'of required forward simulations. Need: {needed_size}')

    # This block is really starting stuff
    if rank == 0:

        # Get iter,step
        it = get_iter(outdir)
        ls = get_step(outdir)

        # Get the meta data directory
        metadir = os.path.join(outdir, 'meta')

        # Read metadata and model
        m = read_model(outdir, it, ls)
        model_names = read_model_names(outdir)

        # Read perturbation
        perturbation = read_perturbation(outdir)

        # Check whether the number of needed MPI is equal to the one in size
        # Otherwise scatter does not work properly. This is also because we are
        # Directly perscribing whuich rank sends to which for finite differences
        needed_size = 0
        for _mname in model_names:
            if _mname in Constants.nosimpars or _mname in Constants.mt_params:
                needed_size += 1
            else:
                needed_size += 1

        # Read original CMT solution
        cmt = cmt3d.CMTSource.from_CMTSOLUTION_file(
            os.path.join(metadir, 'init_model.cmt'))

        # Update the CMTSOLUTION with the current model state
        for _m, _mname in zip(m, model_names):
            setattr(cmt, _mname, _m)

        # Update half-duration afterwards.
        cmt.update_hdur()

        # Make the perturbations
        rankcounter = 0

        # Each rank map entry has the following entries
        # [source, parameternames, 1 | -1, send|receive rank, write]
        # 1 | -1 tells whihc is positive/ negative perturbation
        # send/receive rank: the negative pert is sent to positive pert
        # write: boolean that says whether the rank should write the output
        # array or not
        rankmap = []

        # Prepending synthetics
        syntcmt = deepcopy(cmt)
        rankmap.append([syntcmt, 'synt', None, None])
        rankcounter += 1

        # For the perturbations it's slightly more complicated.
        for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

            # Get the model name
            if _mname in Constants.nosimpars:

                # Get the model name
                syntcmt = deepcopy(cmt)
                rankmap.append([syntcmt, _mname, None, None])
                rankcounter += 1

            elif _mname in Constants.mt_params:

                dcmt = deepcopy(cmt)

                # Set all M... to zero
                for mpar in Constants.mt_params:
                    setattr(dcmt, mpar, 0.0)

                # Set one to none-zero
                setattr(dcmt, _mname, _pert)

                # Append the rank map
                rankmap.append([dcmt, _mname, None, None])
                rankcounter += 1

            else:

                """Computes the centered finite difference."""

                # create cmt copies
                pcmt = deepcopy(cmt)
                mcmt = deepcopy(cmt)

                # Get model values
                m = getattr(cmt, _mname)

                # Set values
                setattr(pcmt, _mname, m + _pert)
                setattr(mcmt, _mname, m - _pert)

                # Append the negative perturbation
                rankmap.append([pcmt, _mname, 1, rankcounter + 1])
                rankmap.append([mcmt, _mname, -1, rankcounter])

                # Increase the counter
                rankcounter += 2

    else:

        rankmap = None
        model_names = None
        perturbation = None

    # Broadcasting and scattering
    model_names = comm.bcast(model_names, root=0)
    perturbation = comm.bcast(perturbation, root=0)

    # Scatter the rank map
    rankmap = comm.scatter(rankmap, root=0)

    # Make the cmt solutions GF3D obsjects.
    cmt = cmt3d2gf3d(rankmap[0])
    par = rankmap[1]
    pert = rankmap[2]
    sr_rank = rankmap[3]

    print(rank, size, par, pert, sr_rank, flush=True)

    # Get the seismograms
    MS = MPISubset(os.path.join(outdir, 'meta', 'subset.h5'))
    data = MS.get_seismograms(cmt)

    comm.barrier()

    if par == 'synt':
        # We can directly write the synthetics
        write_synt_raw(MS.get_stream(cmt, data), outdir)

    elif par == 'time_shift':

        # If the parameter is timeshift we need to compute the gradient
        # and the multply by -1
        # Take the gradient
        data = np.gradient(data, MS.header['dt'], axis=2)
        drp = MS.get_stream(cmt, data*-1)

        # Then we can directly write the perturbed synthetics
        idx = model_names.index(par)
        write_dsdm_raw(drp, outdir, idx)

    elif par in Constants.mt_params:

        idx = model_names.index(par)
        pert = perturbation[idx]

        # For the moment tensor elements we only of the traces we only need to
        drp = MS.get_stream(cmt, data * (1/pert))

        write_dsdm_raw(drp, outdir, idx)

    else:

        idx = model_names.index(par)

        # Somehow the communication here gets pickling errors.
        if pert == 1:
            print(f"{rank:2d}/{size:2d}:", "Receiving...", flush=True)
            neg = np.empty_like(data)
            comm.Recv(neg, source=sr_rank, tag=idx)
            print(f"{rank:2d}/{size:2d}:",
                  "Received:", type(neg), flush=True)

        elif pert == -1:
            print(f"{rank:2d}/{size:2d}:",
                  "Sending:  ", type(data), flush=True)
            comm.Send(data, dest=sr_rank, tag=idx)
            print(f"{rank:2d}/{size:2d}:", "Sent.", flush=True)
        else:
            raise ValueError('Only 1 and -1 can be used for pert since it '
                             'indicates sending or receiving streams')

        if pert == 1:

            # Correction to make the output
            if par == "depth_in_m":
                _pert = perturbation[idx] / 1000.0
                # m/km -> making the dervative per km instead for
                # conformity with GFM.get_frechet, output
            else:
                _pert = perturbation[idx]

            # Combine forward and backward perturabtion for CFD
            data = (data - neg) * (1/(2 * _pert))

            # Write strem to pickle
            drp = MS.get_stream(cmt, data)
            write_dsdm_raw(drp, outdir, idx)

    comm.barrier()


if __name__ == "__main__":
    forward_kernel(sys.argv[1])
