"""

All functions to compute a full linsearch step for the MFPCGH method.
- [M]odel update
- [F]orward kernel
- [P]rocessing of synthetics
- [C]ost computation
- [G]radient computation
- [H]essian computation
- [L]inesearch
- [C]heck optvals


"""

# External
import os
import sys
import dill
import time
import cmt3d
from copy import deepcopy
import numpy as np
from obsproclib import process_stream, stream_multiply
from mpi4py import MPI
from .kernel import write_dsdm, read_dsdm_raw
from .forward import write_synt, read_synt_raw


comm = MPI.COMM_WORLD

# Use dill instead of pickle for transport
rank = comm.Get_rank()
size = comm.Get_size()

# Internal
from ... import read_yaml
from .data import read_data_windowed
from .forward import read_synt
from .kernel import read_dsdm
from .model import read_model, read_model_names, read_perturbation
from .kernel import write_dsdm_raw
from .forward import write_synt_raw
from .linesearch import check_optvals, linesearch
from .constants import Constants
from .log import get_iter, get_step, write_log
from .utils import cmt3d2gf3d
from .opt import update_model
from .cost import cost
from .gradient import gradient
from .hessian import hessian
from ...costgradhess import CostGradHess
from ...source import CMTSource
from ... import read_inventory

from gf3d.mpi_subset import MPISubset
import _pickle


def mpiabort_excepthook(type, value, traceback):

    if hasattr(traceback, "print_exc"):
        traceback.print_exc()

    print("", flush=True)
    time.sleep(1.0)
    import mpi4py.MPI
    mpi_comm = mpi4py.MPI.COMM_WORLD
    mpi_comm.Abort(1)

# Set abort hook on all ranks.
# sys.excepthook = mpiabort_excepthook


def model_update(outdir, it=None, ls=None):

    if rank == 0:
        update_model(outdir, it=it, ls=ls)


    comm.barrier()


def forward_kernel(outdir, it=None, ls=None, verbose=True):

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


    # if needed_size != size:
    #     raise ValueError('The number of MPI processes MUST be equal the number'
    #                      f'of required forward simulations. Need: {needed_size}')


    # This block is really starting stuff
    if rank == 0:

        # Get iter,step
        if it is None:
            it = get_iter(outdir)
        if ls is None:
            ls = get_step(outdir)

        # Get the meta data directory
        metadir = os.path.join(outdir, 'meta')

        # Read metadata and model
        m = read_model(outdir, it, ls)
        model_names = read_model_names(outdir)

        # Read perturbation
        perturbation = read_perturbation(outdir)

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

        # Adding the
        missing_cores = size - len(rankmap)
        rankmap += [[None, None, None, None],] * missing_cores

    else:

        rankmap = None
        model_names = None
        perturbation = None

    if rank == 0 and verbose:
        print('--> Set up rankmap')

    # Broadcasting and scattering
    model_names = comm.bcast(model_names, root=0)
    perturbation = comm.bcast(perturbation, root=0)

    # Scatter the rank map
    rankmap = comm.scatter(rankmap, root=0)

    # Making the cmt solutions GF3D obsjects.
    if rank == 0 and verbose:
        print('--> Scattered rankmap', flush=True)

    # Make the cmt solutions GF3D obsjects.
    tcmp = rankmap[0]
    par = rankmap[1]
    pert = rankmap[2]
    sr_rank = rankmap[3]

    if tcmp is not None:
        cmt = cmt3d2gf3d(rankmap[0])

    print(rank, size, par, pert, sr_rank, flush=True)

    # Get the MPI subset needs to be read by all cores because of the
    # broadcasting!!!!
    MS = MPISubset(os.path.join(outdir, 'meta', 'subset.h5'))

    if rank == 0 and verbose:
        print('--> Created MPISubset class', flush=True)


    # Can be done by relevant ranks only
    if par is not None:
        data = MS.get_seismograms(cmt)

    comm.barrier()

    if par == 'synt':
        # We can directly write the synthetics
        write_synt_raw(MS.get_stream(cmt, data), outdir)

        if verbose:
            print('--> Written synt raw', flush=True)

    elif par == 'time_shift':

        # If the parameter is timeshift we need to compute the gradient
        # and the multply by -1
        # Take the gradient
        data = np.gradient(data, MS.header['dt'], axis=2)
        drp = MS.get_stream(cmt, data*-1)

        # Then we can directly write the perturbed synthetics
        idx = model_names.index(par)
        write_dsdm_raw(drp, outdir, idx)

        if verbose:
            print('--> Written timeshift', flush=True)


    elif par in Constants.mt_params:

        idx = model_names.index(par)
        pert = perturbation[idx]

        # For the moment tensor elements we only of the traces we only need to
        drp = MS.get_stream(cmt, data * (1/pert))


        write_dsdm_raw(drp, outdir, idx)

        if verbose:
            print(f'--> Written {par}', flush=True)

    elif par is None:
        pass

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
        if verbose:
            print(f'--> Written {par}', flush=True)

    if rank == 0 and verbose:
        print(f"--> pre barrier", flush=True)

    comm.barrier()

    if rank==0 and verbose:
        print(f"--> post barrier", flush=True)


def process_all_synt(outdir, it=None, ls=None, verbose=True):

    if rank == 0 and verbose:

        if rank == 0 and verbose:
            print(f"Setup processing")
            print("-> Loading parameters")

        # Get processing parameters
        processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

        # Get iter,step
        if it is None:
            it = get_iter(outdir)
        if ls is None:
            ls = get_step(outdir)

        # Define directory
        metadir = os.path.join(outdir, 'meta')

        # Get CMT
        cmtsource = CMTSource.from_CMTSOLUTION_file(os.path.join(
            metadir, 'init_model.cmt'
        ))

        # Get parameters
        processdict = read_yaml(os.path.join(outdir, 'process.yml'))

        # Read metadata
        stations = read_inventory(os.path.join(metadir, 'stations.xml'))

        # Read model and model name
        mnames = read_model_names(outdir)

        # Get the frechet derivatives
        NM = len(mnames)

        # Get the frechet derivatives
        syntlist = ['synt', ]

        for i in range(NM):
            syntlist.append(i)

        processing_list = []

        for wave in processdict.keys():
            for synt in syntlist:
                print(wave, synt)
                processing_list.append([wave, synt])

        extra_cores =  size - len(processing_list)

        if verbose:
            print(f"--> Size {size} -- Needed: {len(processing_list)} -- idle: {extra_cores}")

        if extra_cores > 0:
            processing_list += [[None, None],] * extra_cores
        elif extra_cores < 0:
            raise ValueError('Too few cores to scatter the processing_tasks.')



    else:

        cmtsource = None
        processdict = None
        stations = None
        processing_list = None
        mnames = None


    if rank == 0 and verbose:
        for _i,item in enumerate(processing_list):
            print(processing_list)
        print(f"--> pre barrier", flush=True)

    comm.barrier()

    # Broadcast
    cmtsource = comm.bcast(cmtsource, root=0)
    processdict = comm.bcast(processdict, root=0)
    stations = comm.bcast(stations, root=0)
    processing_list = comm.scatter(processing_list, root=0)
    mnames = comm.bcast(mnames, root=0)

    # Unpack
    print(rank, size, processing_list, flush=True)
    wave, param = processing_list

    # Only do stuff if wave and param are not None
    if param is not None:
        # Compute start and end time of processing depeding on relative times
        # of the wave type
        starttime = cmtsource.cmt_time \
            + processdict[wave]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[wave]["process"]["relative_endtime"]

        # Adapt processing dictionary
        tprocessdict = deepcopy(processdict[wave]["process"])
        tprocessdict.pop("relative_starttime")
        tprocessdict.pop("relative_endtime")
        tprocessdict["starttime"] = starttime
        tprocessdict["endtime"] = endtime
        tprocessdict["inventory"] = stations
        tprocessdict.update(dict(
            remove_response_flag=False,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=False))

    if param == 'synt':

        # Read the synthetics
        synt = read_synt_raw(outdir)

        # Process the synthetics
        synt = process_stream(synt, **tprocessdict)

        # Write the synthetics
        write_synt(synt, outdir, wave, it, ls)

        if verbose:
            print(f'--> Written synt-{wave}', flush=True)

    elif isinstance(param, int):

        # Read the synthetics
        dsdm = read_dsdm_raw(outdir, param)

        # Process the synthetics
        dsdm = process_stream(dsdm, **tprocessdict)

        if mnames[param] == "depth_in_m":
            stream_multiply(dsdm, 1.0/1000.0)

        # Write the synthetics
        write_dsdm(dsdm, outdir, wave, param, it, ls)

        if verbose:
            print(f'--> Written dsdm#{param:05d}-{wave}', flush=True)


    elif param is None:
        pass

    else:
        raise ValueError(f'Param must be either "synt" or an integer or None, but is {param}')

    if rank == 0 and verbose:
        print(f"--> pre barrier", flush=True)
    comm.barrier()
    if rank == 0 and verbose:
        print(f"--> post barrier", flush=True)

def cghlc(outdir, it=None, ls=None, cgh_only=False, verbose=True):

    # Get iter,step
    if rank==0:
        if verbose==True:
            print("--> Computing cost, gradient and hessian")
            print("--> Reading parameters")


        # Get iter,step
        if it is None:
            it = get_iter(outdir)
        if ls is None:
            ls = get_step(outdir)


        # Get input parameters
        inputparams = read_yaml(os.path.join(outdir, 'input.yml'))

        # Get processparameters
        processparams = read_yaml(os.path.join(outdir, 'process.yml'))

        # Weighting?
        weighting = inputparams['weighting']

        # Normalize?
        normalize = inputparams['normalize']

        # Get number of modelparams
        NM = len(read_model_names(outdir))

    else:
        weighting = None
        normalize = None
        processparams = None
        NM = None

    comm.barrier()

    # Broadcast
    processparams = comm.bcast(processparams, root=0)
    it = comm.bcast(it, root=0)
    ls = comm.bcast(ls, root=0)
    weighting = comm.bcast(weighting, root=0)
    normalize = comm.bcast(normalize, root=0)
    NM = comm.bcast(NM, root=0)

    # Compute total cost
    if rank==0:
        cost(outdir, it=it, ls=ls)

    if rank==1:
        gradient(outdir, it=it, ls=ls)

    if rank==2:
        hessian(outdir, it=it, ls=ls)

    comm.barrier()

    # Read model names
    if rank==0 and not cgh_only:

        linesearch(outdir, it=it, ls=ls)

        # check_optvals(outdir, status=True, it=it, ls=ls)

    comm.barrier()

