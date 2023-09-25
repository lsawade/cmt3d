import os
import cmt3d
from copy import deepcopy
from .model import read_model, read_model_names, read_perturbation
from .kernel import write_dsdm_raw
from .forward import write_synt_raw
from .constants import Constants
from .log import get_iter, get_step
from .utils import cmt3d2gf3d


def forward_kernel(outdir):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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

        # Read original CMT solution
        cmt = cmt3d.CMTSource.from_CMTSOLUTION_file(
            os.path.join(metadir, 'init_model.cmt'))

        # Update the CMTSOLUTION with the current model state
        for _m, _mname in zip(m, model_names):
            setattr(cmt, _mname, _m)

        # Update half-duration afterwards.
        cmt.update_hdur()

        names = ['synt']
        cmts = [cmt]

         # For the perturbations it's slightly more complicated.
        for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

            print(_mname, _pert)

            # Get the model name
            if _mname in Constants.nosimpars:

                # Get the model name
                names.append(_mname)
                cmts.append(deepcopy(cmt))

            elif _mname in Constants.mt_params:

                dcmt = deepcopy(cmt)

                # Set all M... to zero
                for mpar in Constants.mt_params:
                    setattr(dcmt, mpar, 0.0)

                # Set one to none-zero
                setattr(dcmt, _mname, _pert)

                cmts.append(dcmt)
                names.append(_mname)

            else:

                """Computes the centered finite difference."""

                # create cmt copies
                pcmt = deepcopy(cmt)
                mcmt = deepcopy(cmt)

                # Get model values
                m = getattr(cmt, _mname)

                # Set vals
                setattr(pcmt, _mname, m + _pert)
                setattr(mcmt, _mname, m - _pert)

                # Append the negative perturbation
                cmts.append(mcmt)
                names.append(_mname + '_neg')

                # Append the positive perturbation
                cmts.append(pcmt)
                names.append(_mname + '_pos')

        # Loading pickle
        gfm = cmt3d.read_pickle(os.path.join(metadir, 'gfm.pkl'))

    else:

        model_names = None
        names = None
        cmts = None
        gfm = None
        perturbation = None

    # Broadcasting and scattering
    gfm = cmt3d.read_pickle(os.path.join(outdir, 'meta', 'gfm.pkl'))
    model_names = comm.bcast(model_names, root=0)
    perturbation = comm.bcast(perturbation, root=0)
    names = comm.bcast(names, root=0)
    name = comm.scatter(names, root=0)
    cmt = comm.scatter(cmts, root=0)


    # Get the model name
    drp = gfm.get_seismograms(cmt3d2gf3d(cmt))


    if name == 'synt':
        write_synt_raw(drp, outdir)
        drp = [None,]

    elif name == 'time_shift':
        drp.differentiate()
        for tr in drp:
            tr.data *= -1

        idx = model_names.index(name)
        write_dsdm_raw(drp, outdir, idx)
        drp = [None,]

    elif name in Constants.mt_params:

        idx = model_names.index(name)
        pert = perturbation[idx]

        for tr in drp:
            tr.data *= 1/pert

        write_dsdm_raw(drp, outdir, idx)
        drp = [None,]

    else:
        drp = drp

    # Gather all the data
    drp = comm.gather(drp, root=0)

    if rank == 0:

        for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

            if _mname in Constants.mt_params or _mname in Constants.nosimpars:

                continue

            pidx = names.index(_mname + '_pos')
            midx = names.index(_mname + '_neg')

            # Correction to make the output
            if _mname == "depth_in_m":
                pert = _pert * 1000.0
                # m/km -> making the dervative per km instead for
                # conformity with GFM.get_frechet, output
            else:
                pert = _pert

            for ptr, mtr in zip(drp[pidx], drp[midx]):
                ptr.data -= mtr.data
                ptr.data /= 2 * _pert

            write_dsdm_raw(drp[pidx], outdir, _i)




