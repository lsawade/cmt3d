import imp
import os
from copy import deepcopy
from obspy import Stream
from lwsspy.seismo.source import CMTSource


from .constants import Constants
from .model import read_model, read_model_names, read_perturbation
from .utils import write_pickle, read_pickle
from .log import get_iter, get_step


def write_dsdm(dsdm: Stream, outdir, wavetype, nm, it, ls=None):

    # Get the synthetics directory
    dsdmdir = os.path.join(outdir, 'dsdm')

    # Get filename
    if ls is not None:
        fname = f'dsdm{nm:05d}_{wavetype}_it{it:05d}_ls{ls:05d}.pkl'
    else:
        fname = f'dsdm{nm:05d}_{wavetype}_it{it:05d}.pkl'

    # Get output file name
    file = os.path.join(dsdmdir, fname)

    # Write output
    write_pickle(file, dsdm)


def read_dsdm(outdir, wavetype, nm, it, ls=None) -> Stream:

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get the synthetics directory
    dsdmdir = os.path.join(outdir, 'dsdm')

    # Get filename
    if ls is not None:
        fname = f'dsdm{nm:05d}_{wavetype}_it{it:05d}_ls{ls:05d}.pkl'
    else:
        fname = f'dsdm{nm:05d}_{wavetype}_it{it:05d}.pkl'

    # Get output file name
    file = os.path.join(dsdmdir, fname)

    return read_pickle(file)


def update_cmt_dsdm(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    modldir = os.path.join(outdir, 'modl')
    metadir = os.path.join(outdir, 'meta')
    sdsmdir = os.path.join(outdir, 'simu', 'dsdm')

    # Read metadata and model
    m = read_model(outdir, it, ls)
    model_names = read_model_names(outdir)

    # Read perturbation
    perturbation = read_perturbation(outdir)

    # Read original CMT solution
    cmt = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmt, _mname, _m)

    # Update half-duration afterwards.
    cmt.update_hdur()

    # For the perturbations it's slightly more complicated.
    for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

        if _mname not in Constants.nosimpars:

            cmtfiledest = os.path.join(
                sdsmdir, f"dsdm{_i:05d}", "DATA", "CMTSOLUTION")

            # Perturb source at parameter
            cmt_dsdm = deepcopy(cmt)

            if _pert is not None:

                # If parameter a part of the tensor elements then set the
                # rest of the parameters to 0.
                if _mname in Constants.mt_params:
                    for _tensor_el in Constants.mt_params:
                        if _tensor_el != _mname:
                            setattr(cmt_dsdm, _tensor_el, 0.0)
                        else:
                            setattr(cmt_dsdm, _tensor_el, _pert)
                else:

                    # Get the parameter to be perturbed
                    to_be_perturbed = getattr(cmt_dsdm, _mname)

                    # Perturb the parameter
                    to_be_perturbed += _pert

                    # Set the perturb
                    setattr(cmt_dsdm, _mname, to_be_perturbed)

            cmt_dsdm.write_CMTSOLUTION_file(cmtfiledest)


# def frechet(param: int, modldir, metadir, frecdir, it, ls):

#     # Read metadata and model
#     m = read_model(modldir, it, ls)
#     X = read_metadata(metadir)

#     # Forward modeling
#     frechet = dgdm(m, X, param)

#     # Write Frechet derivative
#     write_frechet(frechet, param, frecdir, it, ls)
