import imp
import os
from copy import deepcopy
import obspy
import cmt3d
from  gf3d.seismograms import GFManager
from .constants import Constants
from .model import read_model, read_model_names, read_perturbation
from .log import get_iter, get_step
from .utils import cmt3d2gf3d


def write_dsdm_raw(dsdm: obspy.Stream, outdir: str, nm: int):

    # Get the synthetics directory
    dsdmdir = os.path.join(outdir, 'simu', 'dsdm')

    # Get filename
    fname = f'dsdm{nm:05d}.pkl'

    # Get output file name
    filename = os.path.join(dsdmdir, fname)

    # Write output
    cmt3d.write_pickle(filename, dsdm)


def read_dsdm_raw(outdir: str, nm: int) -> obspy.Stream:

    # Get the synthetics directory
    dsdmdir = os.path.join(outdir, 'simu', 'dsdm')

    # Get filename
    fname = f'dsdm{nm:05d}.pkl'

    # Get output file name
    filename = os.path.join(dsdmdir, fname)

    # Write output
    dsdm = cmt3d.read_pickle(filename)  # type: obspy.Stream

    return dsdm


def write_dsdm(dsdm: obspy.Stream, outdir, wavetype, nm, it, ls=None):

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
    cmt3d.write_pickle(file, dsdm)


def read_dsdm(outdir, wavetype, nm, it, ls=None) -> obspy.Stream:

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

    return cmt3d.read_pickle(file)


def kernel(outdir, gfm: GFManager):

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

    # Get frechet derivatives
    gf3d_source = cmt3d2gf3d(cmt)
    dsdm = gfm.get_frechet(gf3d_source, rtype=2)

    # For the perturbations it's slightly more complicated.
    for _i, (_pert, _mname) in enumerate(zip(perturbation, model_names)):

        if _mname not in Constants.nosimpars:

            # Get the model name
            modelname = Constants.cmt3d2gf3d_par[_mname]

            # write frechet derivative
            write_dsdm_raw(dsdm[modelname], outdir, _i)




