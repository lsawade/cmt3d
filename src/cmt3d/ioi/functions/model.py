
import os
import copy
import numpy as np
import typing as tp
import cmt3d
from .constants import Constants
from .log import get_iter, get_step


def write_perturbation(perturbation, outdir):
    metadir = os.path.join(outdir, 'meta')
    perturbation = [np.nan if _p is None else _p for _p in perturbation]
    np.save(os.path.join(
        metadir, 'perturbation.npy'), perturbation)


def read_perturbation(outdir):
    metadir = os.path.join(outdir, 'meta')
    perturbation = np.load(os.path.join(metadir, 'perturbation.npy'))
    perturbation = [None if np.isnan(_p) else _p for _p in perturbation]
    return perturbation


def write_scaling(scaling, outdir):
    metadir = os.path.join(outdir, 'meta')
    np.save(os.path.join(
        metadir, 'scaling.npy'), scaling)


def read_scaling(outdir):
    metadir = os.path.join(outdir, 'meta')
    return np.load(os.path.join(metadir, 'scaling.npy'))


def write_model_names(model_names, outdir):
    metadir = os.path.join(outdir, 'meta')
    model_names = np.save(os.path.join(
        metadir, 'model_names.npy'), np.array(model_names))


def read_model_names(outdir):
    metadir = os.path.join(outdir, 'meta')
    return np.load(os.path.join(metadir, 'model_names.npy')).tolist()


def print_model_names(outdir):

    # Get model names
    model_names = read_model_names(outdir)

    # Print model names
    for _i, _name in enumerate(model_names):
        print(f"{_i:>5}: {_name}")


def get_cmt(
        outdir: str, it: tp.Optional[int] = None, ls: int = 0,
        outfile: tp.Optional[str] = None):

    # Get iter,step
    if it is None:
        it = get_iter(outdir)

    # Get dirs
    metadir = os.path.join(outdir, 'meta')

    # Read metadata and model
    m = read_model(outdir, it, ls)
    model_names = read_model_names(outdir)

    # Read original CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt')
    )

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmtsource, _mname, _m)

    # Update half-duration afterwards.
    if it != 0 or ls != 0:
        cmtsource.update_hdur()

    # Write CMTSOLUTION to oufile
    if outfile:
        cmtsource.write_CMTSOLUTION_file(outfile)
        return None

    # Otherwise return cmtsource
    else:
        return cmtsource

def get_cmt_all(outdir: str):

    # Read metadata and model
    model_names = read_model_names(outdir)

    # Read original CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', 'init_model.cmt')
    )

    # Get all models
    model_names = read_model_names(outdir)
    models = read_model_all(outdir)

    # Init list of sources
    cmts =[]

    # Loop over models
    for _it, _model in enumerate(models):

        if _it == 0:
            cmts.append(cmtsource)
        else:

            # Copy cmt
            outcmt = copy.deepcopy(cmtsource)

            # Update the CMTSOLUTION with the current model state
            for _m, _mname in zip(_model, model_names):
                setattr(outcmt, _mname, _m)

            # Update half-duration afterwards
            outcmt.update_hdur()

            # Append to cmts list
            cmts.append(outcmt)

    return cmts


def get_simpars(outdir):

    model_names = read_model_names(outdir)
    idx = []
    for _i, _mname in enumerate(model_names):
        if _mname in Constants.nosimpars:
            continue
        else:
            idx.append(_i)

    return idx


def write_model(m, outdir, it, ls=None):
    """Takes in model vector, modldirectory, iteration and linesearch number
    and write model to modl directory.

    Parameters
    ----------
    m : ndarray
        modelvector
    modldir : str
        model directory
    it : int
        iteration number
    ls : int, optional
        linesearch number
    """

    # Create filename that contains both iteration and linesearch number
    if ls is not None:
        fname = f"m_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"m_it{it:05d}.npy"

    file = os.path.join(outdir, 'modl', fname)
    np.save(file, m)


def read_model(outdir, it, ls=None):
    """Reads model vector

    Parameters
    ----------
    modldir : str
        model directory
    it : int
        iteration number
    ls : int, optional
        linesearch number

    Returns
    -------
    ndarray
        model vector
    """

    if ls is not None:
        fname = f"m_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"m_it{it:05d}.npy"
    file = os.path.join(outdir, 'modl', fname)
    m = np.load(file)
    return m



def read_model_all(outdir):

    # Get directory
    modldir = os.path.join(outdir, 'modl')

    modls = []
    for _mfile in sorted(os.listdir(modldir)):
        if "ls00000" in _mfile:
            modls.append(np.load(os.path.join(modldir, _mfile)))

    return np.vstack(modls)


