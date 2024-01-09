import os
import numpy as np
import cmt3d
from .data import read_data_windowed
from .forward import read_synt
from .kernel import read_dsdm
from .model import read_model_names
from .log import get_step, get_iter


def write_hessian(h, outdir, it, ls):

    # Get graddir
    hessdir = os.path.join(outdir, 'hess')

    # Get filename
    if ls is not None:
        fname = f"hess_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"hess_it{it:05d}.npy"

    # Full filename
    file = os.path.join(hessdir, fname)

    # Save
    np.save(file, h)


def read_hessian(outdir, it, ls):

    # Get graddir
    hessdir = os.path.join(outdir, 'hess')

    # Get filename
    if ls is not None:
        fname = f"hess_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"hess_it{it:05d}.npy"

    # Full filename
    file = os.path.join(hessdir, fname)

    return np.load(file)


def read_hessian_all(outdir):

    # Get directory
    hessdir = os.path.join(outdir, 'hess')

    hesss = []
    for _cfile in sorted(os.listdir(hessdir)):
        if "ls00000" in _cfile:
            hesss.append(np.load(os.path.join(hessdir, _cfile)))

    return np.vstack(hesss)



def hessian(outdir, it=None, ls=None):

    # Get iter,step
    if it is None:
        it = get_iter(outdir)
    if ls is None:
        ls = get_step(outdir)

    # Get input parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get processparameters
    processparams = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Weighting?
    weighting = inputparams['weighting']

    # Normalize?
    normalize = inputparams['normalize']

    # Get number of modelparams
    NM = len(read_model_names(outdir))

    # Compute total cost
    hess = np.zeros((NM, NM))

    for _wtype in processparams.keys():

        data = read_data_windowed(outdir, _wtype)
        synt = read_synt(outdir, _wtype, it, ls)

        # Get all frechet derivatives
        dsyn = list()
        for _i in range(NM):
            dsyn.append(read_dsdm(outdir, _wtype, _i, it, ls))

        # Create CostGradHess object
        cgh = cmt3d.CostGradHess(
            data=data,
            synt=synt,
            dsyn=dsyn,
            verbose=False,
            normalize=normalize,
            weight=weighting)

        if weighting:
            hess += cgh.hess() * processparams[_wtype]["weight"]
        else:
            hess += cgh.hess()

    # Write Gradients
    write_hessian(hess, outdir, it, ls)
