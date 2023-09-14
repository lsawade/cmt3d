import os
import numpy as np
from lwsspy.utils.io import read_yaml_file
from lwsspy.seismo.costgradhess import CostGradHess
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


def hessian(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get input parameters
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get processparameters
    processparams = read_yaml_file(os.path.join(outdir, 'process.yml'))

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
        cgh = CostGradHess(
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
