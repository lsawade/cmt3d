import os
import numpy as np
from lwsspy.utils.io import read_yaml_file
from lwsspy.seismo.costgradhess import CostGradHess
from .data import read_data_windowed
from .forward import read_synt
from .kernel import read_dsdm
from .model import read_model_names
from .log import get_iter, get_step, write_log


def write_gradient(g, outdir, it, ls=None):

    # Get graddir
    graddir = os.path.join(outdir, 'grad')

    # Get filename
    if ls is not None:
        fname = f"grad_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"grad_it{it:05d}.npy"

    # Full filename
    file = os.path.join(graddir, fname)

    # Save
    np.save(file, g)


def read_gradient(outdir, it, ls=None):

    # Get graddir
    graddir = os.path.join(outdir, 'grad')

    if ls is not None:
        fname = f"grad_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"grad_it{it:05d}.npy"

    file = os.path.join(graddir, fname)

    return np.load(file)


def gradient(outdir):

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
    grad = np.zeros(NM)

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
            grad += cgh.grad() * processparams[_wtype]["weight"]
        else:
            grad += cgh.grad()

    # Write Gradients
    write_gradient(grad, outdir, it, ls)

    write_log(
        outdir, f"      g: {np.array2string(grad, max_line_width=int(1e10))}")
