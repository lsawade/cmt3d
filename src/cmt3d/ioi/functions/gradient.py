import os
import numpy as np
import cmt3d
from .data import read_data_windowed
from .forward import read_synt
from .kernel import read_dsdm
from .model import read_model_names, read_model
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


def read_gradient_all(outdir):

    # Get directory
    graddir = os.path.join(outdir, 'grad')

    grads = []
    for _cfile in sorted(os.listdir(graddir)):
        if "ls00000" in _cfile:
            grads.append(np.load(os.path.join(graddir, _cfile)))

    return np.vstack(grads)


def constrain_gradient(outdir, m, g):
    """Only constrain gradient if ``parameter_constraints`` is set in
    the input file. Needs both model and gradient to determine, whether
    the model already hit the gradient."""

    # Get input parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # If no parameters should be constraint, return
    if (('parameter_constraints' in inputparams) is False) or (
            inputparams['parameter_constraints'] is None):
        write_log(outdir, f"No constraints")
        return g
    else:
        write_log(outdir, f"{inputparams['parameter_constraints']}")

    # Get lower and upper constraints
    lower = inputparams['parameter_constraints']['lower']
    upper = inputparams['parameter_constraints']['upper']

    # Get model names
    mnames = read_model_names(outdir)

    # Write of
    write_log(outdir, f"Original gradient: {g}")

    # Set lower bound for the model update
    if lower is not None:
        for _par, _low in lower.items():
            idx = mnames.index(_par)

            if np.isclose(m[idx], _low):

                # Only set gradient to zero if it is positive. Here, positive
                # means that we are moving in the negative direction, for
                # the cost function. So, we only set the gradient to zero
                # if it's larger than 0
                if g[idx] > 0:
                    g[idx] = 0
                    write_log(outdir, f"Constraining gradient for {_par} to 0")

    # Set upper bound for the model update
    if upper is not None:
        for _par, _upp in upper.items():
            idx = mnames.index(_par)

            if np.isclose(m[idx], _upp):

                # Only set gradient to zero if it is negative. Here, positive
                # means that we are moving in the negative direction (towards
                # the minimum). But this means that a negative gradient moves
                # inversely to the minimum of the cost function. So, to make
                # sure we don't move past the upper limit, we set all negative
                # gradients to zero.
                if g[idx] < 0:
                    write_log(
                        outdir, f"Constraining gradient of {_par} to0 ")

                    g[idx] = 0

    write_log(outdir, f"Constrained gradient: {g}")

    return g



def gradient(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get input parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get processparameters
    processparams = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Read model to check whether to constrain
    m = read_model(outdir, it, ls)

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
        cgh = cmt3d.CostGradHess(
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

    # Check constraints, given the parameters gradient may or may not be
    # set to 0
    grad = constrain_gradient(outdir, m, grad)

    # Write Gradients
    write_gradient(grad, outdir, it, ls)

    write_log(
        outdir, f"      g: {np.array2string(grad, max_line_width=int(1e10))}")
