import os
import numpy as np
import cmt3d
from .data import read_data_windowed
from .forward import read_synt
from .log import get_iter, get_step, write_log


def write_cost(c, outdir, it, ls=None):

    # Get directory
    costdir = os.path.join(outdir, 'cost')

    if ls is not None:
        fname = f"cost_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"cost_it{it:05d}.npy"
    file = os.path.join(costdir, fname)
    np.save(file, c)


def read_cost(outdir, it, ls=None):

    # Get directory
    costdir = os.path.join(outdir, 'cost')

    if ls is not None:
        fname = f"cost_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"cost_it{it:05d}.npy"
    file = os.path.join(costdir, fname)
    return np.load(file)

def read_cost_all(outdir):

    # Get directory
    costdir = os.path.join(outdir, 'cost')

    cost = []
    for _cfile in sorted(os.listdir(costdir)):
        if "ls00000" in _cfile:
            cost.append(np.load(os.path.join(costdir, _cfile)))

    return np.array(cost)


def cost(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get input parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get processparameters
    processparams = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Weighting?
    weighting = inputparams['weighting']

    # Normalize?
    normalize = inputparams['normalize']

    # Compute total cost
    cost = 0.0
    for _wtype in processparams.keys():

        data = read_data_windowed(outdir, _wtype)
        synt = read_synt(outdir, _wtype, it, ls)

        cgh = cmt3d.CostGradHess(
            data=data,
            synt=synt,
            verbose=False,
            normalize=normalize,
            weight=weighting)

        if weighting:
            cost += cgh.cost() * processparams[_wtype]["weight"]
        else:
            cost += cgh.cost()

    write_cost(cost, outdir, it, ls)
    print(cost)
    write_log(
        outdir, f"      c: {np.array2string(cost, max_line_width=int(1e10))}")
