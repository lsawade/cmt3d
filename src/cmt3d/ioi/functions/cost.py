import os
import numpy as np
from lwsspy.seismo.costgradhess import CostGradHess
from lwsspy.utils.io import read_yaml_file
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


def cost(outdir):

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

    # Compute total cost
    cost = 0.0
    for _wtype in processparams.keys():

        data = read_data_windowed(outdir, _wtype)
        synt = read_synt(outdir, _wtype, it, ls)

        cgh = CostGradHess(
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

    write_log(
        outdir, f"      c: {np.array2string(cost, max_line_width=int(1e10))}")
