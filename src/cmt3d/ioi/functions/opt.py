import os
import wave
import numpy as np
from lwsspy.utils.io import read_yaml_file

from .model import read_model, read_model_names, write_model, read_scaling
from .cost import read_cost, write_cost
from .descent import read_descent
from .forward import read_synt, write_synt
from .kernel import read_dsdm, write_dsdm
from .gradient import read_gradient, write_gradient
from .hessian import read_hessian, write_hessian
from .linesearch import read_optvals
from .log import write_status, get_iter, get_step, write_log


def constrain_model(outdir, m):
    """Only constrains parameters if
    ``parameter_constraints`` is set in the input file"""

    # Get input parameters
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # If no parameters should be constraint, return
    if hasattr(inputparams, 'parameter_constraints') is False:
        return m
    elif inputparams['parameter_constraints'] is None:
        return m
        
    # Get lower and upper constraints
    lower = inputparams['parameter_constraints']['lower']
    upper = inputparams['parameter_constraints']['upper']

    # Get model names
    mnames = read_model_names(outdir)

    # Set lower bound for the model update
    if lower is not None:
        for _par, _low in lower.items():
            idx = mnames.index(_par)

            if m[idx] < _low:
                m[idx] = _low

    # Set upper bound for the model update
    if upper is not None:
        for _par, _upp in upper.items():
            idx = mnames.index(_par)

            if m[idx] > _upp:
                m[idx] = _upp

    return m


def update_model(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Read model, descent direction, and optvals (alpha)
    m = read_model(outdir, it, ls - 1)
    dm = read_descent(outdir, it, 0)
    _, _, _, alpha, _, _, _ = read_optvals(outdir, it, ls-1)

    # Compute new model
    m_new = m + alpha * dm

    # Constrain model if ``parameter_constraints`` is set in the `input.yml`
    m_new = constrain_model(outdir, m_new)

    # Write new model
    write_model(m_new, outdir, it, ls)

    write_log(
        outdir, f"      m: {np.array2string(m_new, max_line_width=int(1e10))}")


def update_mcgh(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Read input params
    processparams = read_yaml_file(os.path.join(outdir, 'process.yml'))

    # Get wave type
    wavetypes = list(processparams.keys())

    # Read all relevant data
    m = read_model(outdir, it, ls)
    c = read_cost(outdir, it, ls)
    g = read_gradient(outdir, it, ls)
    h = read_hessian(outdir, it, ls)

    # Write for the first iteration and 0 ls
    write_model(m, outdir, it+1, 0)
    write_cost(c, outdir, it+1, 0)
    write_gradient(g, outdir, it+1, 0)
    write_hessian(h, outdir, it+1, 0)

    # Get number of parameters
    NM = len(read_model(outdir, it, ls))

    # Copy the
    for _wtype in wavetypes:
        synt = read_synt(outdir, _wtype, it, ls)
        write_synt(synt, outdir, _wtype, it+1, 0)

        for _i in range(NM):
            dsdm = read_dsdm(outdir, _wtype, _i, it, ls)
            write_dsdm(dsdm, outdir, _wtype, _i, it+1, 0)


def check_status(statdir):
    fname = "STATUS.txt"
    file = os.path.join(statdir, fname)

    with open(file, "r") as f:
        message = f.read()

    print("    STATUS:", message)

    if "FAIL" in message:
        return False
    else:
        return True


def check_done(outdir):

    # Get iter,step
    it = get_iter(outdir)
    # ls = get_step(outdir)
    
    # Read input parameters and optimization characteristics
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get stopping parameters from input file
    niter_max = inputparams['optimization']['niter_max']
    stopping_criterion = inputparams['optimization']['stopping_criterion']
    stopping_criterion_cost_change = inputparams['optimization']['stopping_criterion_cost_change']
    stopping_criterion_model = inputparams['optimization']['stopping_criterion_model']

    # Read cost
    cost_init = read_cost(outdir, 0, 0)
    cost_old = read_cost(outdir, it, 0)
    cost = read_cost(outdir, it+1, 0)

    # Get the scaled models
    scaling = read_scaling(outdir)
    smodel_prev = read_model(outdir, it, 0)/scaling
    smodel = read_model(outdir, it+1, 0)/scaling

    # Read necessary vals
    # _, _, _, alpha, _, _, _ = read_optvals(outdir, it, ls)
    # descent = read_descent(descdir, it, ls)
    # descent_prev = read_descent(descdir, it, ls-1)
    # model_init = read_model(modldir, 0, 0)

    STATUS = False

    if (np.abs(cost - cost_old)/cost_init < stopping_criterion_cost_change):
        message = "FINISHED: Cost function not decreasing enough to justify iteration."
        write_status(outdir, message)
        STATUS = True
    elif (cost/cost_init < stopping_criterion):
        message = "FINISHED: Optimization algorithm has converged."
        write_status(outdir, message)
        STATUS = True
    elif np.sum(smodel - smodel_prev)**2/np.sum((smodel_prev)**2) \
            < stopping_criterion_model:
        message = "FINISHED: Model is not updating enough anymore."
        write_status(outdir, message)
        STATUS = True
    elif niter_max == it:
        message = "FINISHED: Maximum # of iterations reached."
        write_status(outdir, message)
        STATUS = True

    return STATUS
