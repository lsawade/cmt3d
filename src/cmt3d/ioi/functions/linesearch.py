import os
import numpy as np
from lwsspy.utils.io import read_yaml_file

from .log import write_log, write_status
from .wolfe import wolfe_conditions, update_alpha
from .cost import read_cost
from .descent import read_descent
from .gradient import read_gradient
from .log import get_iter, get_step


def write_optvals(optvals, outdir, it, ls=None):
    """writes the optimization parameters to file

    Parameters
    ----------
    optvals : list
        the ndarray contains q, alphaleft, alpharight, alpha, w1, w2, w3.
    optdir : str
        optimization directory
    it : int
        iteration number
    ls : int, optional
        iteration number, by default None
    """

    # Get opt dir
    optdir = os.path.join(outdir, 'opt')

    # Fname
    if ls is not None:
        fname = f"optvals_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"optvals_it{it:05d}.npy"

    # Full filename
    file = os.path.join(optdir, fname)

    # save optimization values
    np.save(file, optvals)


def read_optvals(outdir, it, ls=None):
    """Reads the optimization values q, alpha, alpha left, and alpha right,
    and the three wolf condition number w1,w2,w3. into a tuple

    Parameters
    ----------
    optdir : str
        optimization directory
    it : int
        iteration number
    ls : int, optional
        linesearch number, by default None
    """

    # Get opt dir
    optdir = os.path.join(outdir, 'opt')

    if ls is not None:
        fname = f"optvals_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"optvals_it{it:05d}.npy"
    file = os.path.join(optdir, fname)
    optvals = np.load(file)
    optvals = optvals.tolist()

    # Convert the wolfe conditions to booleans
    optvals[-1] = bool(optvals[-1])
    optvals[-2] = bool(optvals[-2])
    optvals[-3] = bool(optvals[-3])

    return optvals


def check_optvals(outdir, status=True):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Read inputparams
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))
    nls_max = inputparams['optimization']['nls_max']

    # Read previous set of optimization values
    _, alpha_l, alpha_r, alpha, w1, w2, w3 = read_optvals(outdir, it, ls)

    # Linesearch failed if w3 is False
    if w3 is False:
        if status:
            write_status(
                outdir,
                f"FAIL: NOT A DESCENT DIRECTION at it {it:05d} and ls {ls:05d}.")

        return "FAIL"

    # Line search successful
    elif (w1 is True) and (w2 is True):

        # Read initial cost and final cost
        initcost = read_cost(outdir, 0, 0)
        cost = read_cost(outdir, it, ls)

        if status:
            # Write log message
            write_log(outdir,
                      f"iter = {it}, "
                      f"f/fo={cost/initcost:5.4e}, "
                      f"nls = {ls}, wolfe1 = {w1} wolfe2 = {w2}, "
                      f"a={alpha}, al={alpha_l}, ar={alpha_r}")

            write_status(
                outdir,
                f"SUCCESS: it {it:05d} and ls {ls:05d}.")

        return "SUCCESS"

    # Check linesearch
    elif ls == (nls_max-1) and ((w1 is False) or (w2 is False)):
        if status:
            write_status(
                outdir,
                f"FAIL: LS ENDED at it {it:05d} and ls {ls:05d}.")

        return "FAIL"

    # None of it failed add step flag
    else:
        if status:
            write_status(
                outdir,
                f"ADDSTEP: it {it:05d} and ls {ls:05d}.")

        return "ADDSTEP"

    return True


def linesearch(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get the model update and grad
    dm = read_descent(outdir, it, 0)
    g = read_gradient(outdir, it, ls)

    # Compute q descent dot grad
    q = np.sum(dm*g)

    # Read inputparams
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))
    factor = inputparams['optimization']['factor']

    # Write first set of linesearch parameters
    if ls == 0:
        # Set all values to the initial values
        alpha = 1
        alpha_l = 0
        alpha_r = 0
        w1, w2, w3 = True, True, True

    # If linesearch is in progress
    else:
        # Read iteration q
        q_old, _, _, _, _, _, _ = read_optvals(
            outdir, it, 0)

        # Read previous set of optimization values
        _, alpha_l, alpha_r, alpha, _, _, _ = read_optvals(
            outdir, it, ls-1)

        # Read current cost
        cost = read_cost(outdir, it, ls)

        # Read old cost 
        # IMPORTANT HERE is that the iterations cost is checked!
        # The previous linesearch may or may not be a lot higher than the 
        # iteration's costs, and we would accept a cost that did not actually 
        # decrease
        cost_old = read_cost(outdir, it, 0)

        # Safeguard check for inf and nans...
        if np.isnan(cost) or np.isinf(cost):
            # assume we've been too far and reduce step
            alpha_r = alpha
            alpha = (alpha_l + alpha_r)*0.5
            w1, w2, w3 = False, False, True

        else:

            # Compute wolfe conditions
            w1, w2, w3 = wolfe_conditions(
                q_old, q, cost_old, cost, alpha)

            if w3 is False or ((w1 is True) and (w2 is True)):
                pass
            else:
                # Write to optimization values to file
                alpha_l, alpha_r, alpha = update_alpha(
                    w1, w2, alpha_l, alpha_r, alpha, factor=factor)

    # Write to optimization values to file
    write_optvals([q, alpha_l, alpha_r, alpha, w1, w2, w3], outdir, it, ls)
