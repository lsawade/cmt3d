import os
import numpy as np
from .model import read_model, read_model_names, read_scaling
from .gradient import read_gradient
from .hessian import read_hessian
from lwsspy.utils.io import read_yaml_file
from .log import get_iter, get_step, write_log


def damp_gH(m, m0, g, H, damping):

    # Compute model residual
    modelres = m - m0

    # Compute damping factor
    factor = damping * np.trace(H) / m.size

    # Update the Hessian and the gradient
    dH = H + factor * np.diag(np.ones(m.size))
    dg = g + factor * modelres

    return dg, dH


def zero_trace_gH(m, g, H, outdir):

    # Create zero trace array
    mnames = read_model_names(outdir)

    # Get zerotrace arrays
    zero_trace_array = np.array(
        [1.0 if _par in ['m_rr', 'm_tt', 'm_pp'] else 0.0 for _par in mnames]
    )
    zero_trace_index_array = np.where(zero_trace_array == 1.)[0]
    zero_trace_array = np.append(zero_trace_array, 0.0)

    k, l = H.shape
    Hz = np.zeros((k+1, l+1))
    Hz[:-1, :-1] = H
    Hz[:, -1] = zero_trace_array
    Hz[-1, :] = zero_trace_array
    H = Hz
    g = np.append(g, 0.0)
    g[-1] = np.sum(m[zero_trace_index_array])

    return g, H


def write_descent(dm, outdir, it, ls=None):

    # Get graddir
    descdir = os.path.join(outdir, 'desc')

    # Get filename
    if ls is not None:
        fname = f"dm_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"dm_it{it:05d}.npy"

    # Full filename
    file = os.path.join(descdir, fname)

    # Save
    np.save(file, dm)


def read_descent(outdir, it, ls=None):

    # Get graddir
    descdir = os.path.join(outdir, 'desc')

    if ls is not None:
        fname = f"dm_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"dm_it{it:05d}.npy"

    file = os.path.join(descdir, fname)

    return np.load(file)


def descent(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Define the directories
    metadir = os.path.join(outdir, 'meta')

    # Get damping value
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get damping value
    damping = inputparams['optimization']['damping']

    # Get zerotrace flag
    zero_trace = inputparams['zero_trace']

    # Read model, gradient, hessian
    m = read_model(outdir, it, ls)
    g = read_gradient(outdir, it, ls)
    H = read_hessian(outdir, it, ls)

    # Read scaling
    s = read_scaling(outdir)

    # Scaling of the gradient and the Hessian
    g *= s
    H = np.diag(s) @ H @ np.diag(s)

    # Add damping if wanted!
    if damping > 0.0:
        m0 = read_model(outdir, 0, 0)
        g, H = damp_gH(m/s, m0/s, g, H, damping) 

    # Add zerotrace constraint if wanted
    if zero_trace:
        # Since we are acting on the scaled gradient and Hessian, the model
        # needs to be scaled as well.
        g, H = zero_trace_gH(m/s, g, H, outdir)

    # Get direction
    dm = np.linalg.solve(H, -g)

    # If zero trace remove last index
    if zero_trace:
        dm = dm[:-1]

    # Write direction to file
    write_descent(dm*s, outdir, it, 0)

    # Print some logging info
    write_log(
        outdir, f"      d: {np.array2string(dm, max_line_width=int(1e10))}")
