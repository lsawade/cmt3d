# %%

from curses import meta
import os
import numpy as np
import matplotlib.pyplot as plt
from lwsspy.seismo.source import CMTSource
from lwsspy.utils.io import read_yaml_file

# These are the function that have to be hard-coded
# from lwsspy.gcmt3d.ioi.make_data import make_data
from lwsspy.gcmt3d.ioi.utils import optimdir, createdir, rmdir, \
    prepare_inversion_dir, prepare_model, prepare_stations, \
    prepare_simulation_dirs
from lwsspy.gcmt3d.ioi.get_data import get_data
from lwsspy.gcmt3d.ioi.gaussian2d import g
# from lwsspy.gcmt3d.ioi.data import \
#     write_data, write_data_processed
# from lwsspy.gcmt3d.ioi.metadata import write_metadata, read_metadata
# from lwsspy.gcmt3d.ioi.model import read_model, write_model
from lwsspy.gcmt3d.ioi.forward import update_cmt_synt
from lwsspy.gcmt3d.ioi.kernel import update_cmt_dsdm
from lwsspy.gcmt3d.ioi.cost import cost
from lwsspy.gcmt3d.ioi.model import get_simpars, read_model_names, read_perturbation
from lwsspy.gcmt3d.ioi.gradient import gradient
from lwsspy.gcmt3d.ioi.hessian import hessian
from lwsspy.gcmt3d.ioi.descent import descent
from lwsspy.gcmt3d.ioi.processing import process_data

# These are the functions that are basefunction
from lwsspy.gcmt3d.ioi.log import clear_log
from lwsspy.gcmt3d.ioi.linesearch import \
    linesearch, check_optvals
from lwsspy.gcmt3d.ioi.opt import \
    update_model, update_mcgh, check_done, check_status
from lwsspy.gcmt3d.ioi.plot import plot_cost, plot_model, plot_hessians

# %%

# Little functions to create and removed
# problem_module = "/Users/lucassawade/lwsspy/lwsspy/src/lwsspy/gcmt3d/io/problem/__init__.py"

# problem = import_problem(problem_module)
# %%

# Inversion parameters
# damping = 0.01
# stopping_criterion = 1e-5
# stopping_criterion_model = 0.001
# stopping_criterion_cost_change = 1e-3
# niter_max = 10
# nls_max = 10
# alpha = 1.0
# perc = 0.1
it0 = 0


# %%
__file__ = '/home/lsawade/lwsspy/lwsspy.gcmt3d/src/lwsspy/gcmt3d/ioi/optimization.py'
cmtfilename = os.path.join(os.path.dirname(__file__), 'C122604A')
inputfilename = os.path.join(os.path.dirname(__file__), 'input.yml')
paramfilename = os.path.join(os.path.dirname(__file__), 'process.yml')


#  %% Remove and create optimdir

outdir, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    optimdir(inputfilename, cmtfilename, get_dirs_only=True)

# %%
# if os.path.exists(outdir):
#     rmdir(outdir)

# %%


outdir, modldir, metadir, datadir, simudir, ssyndir, sfredir, syntdir, \
    frecdir, costdir, graddir, hessdir, descdir, optdir = \
    optimdir(inputfilename, cmtfilename)

prepare_inversion_dir(cmtfilename, outdir, metadir, inputfilename)


# %%
# Create local parameter files and copy the model files
prepare_inversion_dir(cmtfilename, outdir, metadir, inputfilename)

# %%
# Download the data
get_data(outdir)

# %%
prepare_model(outdir, metadir, modldir)

# %%
prepare_stations(metadir)

# %%
process_data(outdir)
# Nparams = int(read_model(modldir, 0, 0).size)

# %%
# Preparing the simulation directory
prepare_simulation_dirs(outdir, ssyndir, sfredir, metadir, simudir)

# %%

# Write sources to the simulation directories
update_cmt_synt(modldir, metadir, ssyndir, it0, 0)


# %%


def printsimInfo(metadir):
    model_names = read_model_names(outdir)
    simpars = get_simpars(metadir)

    # Print model parameter info
    print("Model parameter names:")
    print("----------------------")
    for _i, _mname in enumerate(model_names):
        print(f"{_i:>5d}: {_mname}")

    # Print simulation info
    print("\nParameters to be simulated: ")
    print("----------------------")
    for _i, _mname in enumerate(model_names):
        if _i in simpars:
            print(f"{_i:>5d}: {_mname}")


printsimInfo(metadir)

# %%
for it in range(it0, niter_max):

    print(f"Iteration: {it:05d}")
    print("----------------")

    # Reset the linesearch iterator
    ls = 0

    if it == 0:

        # Forward modeling
        forward(modldir, metadir, syntdir, it, ls)

        # Kernel computation
        for _i in range(Nparams):
            frechet(_i, modldir, metadir, frecdir, it, ls)

        # Computing the cost the gradient and the Hessian
        cost(datadir, syntdir, costdir, it, ls)
        gradient(
            modldir, graddir, syntdir, datadir, frecdir, it, ls)
        hessian(modldir, hessdir, frecdir, it, ls)

    # Get descent direction
    descent(outdir, it, ls)

    # First set of optimization values only computes the initial q and
    # sets alpha to 1
    linesearch(optdir, descdir, graddir, costdir, it, ls)

    for ls in range(1, nls_max):

        print(f"  Linesearch: {ls:05d}")
        print("  -----------------")

        # Update the model
        update_model(modldir, descdir, optdir, it, ls - 1)

        # Forward modeling
        forward(modldir, metadir, syntdir, it, ls)

        # Kernel computation
        for _i in range(Nparams):
            frechet(_i, modldir, metadir, frecdir, it, ls)

        # Computing the cost the gradient and the Hessian
        cost(datadir, syntdir, costdir, it, ls)
        gradient(
            modldir, graddir, syntdir, datadir, frecdir, it, ls)
        hessian(modldir, hessdir, frecdir, it, ls)

        # Get descent direction
        descent(modldir, graddir, hessdir,
                descdir, outdir, damping, it, ls)

        # Compute optimization values
        linesearch(optdir, descdir, graddir, costdir, it, ls)

        # Check optimization values
        if not check_optvals(optdir, outdir, costdir, it, ls, nls_max):
            break

    # Check optimization status
    if not check_status(outdir):
        break
    else:
        print("\n-------------------------------\n")

    # Update model
    # If the linesearch is successful reassign the model grad etc for the next
    # iteration. The final iteration of the linesearch is the first grad of the
    # next iteration
    update_mcgh(modldir, costdir, graddir, hessdir, it, ls)

    # With the new model check wether the new cost satisfies the stopping conditions
    if check_done(
            optdir, modldir, costdir, descdir, outdir, it, ls,
            stopping_criterion=stopping_criterion,
            stopping_criterion_model=stopping_criterion_model,
            stopping_criterion_cost_change=stopping_criterion_cost_change):
        break


plot_cost(outdir)
plot_model(outdir)
plot_hessians(outdir)
plt.show(block=True)
