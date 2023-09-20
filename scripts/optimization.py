# %%
import os
import cmt3d
import cmt3d.ioi as ioi
from gf3d.seismograms import GFManager
from gf3d.client import GF3DClient

# %%


# %%
# Define the inversion directory
__file__ = '/Users/lucassawade/PDrive/Python/Codes/cmt3d/scripts/optimization.py'
datadir = os.path.join(os.path.dirname(__file__), 'data')
cmtfilename = os.path.join(datadir, 'C201801230931A')
inputfilename = os.path.join(datadir, 'input.yml')
paramfilename = os.path.join(datadir, 'process.yml')
subsetfilename = os.path.join(datadir, 'subset.h5')
# %%

# Load CMT solution
cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)

# %%
# gfc = GF3DClient(db='glad-m25')
# gfc.get_subset(subsetfilename, cmtsource.latitude, cmtsource.longitude,
#                cmtsource.depth_in_m/1000.0, radius_in_km=50.0, NGLL=5,
#                fortran=False)



# %% Remove and create optimdir

outdir, _, _, _, _, _, _, _, _, _, _, _, _, _ = \
    ioi.optimdir(inputfilename, cmtfilename, get_dirs_only=True)

# %%
# if os.path.exists(outdir):
#     rmdir(outdir)

# %%
# Make forward dirs
ioi.create_forward_dirs(cmtfilename, inputfilename)

optimization_dict = cmt3d.read_yaml(inputfilename)['optimization']

# %%
# Download the data
# ioi.get_data(outdir)

# %%
ioi.prepare_model(outdir)

# %%
ioi.prepare_stations(outdir)

# %%
ioi.process_data(outdir)
# Nparams = int(read_model(modldir, 0, 0).size)


# %%

# Load Green function
gfm = GFManager(subsetfilename)
gfm.load()

# %%
# Write sources to the simulation directories
ioi.forward(outdir, gfm)
ioi.process_synt(outdir)

# %%
ioi.window(outdir)

# %%
ioi.compute_weights(outdir)

# %%
ioi.kernel(outdir, gfm)

# %%
Nparams = len(ioi.read_model_names(outdir))
for nm in range(Nparams):
    ioi.process_dsdm(outdir, nm)


# %%


def printsimInfo(outdir):
    model_names = ioi.read_model_names(outdir)
    simpars = ioi.get_simpars(outdir)

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


printsimInfo(outdir)

# %%
it0 = 0
ioi.reset_step(outdir)
ioi.reset_iter(outdir)

niter_max = optimization_dict['niter_max']
nls_max = optimization_dict['nls_max']

# Computing the cost the gradient and the Hessian
ioi.cost(outdir)
ioi.gradient(outdir)
ioi.hessian(outdir)


# %%
for it in range(it0, niter_max):

    print(f"Iteration: {it:05d}")
    print("----------------")

    # Get descent direction
    ioi.descent(outdir)

    # Compute optimization values
    ioi.linesearch(outdir)

    for ls in range(1, nls_max):

        print(f"  Linesearch: {ls:05d}")
        print("  -----------------")

        ioi.update_step(outdir)

        ioi.update_model(outdir)

        # Forward modeling
        ioi.forward(outdir, gfm)
        ioi.process_synt(outdir)

        # Kernel computation
        ioi.kernel(outdir, gfm)
        for nm in range(Nparams):
            ioi.process_dsdm(outdir, nm)

        # CGH
        ioi.cost(outdir)
        ioi.gradient(outdir)
        ioi.hessian(outdir)

        # Descent
        # ioi.descent(outdir)

        # Compute optimization values
        ioi.linesearch(outdir)

        # Check the optimation values
        flag = ioi.check_optvals(outdir)

        if flag == "FAIL":
            break

        elif flag == "SUCCESS":
            # If linesearch was successful, transfer model
            ioi.update_mcgh(outdir)
            break

        # Check optimization values
        elif flag == 'ADDSTEP':
            pass

        else:
            raise ValueError(f"Unknown flag: {flag}")

    # Check optimization status
    flag = ioi.check_optvals(outdir, status=False)

    if flag == "FAIL":
        break

    elif flag == "SUCCESS":

        if ioi.check_done(outdir):
            ioi.update_iter(outdir)
            ioi.reset_step(outdir)
            break
        else:
            ioi.update_iter(outdir)
            ioi.reset_step(outdir)
print("\n-------------------------------\n")
