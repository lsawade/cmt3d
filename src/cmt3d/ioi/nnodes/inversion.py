# %%
import os
from nnodes import Node
from .cache import GFM_CACHE
from gf3d.seismograms import GFManager
from cmt3d.source import CMTSource
import cmt3d.ioi as ioi

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check
def main(node: Node):
    node.concurrent = True

    # # Events to be inverted
    # print('Checking events TODO ...')
    # eventfiles = check_events_todo(node.inputfile)

    # # Specific event id(s)
    # eventflag = True if node.eventid is not None else False
    # print('Specfic event(s)?', eventflag)

    # # Maximum inversion flag
    # maxflag = True if node.max_events != 0 else False
    # print('Maximum # of events?', maxflag)

    # # If eventid in files only use the ids
    # if eventflag:
    #     print('Getting specific events...')
    #     nevents = []

    #     eventnames = [
    #         CMTSource.from_CMTSOLUTION_file(_file).eventname
    #         for _file in eventfiles]

    #     # Check whether multiple eventids are requested
    #     if isinstance(node.eventid, list):
    #         eventids = node.eventid
    #     else:
    #         eventids = [node.eventid]

    #     # If id in eventnames, add the eventfile
    #     for _id in eventids:
    #         idx = eventnames.index(_id)
    #         nevents.append(eventfiles[idx])

    #     eventfiles = nevents

    # # If max number of inversion select first X
    # if maxflag:
    #     print('Getting max # of events ...')
    #     eventfiles = eventfiles[:node.max_events]

    # # print list of events if not longer than 10
    # if len(eventfiles) < 11:
    #     for _ev in eventfiles:
    #         print(_ev)

    # Loop over inversions
    # for event in eventfiles:
    #     eventname = CMTSource.from_CMTSOLUTION_file(event).eventname
    #     out = optimdir(node.inputfile, event, get_dirs_only=True)
    #     outdir = out[0]

    scriptdir = '/Users/lucassawade/PDrive/Python/Codes/cmt3d/scripts'
    datadir = os.path.join(scriptdir, 'data')
    subsetdir = os.path.join(datadir, 'subsets')
    eventdir = os.path.join(datadir, 'events')
    cmtfilename = os.path.join(eventdir, 'C201009071613A')
    subsetfilename = os.path.join(eventdir, 'subset.h5')

    eventname = os.path.basename(cmtfilename)
    out = ioi.optimdir(node.inputfile, cmtfilename, get_dirs_only=True)
    outdir = out[0]
    if os.path.exists(outdir):
        node.rm(outdir)

    node.add(cmtinversion, concurrent=False, name=eventname,
             eventname=eventname,
             outdir=outdir,
             eventfile=cmtfilename,
             subsetdir=subsetdir,
             log=os.path.join(outdir, 'logs'))
# -----------------------------------------------------------------------------


# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event
def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')
    node.add(iteration)


# Performs iteration
def iteration(node: Node):

    node.concurrent = False

    try:
        # Will fail if ITER.txt does not exist
        firstiterflag = ioi.get_iter(node.outdir) == 0

    except Exception:
        firstiterflag = True

    if firstiterflag:

        # Create the inversion directory/makesure all things are in place
        node.add(ioi.create_forward_dirs, args=(node.eventfile, node.inputfile),
                 name=f"create-dir", cwd=node.log)

        # Get data
        node.add(ioi.get_data, args=(node.outdir,))

        # Load GF
        # Load Green function
        node.add(get_subset)

        # Forward and frechet modeling
        node.add(forward_frechet)

        # Process the data and the synthetics
        node.add(process_all, name='process-all', cwd=node.log)

        # Windowing
        node.add_mpi(ioi.window, args=(node.outdir,),
                     nprocs=1, cpus_per_proc=4,
                     cwd=node.log)

        # Weighting
        node.add(ioi.compute_weights, args=(node.outdir,))

        # Cost, Grad, Hess
        node.add(compute_cgh)

    # Get descent direction
    node.add(compute_descent)

    # Computes optimization parameters (Wolfe etc.)
    node.add(ioi.linesearch, args=(node.outdir, ))

    # Runs actual linesearch
    node.add(linesearch)

    # Checks whether to add another iteration
    node.add(iteration_check)


# Performs linesearch
def linesearch(node):
    node.add(search_step)


def search_step(node):
    node.add(ioi.update_step, args=(node.outdir,))
    node.add(ioi.update_model, args=(node.outdir,))
    node.add(forward_frechet)
    node.add(process_synthetics)
    node.add(compute_cgh)
    # node.add(compute_descent)
    node.add(ioi.linesearch, args=(node.outdir,))
    node.add(search_check)


# -------------------
# Forward Computation
def forward_frechet(node: Node):
    node.concurrent = True
    node.add(forward)
    node.add(frechet)

def forward(node: Node):
    ioi.forward(node.outdir, GFM_CACHE[node.eventname])

def frechet(node: Node):
    ioi.kernel(node.outdir, GFM_CACHE[node.eventname])


# ------------------
# Get the subset
def get_subset(node: Node):

    subsetfilename = os.path.join(node.subsetdir,
                                  f"{node.eventname}.h5")

    if os.path.exists(subsetfilename) is False:
        from gf3d.client import GF3DClient
        gfc = GF3DClient(db=node.dbname)
        cmt = ioi.get_cmt(node.outdir, 0, 0)
        gfc.get_subset(subsetfilename, cmt.latitude, cmt.longitude,
                       cmt.depth_in_m/1000.0, radius_in_km=50.0, NGLL=5,
                       fortran=False)

    print('loading gfm')
    GFM_CACHE[node.eventname] = GFManager(subsetfilename)
    GFM_CACHE[node.eventname].load()
    print('loaded  gfm')


# ----------
# Processing
def process_all(node: Node):
    node.concurrent = True

    node.add_mpi(
        ioi.process_data, args=(node.outdir,),
        nprocs=1, cpus_per_proc=4,
        name='process_data', cwd=node.log)
    node.add(process_synthetics)


# Process forward & frechet
def process_synthetics(node: Node):
    node.concurrent = True

    # Process the normal synthetics
    node.add_mpi(
        ioi.process_synt, args=(node.outdir, ),
        nprocs=1, cpus_per_proc=4,
        name='process_synt',
        cwd=node.log)

    # Process the frechet derivatives
    NM = len(ioi.read_model_names(node.outdir))

    for _i in range(NM):

        node.add_mpi(
            ioi.process_dsdm, args=(node.outdir, _i),
            nprocs=1, cpus_per_proc=4,
            name=f'process_dsdm{_i:05d}',
            cwd=node.log)

# ------------------
# Updating the model

# update linesearch


def compute_new_model(node: Node):
    ioi.update_model(node.outdir)


# Transer to next iteration
def transfer_mcgh(node: Node):
    node.add(ioi.update_mcgh, args=(node.outdir,),
             name=f"transfer-mcgh", cwd=node.log)


# -------------
# Pre-inversion
def compute_weights(node: Node):
    node.add_mpi(
        ioi.compute_weights,  args=(node.outdir,),
        nprocs=1, cpus_per_proc=4,
        name=f"compute-weights",
        cwd=node.log)


# --------------------------------
# Cost, Gradient, Hessian, Descent
def compute_cgh(node: Node):
    node.concurrent = True
    node.add(compute_cost)
    node.add(compute_gradient)
    node.add(compute_hessian)

# Cost
def compute_cost(node: Node):
    node.add(ioi.cost, args=(node.outdir,), name=f"cost", cwd=node.log)


# Gradient
def compute_gradient(node: Node):
    node.add(ioi.gradient, args=(node.outdir,), name=f"grad", cwd=node.log)


# Hessian
def compute_hessian(node: Node):
    node.add(ioi.hessian, args=(node.outdir,), name=f"hess", cwd=node.log)


# Descent
def compute_descent(node: Node):
    node.add(ioi.descent, args=(node.outdir,), name=f"descent", cwd=node.log)

# ----------
# Linesearch

def compute_optvals(node: Node):
    node.add(ioi.check_optvals, args=(node.outdir,))


# Check whether to add another iteration
def iteration_check(node: Node):

    flag = ioi.check_optvals(node.outdir, status=False)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        if ioi.check_done(node.outdir) is False:
            node.add(ioi.update_iter, args=(node.outdir,))
            node.add(ioi.reset_step, args=(node.outdir,))
            node.parent.parent.add(iteration)
        else:
            node.add(ioi.update_iter, args=(node.outdir,))
            node.add(ioi.reset_step, args=(node.outdir,))

def search_check(node: Node):
    # Check linesearch result.
    flag = ioi.check_optvals(node.outdir)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        # If linesearch was successful, transfer model
        node.add(transfer_mcgh)

    elif flag == "ADDSTEP":
        # Update step
        node.parent.parent.add(search_step)
