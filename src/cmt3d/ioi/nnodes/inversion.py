# %%
import os
from nnodes import Node
from .cache import GFM_CACHE
from gf3d.seismograms import GFManager
import cmt3d
import cmt3d.ioi as ioi

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check


def main(node: Node):

    print('Hello')
    node.concurrent = False

    # Get input file
    inputparams = cmt3d.read_yaml(node.inputfile)

    # Get todo events
    event_dir = inputparams['events']

    # Get all events
    events = [os.path.join(event_dir, ev_file)
              for ev_file in os.listdir(event_dir)]

    # Sort the events
    events.sort()

    # Filter by end index
    if node.end_index:
        events = events[:node.end_index]

    # Filter by start index
    if node.start_index:
        events = events[node.start_index:]

    # # Filter by checking which events are done.
    # if not node.redo:

    #     # get out dirs
    #     new_events = []

    #     for _event in events:

    #         # Get name
    #         _eventname = os.path.basename(_event)

    #         try:
    #             # Get inversion directory
    #             out = ioi.optimdir(node.inputfile, _event, get_dirs_only=True)
    #             outdir = out[0]

    #             # Check status
    #             status = ioi.read_status(outdir)

    #             # If events
    #             if 'FINISHED' not in status:
    #                 ioi.reset_iter(outdir)
    #                 ioi.reset_step(outdir)

    #             new_events.append(_event)

    #         except FileNotFoundError:
    #             print(f"No inversion directory for {_eventname}")

    #     # Replace original list of events
    #     events = new_events

    # The massdownloader suggest only 4 threads at a time. So here
    # we are doing 4 simultaneous events with each 1 thread
    if node.max_conc is not None:
        event_chunks = cmt3d.chunkfunc(events, node.max_conc)
    else:
        event_chunks = [events, ]

    # Check if done
    for _i, chunk in enumerate(event_chunks):
        # print(_i, [os.path.basename(_ev) for _ev in chunk])
        node.add(invert_chunk, chunk=chunk, name=",".join(
            os.path.basename(_ev) for _ev in chunk))

# -----------------------------------------------------------------------------


def invert_chunk(node: Node):

    # Each chunk should run concurrently
    node.concurrent = True

    for _event in node.chunk:

        eventname = os.path.basename(_event)
        out = ioi.optimdir(node.inputfile, _event, get_dirs_only=True)
        outdir = out[0]

        node.add(cmtinversion, name=eventname,
                 eventname=eventname,
                 outdir=outdir,
                 eventfile=_event,
                 log=os.path.join(outdir, 'logs'))

# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event


def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')
    concurrent = False

    node.add(preprocess)

    node.add(maybe_invert)


def maybe_invert(node: Node):

    status = ioi.read_status(node.outdir)

    if 'FAIL' in status:
        pass
    else:
        # Weighting
        node.add_mpi(ioi.compute_weights, args=(node.outdir,))

        # Cost, Grad, Hess
        node.add(compute_cgh)

        # Adding the first iteration!
        node.add(iteration)


def preprocess(node: Node):
    node.concurrent = False

    try:
        # Will fail if ITER.txt does not exist
        firstiterflag = ioi.get_iter(node.outdir) == 0

        # Will fail if no status message is present
        status = ioi.read_status(node.outdir)

    except Exception:
        firstiterflag = True
        status = 'NEW'

    if firstiterflag or node.redo or 'FAIL' in status:

        # Create the inversion directory/makesure all things are in place
        node.add(ioi.create_forward_dirs,
                 args=(node.eventfile, node.inputfile),
                 name=f"create-dir", cwd=node.log)

        # Load Green function
        node.add(get_subset)

        # Get data
        # node.add(ioi.get_data, args=(node.outdir,))

        # Forward and frechet modeling
        node.add(forward_frechet_mpi)

        # Process the data and the synthetics
        node.add(process_all, name='process-all', cwd=node.log)

        # Windowing
        node.add(window, name='window')

        # Check Window count
        node.add_mpi(ioi.check_window_count, args=(node.outdir,))

    # if 'FINISHED' in status:
        # Performs iteration


def iteration(node: Node):

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
    node.add(forward_frechet_mpi)
    node.add(process_synt_and_dsdm)
    node.add(compute_cgh)
    node.add(ioi.linesearch, args=(node.outdir,))
    node.add(search_check)


# -------------------
# Forward Computation
def forward_frechet(node: Node):
    node.concurrent = True
    node.add(forward)
    node.add(frechet)


def forward_frechet_mpi(node: Node):

    mnames = ioi.read_model_names(node.outdir)
    counter = 0
    for _mname in mnames:

        if _mname in ioi.Constants.locations:
            counter += 2
        else:
            counter += 1

    # One extra for synthetics.
    counter += 1

    # Forward
    node.add_mpi(ioi.forward_kernel, nprocs=counter,
                 args=(node.outdir,), name='forward-frechet-mpi',
                 cwd=node.log)


def forward(node: Node):
    ioi.forward(node.outdir, GFM_CACHE[node.eventname])


def frechet(node: Node):
    ioi.kernel(node.outdir, GFM_CACHE[node.eventname])


# ------------------
# Get the subset
def get_subset(node: Node):
    inputparams = cmt3d.read_yaml(os.path.join(node.outdir, 'input.yml'))

    backend = node.backend

    if backend == 'mpi' and True:
        node.add(get_subset_mpi)
    else:
        node.add(get_subset_local)


def get_subset_mpi(node: Node):
    node.add_mpi(ioi.create_gfm, nprocs=1, cpus_per_proc=20,
                 args=(node.outdir, node.dbname, node.db_is_local),
                 name='Create_GFM', cwd=node.log)


def get_subset_local(node: Node):

    subsetfilename = os.path.join(node.outdir, 'meta', "subset.h5")

    if os.path.exists(subsetfilename) is False:

        if node.db_is_local:

            from glob import glob

            # Check for files given database path
            db_globstr = os.path.join(node.dbname, '*', '*', '*.h5')

            # Get all files
            db_files = glob(db_globstr)

            # Check if there are any files
            if len(db_files) == 0:
                raise ValueError(f'No files found in {node.dbname} directory. '
                                 'Please check path.')

            else:
                # Get subset
                GFM = GFManager(db_files)
                GFM.load_header_variables()
                cmt = ioi.get_cmt(node.outdir, 0, 0)
                GFM.write_subset_directIO(subsetfilename,
                                          cmt.latitude, cmt.longitude, cmt.depth_in_m/1000.0,
                                          dist_in_km=50.0, NGLL=5,
                                          fortran=False)

        else:

            from gf3d.client import GF3DClient
            gfc = GF3DClient(db=node.dbname)
            cmt = ioi.get_cmt(node.outdir, 0, 0)
            gfc.get_subset(subsetfilename, cmt.latitude, cmt.longitude,
                           cmt.depth_in_m/1000.0, dist_in_km=50.0, NGLL=5,
                           fortran=False)

    print('loading gfm')
    GFM_CACHE[node.eventname] = GFManager(subsetfilename)
    GFM_CACHE[node.eventname].load()
    print('loaded  gfm')

# ----------
# Processing


def process_all(node: Node):
    node.concurrent = True
    node.add(process_data)
    node.add(process_synthetics)
    node.add(process_dsdm)


def process_synt_and_dsdm(node: Node):
    node.concurrent = True
    node.add(process_synthetics)
    node.add(process_dsdm)


def process_data(node: Node):
    node.concurrent = True

    # Get processing parameters
    nproc = node.process_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    for wavetype in processdict.keys():

        if backend == 'multiprocessing':
            node.add_mpi(
                ioi.process_data_wave, args=(node.outdir, wavetype, nproc),
                nprocs=1, cpus_per_proc=nproc,
                name=f'process_data_{wavetype}', cwd=node.log)

        elif backend == 'mpi':

            node.add_mpi(
                ioi.process_data_wave_mpi, args=(node.outdir, wavetype, True),
                nprocs=nproc,
                name=f'process_data_{wavetype}_mpi', cwd=node.log)

        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# Process forward & frechet
def process_synthetics(node: Node):
    node.concurrent = True

    # Get processing parameters
    nproc = node.process_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    for wavetype in processdict.keys():

        if nproc == 1 or backend == 'multiprocessing':
            # Process the normal synthetics
            node.add_mpi(
                ioi.process_synt_wave, args=(node.outdir, wavetype, nproc),
                nprocs=1, cpus_per_proc=nproc,
                name=f'process_synt_{wavetype}',
                cwd=node.log)

        elif nproc > 1 and backend == 'mpi':
            node.add_mpi(
                ioi.process_synt_wave_mpi, args=(node.outdir, wavetype),
                nprocs=nproc, name=f'process_synt_{wavetype}_mpi',
                cwd=node.log)
        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# Process forward & frechet
def process_dsdm(node: Node):

    node.concurrent = True

    # Get processing parameters
    nproc = node.process_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    # Process the frechet derivatives
    NM = len(ioi.read_model_names(node.outdir))

    # Loop over wave types and model parameters
    for wavetype in processdict.keys():

        for _i in range(NM):

            if nproc == 1 or backend == 'multiprocessing':
                node.add_mpi(
                    ioi.process_dsdm_wave, args=(
                        node.outdir, _i, wavetype, True),
                    nprocs=1, cpus_per_proc=nproc,
                    name=f'process_dsdm{_i:05d}_{wavetype}',
                    cwd=node.log)

            elif nproc > 1 and backend == 'mpi':
                node.add_mpi(
                    ioi.process_dsdm_wave_mpi, args=(
                        node.outdir, _i, wavetype, True),
                    nprocs=1, cpus_per_proc=nproc,
                    name=f'process_dsdm{_i:05d}_{wavetype}_mpi',
                    cwd=node.log)

            else:
                raise ValueError(
                    'Double check your backend/multiprocessing setup')


# ------------------
# windowing
def window(node: Node):
    node.concurrent = True

    # Get processing parameters
    nproc = node.window_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    for wavetype in processdict.keys():

        if backend == 'multiprocessing':
            # Process the normal synthetics
            node.add_mpi(ioi.window_wave, args=(node.outdir, wave, nproc),
                         nprocs=1, cpus_per_proc=nproc,
                         name=f'window_{wavetype}',
                         cwd=node.log)

        elif backend == 'mpi':

            node.add_mpi(
                ioi.window_wave_mpi, args=(node.outdir, wavetype, True),
                nprocs=nproc, name=f'window_{wavetype}_mpi',
                cwd=node.log)
        else:

            raise ValueError('Double check your backend/multiprocessing setup')


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
    node.add_mpi(ioi.cost, args=(node.outdir,), name=f"cost", cwd=node.log)


# Gradient
def compute_gradient(node: Node):
    node.add_mpi(ioi.gradient, args=(node.outdir,), name=f"grad", cwd=node.log)


# Hessian
def compute_hessian(node: Node):
    node.add_mpi(ioi.hessian, args=(node.outdir,), name=f"hess", cwd=node.log)


# Descent
def compute_descent(node: Node):
    node.add_mpi(ioi.descent, args=(node.outdir,),
                 name=f"descent", cwd=node.log)

# ----------
# Linesearch


def compute_optvals(node: Node):
    node.add_mpi(ioi.check_optvals, args=(node.outdir,))


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
            node.rm(os.path.join(node.outdir, 'meta', 'subset.h5'))


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
