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
    node.concurrent = True

    # Get input file
    inputparams = cmt3d.read_yaml(node.inputfile)

    # Get todo events
    event_dir = inputparams['events']

    if node.eventid is not None:
        print("Getting specific event...")
        if "," in node.eventid:
            ids = node.eventid.split(",")
            events = [os.path.join(event_dir, id) for id in ids]
        else:
            events = [os.path.join(event_dir, node.eventid), ]
    else:
        print("Getting all events ...")
        # Get todo events
        event_dir = inputparams['events']

        # Get all events
        events = [os.path.join(event_dir, ev_file)
                  for ev_file in os.listdir(event_dir)]

        # Sort the events
        events.sort()

        # Filter by end index
        if node.end_index:
            print(f"Getting events until idx {node.end_index} ...")
            events = events[:node.end_index]

        # Filter by start index
        if node.start_index:
            print(f"Getting events from idx {node.start_index} ...")
            events = events[node.start_index:]

    node.add(create_dirs_and_subsets, events=events, name='Subset-Events')
    node.add(eventloop, events=events, name='Event-Loop')

# def subworkflow(node: Node):

#     node.concurrent = True
#     node.add(create_dirs_and_subsets, events=node.events, name='Subset-Events')
#     node.add(eventloop, events=node.events, name='Event-Loop')

def create_dirs_and_subsets(node: Node):

    event_chunks = cmt3d.chunkfunc(node.events, node.subset_max_conc)

    node.add(create_dirs_and_subsets_chunk, event_chunks=event_chunks,)

def create_dirs_and_subsets_chunk(node: Node):

    node.concurrent = True

    # Check if done
    if len(node.event_chunks) >= 1:

        # Loop over events in first chunk
        for _i, event in enumerate(node.event_chunks[0]):

            # Get event name and and outfile
            out = ioi.optimdir(node.inputfile, event, get_dirs_only=True)
            outdir = out[0]

            # directory and subset creation stage
            node.add(create_dir_and_subset, event=event, eventfile=event,
                     name=f"Create-Dir-and-Subset-{event}",
                     outdir=outdir, log=os.path.join(outdir, 'logs'))

    # Feed rest of chunk
    if len(node.event_chunks) > 1:
        # here the name has to be the second index (1)!
        # otherwise the name will be overwritten.
        node.parent.add(create_dirs_and_subsets_chunk,
                        event_chunks=node.event_chunks[1:])

        #name=",".join([os.path.basename(event) for event in node.event_chunks[1]])

# -----------------------------------------------------------------------------
def create_dir_and_subset(node: Node):

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

        # Remove files before rerunning the inversion
        # all
        if node.redo:
            node.rm(node.outdir)
        # everything but subset, so that we don't need to reget the subset
        elif node.redo_keep_subset:
            node.add(f"find {node.outdir} ! -name 'subset.h5' -type f -exec rm -f {{}}+")

        # Create the inversion directory/makesure all things are in place
        command = f"cmt3d-ioi create {node.eventfile} {node.inputfile}"
        node.add(command, name=f"Create-Inv-dir", cwd=node.log)

        # Load Green function
        if not os.path.exists(os.path.join(node.outdir, 'meta', 'subset.h5')):
            node.add(get_subset)

        node.add(f'touch {node.outdir}/INIT.txt')


def eventloop(node: Node):

    # Each chunk should run concurrently
    node.concurrent = True

    # Node loop function
    node.add(eventcheck, events=node.events)


def eventcheck(node: Node):

    # Get events
    events = node.events

    # Check if subset exists for file
    idx = []

    print("Events: ", ",".join([os.path.basename(event) for event in events]))

    # Loop over events
    for _i, _event in enumerate(events):

        # Get even name and and outfile
        eventname = os.path.basename(_event)
        out = ioi.optimdir(node.inputfile, _event, get_dirs_only=True)
        outdir = out[0]

        # Check if subset is there and add to list
        if os.path.exists(os.path.join(outdir, 'meta', 'subset.h5')) \
            and os.path.exists(os.path.join(outdir, 'INIT.txt')):
            pass
        else:
            continue

        print(f'Adding event {eventname} to inversion loop.')

        # Save the event index to pop after
        idx.append(_i)

        node.parent.add(cmtinversion,
                 name=eventname, eventname=eventname,
                 outdir=outdir, eventfile=_event,
                 log=os.path.join(outdir, 'logs'))

    # Reverse indeces to pop backwards and not mess with the list
    idx.sort(reverse=True)

    # Pop events
    for _i in idx:
        events.pop(_i)

    # Add more events if there are any left.
    if len(events) > 0:
        child_node = node.add('sleep 60', name="Event-Loop-Resting-Phase")
        child_node.add(update_parent, events=events)

def update_parent(node: Node):
    node.parent.parent.parent.add(eventcheck, events=node.events)

# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event


def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')
    node.concurrent = False

    node.add(preprocess)

    node.add(maybe_invert)


def maybe_invert(node: Node):

    status = ioi.read_status(node.outdir)

    if 'FAIL' in status:
        pass
    else:
        # Weighting
        node.add(f"cmt3d-ioi weights {node.outdir}", name="Weights")

        # Cost, Grad, Hess
        node.add(compute_cgh)

        # Adding the first iteration!
        node.add(iteration)


def preprocess(node: Node):

        # Get data
        # node.add(ioi.get_data, args=(node.outdir,))

        # Forward and frechet modeling
        node.add(forward_frechet_mpi)

        # Process the data and the synthetics
        node.add(process_all, name='process-all', cwd=node.log)

        # Windowing
        node.add(window, name='window')

        # Check Window count
        node.add(f"cmt3d-ioi window count {node.outdir}", cwd=node.log,
                 name='Count-Windows')

    # if 'FINISHED' in status:
        # Performs iteration


def iteration(node: Node):
    node.concurrent = False

    # Get descent direction
    node.add(compute_descent)

    # Computes optimization parameters (Wolfe etc.)
    node.add(f"cmt3d-ioi linesearch {node.outdir}")

    # Runs actual linesearch
    node.add(linesearch)

    # Checks whether to add another iteration
    node.add(iteration_check)


# Performs linesearch
def linesearch(node):
    node.add(search_step)


def search_step(node):
    node.add(f"cmt3d-ioi update-step {node.outdir}", name='Update-Step')
    node.add(f"cmt3d-ioi model update {node.outdir}", name='Update-Model')
    node.add(forward_frechet_mpi)
    node.add(process_synt_and_dsdm)
    node.add(compute_cgh)
    node.add(f"cmt3d-ioi linesearch {node.outdir}", name="Compute-Optvals")
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
    command = f"cmt3d-ioi forward-kernel {node.outdir}"
    node.add_mpi(command, nprocs=counter,
                 name='forward-frechet-mpi', cwd=node.log)


def forward(node: Node):
    ioi.forward(node.outdir, GFM_CACHE[node.eventname])


def frechet(node: Node):
    ioi.kernel(node.outdir, GFM_CACHE[node.eventname])


# ------------------
# Get the subset
def get_subset(node: Node):

    if node.db_is_local:
        flag = "--local"
    else:
        flag = ""
    command = f"cmt3d-ioi subset {flag} {node.outdir} {node.dbname}"

    node.add_mpi(command, nprocs=1, cpus_per_proc=20, name='Getting-Subset',
                 cwd=node.log)

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

    for wave in processdict.keys():

        if backend == 'multiprocessing':
            command = f"cmt3d-ioi process data --nproc={nproc} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                         name=f'process_data_{wave}', cwd=node.log)

        elif backend == 'mpi':
            command = f"cmt3d-ioi process data {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'process_data_{wave}_mpi', cwd=node.log)

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

    for wave in processdict.keys():

        if nproc == 1 or backend == 'multiprocessing':
            # Process the normal synthetics
            command = f"cmt3d-ioi process synt --nproc={nproc} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                         name=f'process_synt_{wave}', cwd=node.log)

        elif nproc > 1 and backend == 'mpi':
            command = f"cmt3d-ioi process synt {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'process_synt_{wave}_mpi', cwd=node.log)
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
    print(node.eventname, node.outdir, processdict.keys())

    # Process the frechet derivatives
    NM = len(ioi.read_model_names(node.outdir))

    # Loop over wave types and model parameters
    for wave in processdict.keys():

        for _i in range(NM):

            if nproc == 1 or backend == 'multiprocessing':
                command = f"cmt3d-ioi process dsdm --nproc={nproc} {node.outdir} {_i} {wave}"
                node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                             name=f'process_dsdm{_i:05d}_{wave}',
                             cwd=node.log)

            elif nproc > 1 and backend == 'mpi':
                command = f"cmt3d-ioi process dsdm {node.outdir} {_i} {wave}"
                node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                             name=f'process_dsdm{_i:05d}_{wave}_mpi',
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

    for wave in processdict.keys():

        if backend == 'multiprocessing':
            # Window using multiprocessing
            command = f"cmt3d-ioi window select --nproc={nproc} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                         name=f'window_{wave}',
                         cwd=node.log)

        elif backend == 'mpi':
            command = f"cmt3d-ioi window select {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'window_{wave}_mpi', cwd=node.log)
        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# ------------------
# Updating the model

# Transer to next iteration
def transfer_mcgh(node: Node):
    command = f"cmt3d-ioi model transfer {node.outdir}"
    node.add(command, name='Model-Transfer')


# -------------
# Pre-inversion
def compute_weights(node: Node):
    command = f"cmt3d-ioi weights {node.outdir}"
    node.add(command, name='Model-Update')

# --------------------------------
# Cost, Gradient, Hessian, Descent


def compute_cgh(node: Node):
    node.concurrent = True
    node.add(compute_cost)
    node.add(compute_gradient)
    node.add(compute_hessian)

# Cost


def compute_cost(node: Node):
    command = f"cmt3d-ioi cost {node.outdir}"
    node.add(command, name=f"cost", cwd=node.log)


# Gradient
def compute_gradient(node: Node):
    command = f"cmt3d-ioi gradient {node.outdir}"
    node.add(command, name=f"grad", cwd=node.log)


# Hessian
def compute_hessian(node: Node):
    command = f"cmt3d-ioi hessian {node.outdir}"
    node.add(command, name=f"hess", cwd=node.log)


# Descent
def compute_descent(node: Node):
    command = f"cmt3d-ioi descent {node.outdir}"
    node.add_mpi(command, name=f"descent", cwd=node.log)

# ----------
# Linesearch


# Check whether to add another iteration
def iteration_check(node: Node):

    flag = ioi.check_optvals(node.outdir, status=False)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        if ioi.check_done(node.outdir) is False:
            node.add
            node.add(
                f"cmt3d-ioi update-iter {node.outdir}", name="Update-iter")
            node.add(f"cmt3d-ioi reset-step {node.outdir}", name="Reset-step")
            node.parent.parent.add(iteration)
        else:
            node.add(f"cmt3d-ioi update-iter {node.outdir}")
            node.add(f"cmt3d-ioi reset-step {node.outdir}")
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
