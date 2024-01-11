# %%
import os
from nnodes import Node
from nnodes.job import Slurm
from .cache import GFM_CACHE
import cmt3d
import cmt3d.ioi as ioi
import asyncio

# ----------------------------- MAIN NODE -------------------------------------
# Loops over events: TODO smarter event check


def main(node: Node):

    print('Hello')
    node.concurrent = True

    # Get input file
    inputparams = cmt3d.read_yaml(node.inputfile)

    # Get todo events
    event_dir = inputparams['events']
    db_dir = inputparams['database']

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

    # if redo, remove INIT files
    if node.redo or node.redo_keep_subset:
        # Remove INIT files
        for event in events:
            name = os.path.basename(event)
            initpath = os.path.join(db_dir, name, 'INIT.txt')
            print(f"Removing {initpath}")
            node.rm(os.path.join(initpath))

    node.add(create_dirs_and_subsets, events=events, name='Subset-Events', concurrent=False)
    node.add(eventloop, events=events, name='Event-Loop', concurrent=True)

# def subworkflow(node: Node):

#     node.concurrent = True
#     node.add(create_dirs_and_subsets, events=node.events, name='Subset-Events')
#     node.add(eventloop, events=node.events, name='Event-Loop')

def create_dirs_and_subsets(node: Node):

    event_chunks = cmt3d.chunkfunc(node.events, node.subset_max_conc)

    node.add(create_dirs_and_subsets_chunk, event_chunks=event_chunks,
             concurrent=True)

def create_dirs_and_subsets_chunk(node: Node):

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
                     outdir=outdir, log=os.path.join(outdir, 'logs'),
                     concurrent=False)

    # Feed rest of chunk
    if len(node.event_chunks) > 1:
        # here the name has to be the second index (1)!
        # otherwise the name will be overwritten.
        node.parent.add(create_dirs_and_subsets_chunk,
                        event_chunks=node.event_chunks[1:],
                        concurrent=True)

        #name=",".join([os.path.basename(event) for event in node.event_chunks[1]])

# -----------------------------------------------------------------------------
def create_dir_and_subset(node: Node):

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
        if node.redo_keep_subset:
            node.add(f"find {node.outdir} ! -name 'subset.h5' -type f -exec rm -f {{}} +")
        else:
            node.rm(node.outdir)

        # Create the inversion directory/makesure all things are in place
        command = f"cmt3d-ioi create {node.eventfile} {node.inputfile}"
        node.add_mpi(command, name=f"Create-Inv-dir", cwd=node.log,
                     exec_args={Slurm: '-N1 --time=5'},
                     nprocs=1, cpus_per_proc=3, timeout=60*6, retry=3)

        # Load Green function
        if not os.path.exists(os.path.join(node.outdir, 'meta', 'subset.h5')):
            node.add(get_subset)

        node.add(f'touch {node.outdir}/INIT.txt',
                 name=f'INIT-{os.path.basename(node.outdir)}',
                 timeout=60*6, retry=3)

def stop_event_after_fail(node):
    """Stops the parent inversion node after failing,
    and removes the event from the list of events to be run."""
    pass
    # if node.parent.name == 'eventloop':
    #     node.

def eventloop(node: Node):

    # Each chunk should run concurrently
    node.concurrent = True

    # Node loop function
    node.add(eventcheck, events=node.events, concurrent = False, counter=0,
             name=f'Event-Check-{0:04d}')


def eventcheck(node: Node):

    # Get events
    events = node.events

    # Check if subset exists for file
    idx = []

    # Loop over events
    for _i, _event in enumerate(events):

        # Check how many nodes are in the event loop that are to be run
        N_notdone = sum([_n.done==False for _n in node.parent])

        # If number larger than max, break
        if N_notdone + 1 >= node.inversion_max_conc:
            break

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
                 log=os.path.join(outdir, 'logs'),
                 concurrent=False,
                 it=0, step=0)

    # Reverse indeces to pop backwards and not mess with the list
    idx.sort(reverse=True)

    # Pop events
    for _i in idx:
        events.pop(_i)

    # Add more events if there are any left.
    if len(events) > 0:
        print("Left events: ", ",".join([os.path.basename(event) for event in events]))

        # Add sleep cycle
        child_node = node.add_mpi(
            f'sleep {node.eventchecksleep} && echo "done sleeping"',
            name=f"Event-Loop-Resting-Phase-#{node.counter:04d}",
            exec_args={Slurm: f'-N1 --time={int(node.eventchecksleep/60)+1}'},
            nprocs=1, cwd='./logs',
            timeout=node.eventchecksleep+2*60, retry=3)

        # Add parent node to run another event check
        child_node.add(update_parent, events=events, counter=node.counter+1)


def update_parent(node: Node):
    node.parent.parent.parent.add(eventcheck, events=node.events,
                                  counter=node.counter,
                                  name=f'Event-Check-{node.counter:04d}')

# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event


def cmtinversion(node: Node):
    # node.write(20 * "=", mode='a')

    # Read model and model name
    mnames = ioi.read_model_names(node.outdir)

    # Get number of processes required for kernel evaluation
    fw_kernel_sim_nproc = 0
    for _mname in mnames:
        if _mname in ioi.Constants.locations:
            fw_kernel_sim_nproc += 2
        else:
            fw_kernel_sim_nproc += 1

    # Get the frechet derivatives
    NM = len(mnames)

    # Get parameters
    NW = len(cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml')).keys())

    # Processes required for forward and kernel processing
    kernel_proc_nproc = NW*(NM+1)

    # Get number of processes required for processing
    if kernel_proc_nproc > fw_kernel_sim_nproc:
        step_nproc = kernel_proc_nproc
    else:
        step_nproc = fw_kernel_sim_nproc

    node.add(preprocess, concurrent=False, step_nproc=step_nproc)

    node.add(maybe_invert, concurrent=False, step_nproc=step_nproc)


def maybe_invert(node: Node):

    status = ioi.read_status(node.outdir)

    if 'FAIL' in status:
        pass
    else:
        # Weighting
        node.add_mpi(f"cmt3d-ioi weights {node.outdir}", name="Weights",
                     exec_args={Slurm: '-N1 --time=5'},
                     nprocs=1, cwd=node.log, timeout=60*6, retry=3)

        # Cost, Grad, Hess
        node.add(compute_cgh, concurrent=True)
        # node.add_mpi(f'cmt3d-ioi step-mfpcghc --it {node.it} --ls {node.step} --verbose --cgh-only {node.outdir}',
        #              name=f'Cost-Grad-Hess-#{node.it:03d}-ls#{node.step:03d}',
        #              nprocs=3, cwd=node.log, timeout=60*6, retry=3,
        #              exec_args={Slurm: '-N1 --time=5'},)

        # Adding the first iteration!
        node.add(iteration, concurrent=False, name=f'Iteration-#{node.it:03d}')


def preprocess(node: Node):

        # Get data
        # node.add(ioi.get_data, args=(node.outdir,))

        # Forward model and process data and synthetics.
        node.add(forward_frechet_mpi)
        node.add(process_all, name='Process-All',
                 cwd=node.log, concurrent=True)

        # Windowing
        node.add(window, name='window', concurrent=True)

        # Check Window count
        node.add_mpi(f"cmt3d-ioi window count {node.outdir}", cwd=node.log,
                     exec_args={Slurm: '-N1 --time=4'},
                     nprocs=1, timeout=60*5, retry=3, name='Count-Windows')

    # if 'FINISHED' in status:
        # Performs iteration


def iteration(node: Node):

    # Get descent direction
    node.add(compute_descent)

    # Computes optimization parameters (Wolfe etc.)
    node.add_mpi(f"cmt3d-ioi linesearch --it {node.it} --ls {node.step} {node.outdir}",
                 exec_args={Slurm: '-N1 --time=1'},
                 name="Compute-Optvals", nprocs=1, cwd=node.log, timeout=120, retry=3)

    # Runs actual linesearch
    node.add(linesearch, concurrent=False)

    # Checks whether to add another iteration
    node.add(iteration_check, concurrent=False)


# Performs linesearch
def linesearch(node):
    node.add(search_step, concurrent=False, name=f'Line-Search-#{node.step:03d}')

def search_step(node):
    # Update the overall inversion parameters (Note that it's necessary)
    #    lines  iter
    node.parent.parent.step = node.step + 1

    # node.add(f"cmt3d-ioi update-step --it {node}{node.outdir}", name='Update-Step')
    node.add(f"cmt3d-ioi model update --it {node.it} --ls {node.step} {node.outdir}", name='Update-Model')
    node.add(forward_frechet_mpi)
    node.add(process_synt_and_dsdm)
    node.add(compute_cgh)
    node.add(f"cmt3d-ioi linesearch --it {node.it} --ls {node.step} {node.outdir}",
             name="Compute-Optvals")
    # node.add(search_check)

    # Forward modeling and processing
    # node.add_mpi(f'cmt3d-ioi step-mfpcghc --it {node.it} --ls {node.step} --verbose {node.outdir}',
    #              nprocs=node.step_nproc, cwd=node.log, timeout=60*6, retry=3,
    #              name=f'Step-MFPCGHC-MPI-it#{node.it:03d}-ls#{node.step:03d}',
    #              exec_args={Slurm: '--nodes=1-4 --time=5'},)

    node.add(search_check, concurrent=False)

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
    command = f"cmt3d-ioi forward-kernel --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, nprocs=counter,
                 name='forward-frechet-mpi', cwd=node.log,
                 exec_args={Slurm: '-N1 --time=5'},
                 timeout=60*6, retry=3)




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
                 exec_args={Slurm: '-N1 --time=10'},
                 cwd=node.log, retry=3, timeout=60*12)

# ----------
# Processing


def process_all(node: Node):
    node.add(process_data, concurrent=True)
    # node.add_mpi(f'cmt3d-ioi step-mfpcghc --it {node.it} --ls {node.step} --verbose --fw-only {node.outdir}',
    #                  nprocs=node.step_nproc, cwd=node.log, timeout=60*6, retry=3,
    #                  name=f'Step-MFPCGHC-MPI-it#{node.it:03d}-ls#{node.step:03d}',
    #                  exec_args={Slurm: '--nodes=1-4 --time=5'},)
    node.add(process_synthetics, concurrent=True)
    node.add(process_dsdm, concurrent=True)


def process_synt_and_dsdm(node: Node):
    node.add(process_synthetics, concurrent=True)
    node.add(process_dsdm, concurrent=True)


def process_data(node: Node):

    # Get processing parameters
    nproc = node.process_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    for wave in processdict.keys():

        if backend == 'multiprocessing':
            command = f"cmt3d-ioi process data --nproc={nproc} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                         name=f'process_data_{wave}', cwd=node.log,
                         timeout=60*12, retry=3,
                         exec_args={Slurm: '-N1 --time=10'})

        elif backend == 'mpi':
            command = f"cmt3d-ioi process data {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'process_data_{wave}_mpi', cwd=node.log,
                         timeout=60*12, retry=3,
                         exec_args={Slurm: '-N1 --time=10'})

        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# Process forward & frechet
def process_synthetics(node: Node):

    # Get processing parameters
    nproc = node.process_nproc
    backend = node.backend

    # Get parameters
    processdict = cmt3d.read_yaml(os.path.join(node.outdir, 'process.yml'))

    for wave in processdict.keys():

        if nproc == 1 or backend == 'multiprocessing':
            # Process the normal synthetics
            command = f"cmt3d-ioi process synt --it {node.it} --ls {node.step} --nproc={nproc} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                         name=f'process_synt_{wave}', cwd=node.log,
                         timeout=60*6, retry=3,
                         exec_args={Slurm: '-N1 --time=5'})

        elif nproc > 1 and backend == 'mpi':
            command = f"cmt3d-ioi process synt  --it {node.it} --ls {node.step} {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'process_synt_{wave}_mpi', cwd=node.log,
                         timeout=60*6, retry=3,
                         exec_args={Slurm: '-N1 --time=5'})
        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# Process forward & frechet
def process_dsdm(node: Node):

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
                command = f"cmt3d-ioi process dsdm  --it {node.it} --ls {node.step} --nproc={nproc} {node.outdir} {_i} {wave}"
                node.add_mpi(command, nprocs=1, cpus_per_proc=nproc,
                             name=f'process_dsdm{_i:05d}_{wave}',
                             cwd=node.log, timeout=60*6, retry=3,
                             exec_args={Slurm: '-N1 --time=5'})

            elif nproc > 1 and backend == 'mpi':
                command = f"cmt3d-ioi process dsdm --it {node.it} --ls {node.step} {node.outdir} {_i} {wave}"
                node.add_mpi(command, nprocs=nproc, cpus_per_proc=1,
                             name=f'process_dsdm{_i:05d}_{wave}_mpi',
                             cwd=node.log, timeout=60*6, retry=3,
                             exec_args={Slurm: '-N1 --time=5'})

            else:
                raise ValueError(
                    'Double check your backend/multiprocessing setup')


# ------------------
# windowing
def window(node: Node):

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
                         cwd=node.log, timeout=60*12, retry=3,
                         exec_args={Slurm: '-N1 --time=10'})

        elif backend == 'mpi':
            command = f"cmt3d-ioi window select {node.outdir} {wave}"
            node.add_mpi(command, nprocs=nproc,
                         name=f'window_{wave}_mpi', cwd=node.log,
                         timeout=60*12, retry=3,
                         exec_args={Slurm: '-N1 --time=10'})
        else:
            raise ValueError('Double check your backend/multiprocessing setup')


# ------------------
# Updating the model

# Transer to next iteration
def transfer_mcgh(node: Node):
    command = f"cmt3d-ioi model transfer --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, name='Model-Transfer',
                 nprocs=1, cwd=node.log, timeout=120, retry=3,
                 exec_args={Slurm: '-N1 --time=1'})


# -------------
# Pre-inversion
def compute_weights(node: Node):
    command = f"cmt3d-ioi weights {node.outdir}"
    node.add_mpi(command, name='Compute-Weights',
                 nprocs=1, cwd=node.log, timeout=120, retry=3,
                 exec_args={Slurm: '-N1 --time=1'})

# --------------------------------
# Cost, Gradient, Hessian, Descent


def compute_cgh(node: Node):
    node.add(compute_cost, retry=3)
    node.add(compute_gradient, retry=3)
    node.add(compute_hessian, retry=3)

# Cost


def compute_cost(node: Node):
    command = f"cmt3d-ioi cost --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, name=f"cost", cwd=node.log, retry=3,
                 nprocs=1, timeout=60*12,
                 exec_args={Slurm: '-N1 --time=10'})


# Gradient
def compute_gradient(node: Node):
    command = f"cmt3d-ioi gradient --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, name=f"grad", cwd=node.log,
                 nprocs=1,timeout=60*12, retry=3,
                 exec_args={Slurm: '-N1 --time=10'})


# Hessian
def compute_hessian(node: Node):
    command = f"cmt3d-ioi hessian --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, name=f"hess", cwd=node.log,
                 nprocs=1, timeout=60*12, retry=3,
                 exec_args={Slurm: '-N1 --time=10'})


# Descent
def compute_descent(node: Node):
    command = f"cmt3d-ioi descent  --it {node.it} --ls {node.step} {node.outdir}"
    node.add_mpi(command, name=f"descent", cwd=node.log,
                 nprocs=1, timeout=60*6, retry=3,
                 exec_args={Slurm: '-N1 --time=5'})

# ----------
# Linesearch


# Check whether to add another iteration
def iteration_check(node: Node):

    # Check linesearch result.
    flag = ioi.check_optvals(node.outdir, status=False, it=node.it, ls=node.step)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        if ioi.check_done(node.outdir) is False:
            node.parent.parent.add(iteration, it=node.it + 1, step=0,
                                   name=f'Iteration-#{node.it+1:03d}')
        else:
            node.rm(os.path.join(node.outdir, 'meta', 'subset.h5'))


def search_check(node: Node):

    # Check linesearch result.
    flag = ioi.check_optvals(node.outdir, status=True, it=node.it, ls=node.step)

    if flag == "FAIL":
        pass

    elif flag == "SUCCESS":
        # If linesearch was successful, transfer model
        node.add(transfer_mcgh, retry=3)

    elif flag == "ADDSTEP":
        # Update step
        node.parent.parent.add(search_step, concurrent=False,
                               name=f'Line-Search-#{node.step:03d}')



