# %%
from multiprocessing import Event
import os
import toml
import typing as tp
from nnodes import Node
from lwsspy.seismo.source import CMTSource
from lwsspy.gcmt3d.ioi.functions.utils import optimdir, wcreate_forward_dirs
from lwsspy.gcmt3d.ioi.functions.forward import update_cmt_synt
from lwsspy.gcmt3d.ioi.functions.kernel import update_cmt_dsdm
from lwsspy.gcmt3d.ioi.functions.processing import process_data, window, process_synt, wprocess_dsdm
from lwsspy.gcmt3d.ioi.functions.model import get_simpars, read_model_names
from lwsspy.gcmt3d.ioi.functions.weighting import compute_weights as compute_weights_func
from lwsspy.gcmt3d.ioi.functions.cost import cost
from lwsspy.gcmt3d.ioi.functions.descent import descent
from lwsspy.gcmt3d.ioi.functions.gradient import gradient
from lwsspy.gcmt3d.ioi.functions.hessian import hessian
from lwsspy.gcmt3d.ioi.functions.linesearch import linesearch as get_optvals, check_optvals
from lwsspy.gcmt3d.ioi.functions.opt import check_done, update_model, update_mcgh
from lwsspy.gcmt3d.ioi.functions.log import update_iter, update_step, reset_step, get_iter
from lwsspy.gcmt3d.ioi.functions.events import check_events_todo

from radical.entk import Pipeline, Manager, Task, Stage


class EventConfig:
    outdir: str
    file: str
    name: str
    input: str
    specfemp: dict
    conda_init: str
    simpars: tp.List[int]
    NM: int

# ----------------------------- MAIN NODE -------------------------------------


def generate_pipelines(jobconfigfile: str):

    # Reading the config
    cfg = toml.load(jobconfigfile)
    conda_init = cfg['conda_init']
    specfemp = cfg['specfem']

    # Read input file
    inputfile = cfg['root']['inputfile']

    # Events to be inverted
    print('Checking events TODO ...')
    eventfiles = check_events_todo(inputfile)

    # Specific event id(s)
    eventflag = True if cfg['root']['eventid'] is not None else False
    print('Specfic event(s)?', eventflag)

    # Maximum download flag
    maxflag = True if cfg['root']['max_events'] != 0 else False
    print('Maximum # of events?', eventflag)

    # If eventid in files only use the ids
    if eventflag:
        print('Getting specific events...')
        nevents = []

        eventnames = [
            CMTSource.from_CMTSOLUTION_file(_file).eventname
            for _file in eventfiles]

        # Check whether multiple eventids are requested
        if isinstance(cfg['root']['eventid'], list):
            eventids = cfg['root']['eventid']
        else:
            eventids = [cfg['root']['eventid']]

        # If id in eventnames, add the eventfile
        for _id in eventids:
            idx = eventnames.index(_id)
            nevents.append(eventfiles[idx])

        eventfiles = nevents

    # If max number of inversion select first X
    if maxflag:
        print('Getting max # of events ...')
        eventfiles = eventfiles[: cfg['root']['max_events']]

    # print list of events if not longer than 10
    if len(eventfiles) < 11:
        for _ev in eventfiles:
            print(_ev)

    # Get simpars and NM
    simpars = get_simpars_from_input(inputfile)
    NM = get_NM_from_input(inputfile)
    pipelines = []

    # Loop over inversions
    for eventfile in eventfiles:
        eventname = CMTSource.from_CMTSOLUTION_file(eventfile).eventname
        out = optimdir(inputfile, eventfile, get_dirs_only=True)
        outdir = out[0]

        # Event specfic workflow config
        E = EventConfig()
        E.outdir = outdir
        E.input = inputfile
        E.file = eventfile
        E.name = eventname
        E.conda_init = conda_init
        E.specfemp = specfemp
        E.simpars = simpars
        E.NM = NM

        pipelines.append(cmtinversion(E))


# -----------------------------------------------------------------------------


# ---------------------------- CMTINVERSION -----------------------------------

# Performs inversion for a single event
def cmtinversion(E: EventConfig):

    # Giving the pipeline the eventdir attribute so that the post func has
    # access!
    p = Pipeline()
    p.__setattr__('eventdir', E.outdir)

    try:
        # Will fail if ITER.txt does not exist
        firstiterflag = get_iter(E.outdir) == 0

    except Exception:
        firstiterflag = True

    if firstiterflag:
        # Before computing a descent direction
        p.add_stages(preinv_stages(E))

    # node.write(20 * "=", mode='a')
    iteration(E, p)

    return p


def preinv_stages(E: EventConfig) -> tp.List[Stage]:

    stages = []

    # Create the inversion directory/make sure all things are in place
    stages.append(createdirs(E))

    # Forward and frechet modeling
    stages.append(forwardfrechet(E))

    # Process the data and the synthetics
    stages.append(process_all(E))

    # Windowing
    stages.append(window(E))

    # Weighting
    stages.append(compute_weights(E))

    # Cost, Grad, Hess
    stages.append(cgh(E))

    return stages


# Performs iteration
def iteration(E: EventConfig, p: Pipeline) -> None:

    # Initialize empty stages
    stages = []

    # Compute descent
    stages.append(descent(E))

    # Compute optimization values
    stages.append(compute_optvals(E))

    # Add stages to the pipeline
    p.add_stages(stages)

    # Linesearch
    linesearch(E, p)


def linesearch(E: EventConfig, p: Pipeline) -> None:

    def check_step_and_iter():

        # Check linesearch result.
        flag = check_optvals(E.outdir)

        if flag == "FAIL":
            pass

        elif flag == "SUCCESS":
            # If linesearch was successful, transfer model
            update_mcgh(E.outdir)

            if check_done(E.outdir) is False:
                update_iter(E.outdir)
                reset_step(E.outdir)
                p.add_stages(iteration(E, p))
            else:
                update_iter(E.outdir)
                reset_step(E.outdir)

        elif flag == "ADDSTEP":
            # Update step
            p.add_stages(linesearch(E, p))

    stages = []

    # Update model using descent, and linesearch step length
    stages.append(compute_new_model(E))
    stages.append(forwardfrechet(E))
    stages.append(process_all_synt(E))

    stages.append(cgh(E))
    stages.append(descent(E))

    # Compute optimization values
    S = compute_optvals(E)
    S.post_exec = check_step_and_iter
    stages.append(S)

    p.add_stages(stages)


def compute_new_model(E: EventConfig):

    # Create Dirs Task
    T = Task()
    T.name = f"{E.name}-T-Compute-New-Model"
    T.pre_exec = [
        E.conda_init,
        f'gcmt3d-update-step {E.outdir}'
    ]
    T.executable = 'gcmt3d-compute-new-model'
    T.arguments = [f'{E.outdir}']
    =
    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-New-Model"
    S.add_tasks([T])

    return S


# -------- PREINV -------------------------------------------------------------
def createdirs(E: EventConfig) -> Stage:

    # Create Dirs Task
    T = Task()
    T.name = f"{E.name}.T.CreateDirs"
    T.executable = 'gcmt3d-create-inv-dir'
    T.arguments = [f'{E.file}', f'{E.input}']
    T.pre_exec = [
        E.conda_init
    ]

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-CreateDirs"
    S.add_tasks([T])

    return S


def forwardfrechet(E: EventConfig) -> Stage:

    # Create Stage
    S = Stage()
    S.name = f"{E.name}.S.ForwardFrechet"
    S.add_tasks(forward(E))

    S.add_tasks(frechet(E))

    return S


def forward(E: EventConfig) -> tp.List[Task]:

    # Create forward modelling task
    T = Task()
    T.name = f"{E.name}.T.Forward"
    T.executable = 'bin/xspecfem3D'
    T.sandbox = os.path.join(E.outdir, 'simu', 'synt')
    T.pre_exec = [
        E.conda_init,
        f'module load {E.specfemp["modules"]}',
        'gcmt3d-update-cmt-synt'
    ]
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=E.specfemp['mpis'], cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)
    T.task.gpu_reqs = dict(
        gpu_processes=1, gpu_process_type=None,  gpu_threads=1, gpu_thread_type='CUDA')

    return [T]


def frechet(E: EventConfig) -> tp.List[Task]:

    # Empty Task list
    t = []

    for _i in E.simpars:

        # Create frechet derivative task depending on the parameters to download.
        T = Task()
        T.name = f"{E.name}.T.Frechet.dsdm{_i:05d}"
        T.executable = 'bin/xspecfem3D'
        T.sandbox = os.path.join(
            E.outdir, 'simu', 'simu', 'dsdm', f'dsdm{_i:05d}')
        T.pre_exec = [
            E.conda_init,
            f'module load {E.specfemp["modules"]}',
            'gcmt3d-update-cmt-dsdm'
        ]
        # Same number of CPUs and GPUs
        T.task.cpu_reqs = dict(
            cpu_processes=E.specfemp['mpis'], cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)
        T.task.gpu_reqs = dict(
            gpu_processes=1, gpu_process_type=None,  gpu_threads=1, gpu_thread_type='CUDA')

        t.append(T)

    return t


def process_all(E: EventConfig) -> Stage:

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-ProcessAll"

    # Processing the data
    S.add_tasks(process_data(E))

    # Processing the data
    S.add_tasks(process_synt(E))

    # Processing the data
    S.add_tasks(process_dsdm(E))

    return S


def process_all_synt(E: EventConfig) -> Stage:

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-Process-All-Synt"

    # Processing the data
    S.add_tasks(process_synt(E))

    # Processing the data
    S.add_tasks(process_dsdm(E))

    return S


def process_data(E: EventConfig) -> tp.List[Task]:

    # Create forward modelling task
    T = Task()
    T.name = f"{E.name}-T-Process-Data"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-process-data'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=30, cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)

    return [T]


def process_synt(E: EventConfig) -> tp.List[Task]:

    # Create forward modelling task
    T = Task()
    T.name = f"{E.name}-T-Process-Synt"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-process-synt'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=30, cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)

    return [T]


def process_dsdm(E: EventConfig) -> tp.List[Task]:

    # Empty Task list
    t = []

    # Create forward modelling task
    for _i in range(E.NM):
        T = Task()
        T.name = f"{E.name}-T-Process-DSDM-process-dsdm{_i:05d}"
        T.pre_exec = [E.conda_init]
        T.executable = 'gcmt3d-process-synt'
        T.arguments = [f'{E.outdir}', f'_i']
        # Same number of CPUs and GPUs
        T.task.cpu_reqs = dict(
            cpu_processes=30, cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)

        t.append(T)
    return t


def window(E: EventConfig) -> Stage:

    # Create forward modelling task
    T = Task()
    T.name = f"{E.name}-T-Window-Data"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-window'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=30, cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-Window"

    return S


def weights(E: EventConfig) -> Stage:

    # Create task
    T = Task()
    T.name = f"{E.name}-T-Compute-weights"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-weights'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=30, cpu_process_type='MPI', cpu_threads=1, cpu_thread_type=None)

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-Weights"

    return S


def cgh(E: EventConfig) -> Stage:

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-CGH"

    # Compute cost
    S.add_tasks(compute_cost(E))

    # Compute gradient
    S.add_tasks(compute_gradient(E))

    # Hessian
    S.add_tasks(compute_hessian(E))

    return S


def compute_cost(E: EventConfig) -> tp.List[Task]:
    # Create compte cost task
    T = Task()
    T.name = f"{E.name}-T-Compute-Cost"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-cost'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=1, cpu_process_type=None, cpu_threads=1, cpu_thread_type=None)

    return [T]


def compute_gradient(E: EventConfig) -> tp.List[Task]:
    # Create compte cost task
    T = Task()
    T.name = f"{E.name}-T-Compute-Gradient"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-gradient'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=1, cpu_process_type=None, cpu_threads=1, cpu_thread_type=None)

    return [T]


def compute_hessian(E: EventConfig) -> tp.List[Task]:

    # Create compte cost task
    T = Task()
    T.name = f"{E.name}-T-Compute-Hessian"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-hessian'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=1, cpu_process_type=None, cpu_threads=1, cpu_thread_type=None)

    S.add_tasks([T])

    return [T]


def descent(E: EventConfig) -> Stage:

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-CGH"

    S.add_tasks(compute_descent(E))

    return S


def compute_descent(E: EventConfig) -> tp.List[Task]:

    # Create compute cost task
    T = Task()
    T.name = f"{E.name}-T-Compute-Descent"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-descent'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=1, cpu_process_type=None, cpu_threads=1, cpu_thread_type=None)

    return [T]


def compute_optvals(E: EventConfig) -> Stage:

    # Create compute cost task
    T = Task()
    T.name = f"{E.name}-T-Compute-Descent"
    T.pre_exec = [E.conda_init]
    T.executable = 'gcmt3d-compute-descent'
    T.arguments = [f'{E.outdir}']
    # Same number of CPUs and GPUs
    T.task.cpu_reqs = dict(
        cpu_processes=1, cpu_process_type=None, cpu_threads=1, cpu_thread_type=None)

    # Create Stage
    S = Stage()
    S.name = f"{E.name}-S-Compute-Optvals"
    S.add_tasks([T])

    return S
