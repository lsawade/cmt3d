import click


@click.group()
def cli():
    pass


@cli.command(name='create')
@click.argument('eventfile', type=str)
@click.argument('inputfile', type=str)
def create(eventfile: str, inputfile: str):
    from .functions.utils import create_forward_dirs
    create_forward_dirs(eventfile, inputfile)


@cli.command(name='subset')
@click.argument('outdir', type=str)
@click.argument('dbname', type=str)
@click.option('--local', is_flag=True, show_default=True, default=False,
              help='database is local')
def make_subset(outdir: str, dbname: str, local: bool):
    from .functions.utils import create_gfm
    create_gfm(outdir, dbname, local)


@cli.command(name='download')
@click.argument('outdir', type=str)
def download(outdir: str):
    from .functions.get_data import get_data
    get_data(outdir)


@cli.command(name='forward-kernel')
@click.argument('outdir', type=str)
def forward_kernel_mpi(outdir: str):
    from .functions.forward_kernel import forward_kernel
    forward_kernel(outdir)


@cli.group()
def process():
    pass


@process.command(name="data")
@click.argument('outdir', type=str)
@click.argument('wave', type=str)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_data(outdir, wave, nproc):
    from .functions.processing import process_data_wave, process_data_wave_mpi
    if nproc == 0:
        process_data_wave_mpi(outdir, wave)
    else:
        process_data_wave(outdir, wave, multiprocesses=nproc)


@process.command(name="synt")
@click.argument('outdir', type=str)
@click.argument('wave', type=str)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_synt(outdir, wave, nproc):
    from .functions.processing import process_synt_wave, process_synt_wave_mpi
    if nproc == 0:
        process_synt_wave_mpi(outdir, wave)
    else:
        process_synt_wave(outdir, wave, multiprocesses=nproc)


@process.command(name="dsdm")
@click.argument('outdir', type=str)
@click.argument('nm', type=int)
@click.argument('wave', type=str)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_dsdm(outdir, nm, wave, nproc):
    from .functions.processing import process_dsdm_wave, process_dsdm_wave_mpi
    if nproc == 0:
        process_dsdm_wave_mpi(outdir, nm, wave)
    else:
        process_dsdm_wave(outdir, nm, wave, multiprocesses=nproc)


@cli.group()
def window():
    pass


@window.command(name='select')
@click.argument('outdir', type=str)
@click.argument('wave', type=str)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def window_select(outdir, wave, nproc):
    from .functions.processing import window_wave, window_wave_mpi
    if nproc == 0:
        window_wave_mpi(outdir, wave)
    else:
        window_wave(outdir, wave, nproc)


@window.command(name='count')
@click.argument('outdir')
def window_count(outdir: str):
    from .functions.processing import check_window_count
    check_window_count(outdir)


@cli.command(name='weights')
@click.argument('outdir', type=str)
def compute_weights(outdir: str):
    from .functions.weighting import compute_weights
    compute_weights(outdir)


@cli.group()
def model():
    pass


@model.command(name="update")
@click.argument('outdir', type=str)
def model_update(outdir: str):
    from .functions.opt import update_model
    update_model(outdir)


@model.command(name="transfer")
@click.argument('outdir', type=str)
def model_transfer(outdir: str):
    from .functions.opt import update_mcgh
    update_mcgh(outdir)


@cli.command(name="cost")
@click.argument('outdir', type=str)
def compute_cost(outdir: str):
    from .functions.cost import cost
    cost(outdir)


@cli.command(name="gradient")
@click.argument('outdir', type=str)
def compute_gradient(outdir: str):
    from .functions.gradient import gradient
    gradient(outdir)


@cli.command(name="hessian")
@click.argument('outdir', type=str)
def compute_hessian(outdir: str):
    from .functions.hessian import hessian
    hessian(outdir)


@cli.command(name="descent")
@click.argument('outdir', type=str)
def compute_descent(outdir: str):
    from .functions.descent import descent
    descent(outdir)


@cli.command(name='update-step')
@click.argument('outdir', type=str)
def update_step(outdir):
    from .functions.log import update_step
    update_step(outdir)


@cli.command(name='update-iter')
@click.argument('outdir', type=str)
def update_iter(outdir):
    from .functions.log import update_iter
    update_iter(outdir)


@cli.command(name='reset-step')
@click.argument('outdir', type=str)
def reset_step(outdir):
    from .functions.log import reset_step
    reset_step(outdir)


@cli.command(name='linesearch')
@click.argument('outdir', type=str)
def linesearch(outdir: str):
    from .functions.linesearch import linesearch
    linesearch(outdir)
