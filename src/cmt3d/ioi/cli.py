#!/usr/bin/env python

import sys
import click

def mpiabort_excepthook(type, value, traceback):
    from mpi4py import MPI

    traceback.print_exc()
    print('', flush=True)

    mpi_comm = MPI.COMM_WORLD
    mpi_comm.Abort(1)
    sys.__excepthook__(type, value, traceback)


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
@click.argument('inputfilename', type=str)
@click.argument('cmtfilename', type=str)
def download(inputfilename: str, cmtfilename: str):
    from .functions.get_data import get_data
    get_data(inputfilename, cmtfilename)


@cli.command(name='forward-kernel')
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def forward_kernel_mpi(outdir: str, it=None, ls=None):
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD

    def mpiabort_excepthook(type, value, traceback):
        mpi_comm.Abort(1)
        sys.__excepthook__(type, value, traceback)

    from .functions.forward_kernel import forward_kernel
    sys.excepthook = mpiabort_excepthook
    forward_kernel(outdir, it=it, ls=ls)
    sys.excepthook = sys.__excepthook__



@cli.command(name='step-mfpcghc')
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
@click.option('--verbose', is_flag=True, show_default=True, default=False,
              help='verbose    ', type=bool)
@click.option('--cgh-only', is_flag=True, show_default=True, default=False,
              help='only compute cgh    ', type=bool)
@click.option('--fw-only', is_flag=True, show_default=True, default=False,
              help='only forward modeling    ', type=bool)
def step_mfpcghlc(outdir: str, it=None, ls=None, verbose=False,
                 cgh_only=False, fw_only=False):

    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD

    def mpiabort_excepthook(type, value, traceback):
        mpi_comm.Abort(1)
        sys.__excepthook__(type, value, traceback)

    from .functions.step_mfpcghlc import \
        model_update, forward_kernel, process_all_synt, cghlc

    # sys.excepthook = mpiabort_excepthook

    if not cgh_only and not fw_only:
        model_update(outdir, it=it, ls=ls)
    mpi_comm.barrier()
    if not cgh_only:
        forward_kernel(outdir, it=it, ls=ls, verbose=verbose)
        mpi_comm.barrier()
        process_all_synt(outdir, it=it, ls=ls, verbose=verbose)
    mpi_comm.barrier()
    if not fw_only:
        cghlc(outdir, it=it, ls=ls, cgh_only=cgh_only, verbose=verbose)

    # sys.excepthook = sys.__excepthook__



@cli.group()
def process():
    pass


@process.command(name="data")
@click.argument('outdir', type=str)
@click.argument('wave', type=str)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_data(outdir, wave, nproc):
    if nproc == 0:
        from .functions.processing import process_data_wave_mpi
        sys.excepthook = mpiabort_excepthook
        process_data_wave_mpi(outdir, wave)
        sys.excepthook = sys.__excepthook__
    else:
        from .functions.processing import process_data_wave
        process_data_wave(outdir, wave, multiprocesses=nproc)


@process.command(name="synt")
@click.argument('outdir', type=str)
@click.argument('wave', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_synt(outdir, wave, it=None, ls=None, nproc=0):
    if nproc == 0:
        from .functions.processing import process_synt_wave_mpi
        sys.excepthook = mpiabort_excepthook
        process_synt_wave_mpi(outdir, wave, it=it, ls=ls)
        sys.excepthook = sys.__excepthook__
    else:
        from .functions.processing import process_synt_wave
        process_synt_wave(outdir, wave, it=it, ls=ls, multiprocesses=nproc)


@process.command(name="dsdm")
@click.argument('outdir', type=str)
@click.argument('nm', type=int)
@click.argument('wave', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
@click.option('--nproc', show_default=True, default=0,
              help='number of multiprocesses (relevant for not MPI)')
def process_dsdm(outdir, nm, wave, it=None, ls=None, nproc=0):
    if nproc == 0:
        from .functions.processing import process_dsdm_wave_mpi
        sys.excepthook = mpiabort_excepthook
        process_dsdm_wave_mpi(outdir, nm, wave, it=it, ls=ls)
        sys.excepthook = sys.__excepthook__
    else:
        from .functions.processing import process_dsdm_wave
        process_dsdm_wave(outdir, nm, wave, it=it, ls=ls, multiprocesses=nproc)


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
        from .functions.processing import window_wave_mpi
        sys.excepthook = mpiabort_excepthook
        window_wave_mpi(outdir, wave)
        sys.excepthook = sys.__excepthook__
    else:
        from .functions.processing import window_wave
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
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def model_update(outdir: str, it=None, ls=None):
    from .functions.opt import update_model
    update_model(outdir, it=it, ls=ls)


@model.command(name="transfer")
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def model_transfer(outdir: str, it=None, ls=None):
    from .functions.opt import update_mcgh
    update_mcgh(outdir, it=it, ls=ls)


@cli.command(name="cost")
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
@click.argument('outdir', type=str)
def compute_cost(outdir: str, it=None, ls=None):
    from .functions.cost import cost
    cost(outdir, it=it, ls=ls)


@cli.command(name="gradient")
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def compute_gradient(outdir: str, it=None, ls=None):
    from .functions.gradient import gradient
    gradient(outdir, it=it, ls=ls)


@cli.command(name="hessian")
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def compute_hessian(outdir: str, it=None, ls=None):
    from .functions.hessian import hessian
    hessian(outdir, it=it, ls=ls)


@cli.command(name="descent")
@click.argument('outdir', type=str)
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def compute_descent(outdir: str, it=None, ls=None):
    from .functions.descent import descent
    descent(outdir, it=it, ls=ls)


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
@click.option('--it', show_default=True, default=None, help='iteration #',
              type=int)
@click.option('--ls', show_default=True, default=None, help='step      #',
              type=int)
def linesearch(outdir: str, it=None, ls=None):
    from .functions.linesearch import linesearch
    linesearch(outdir, it=it, ls=ls)


@cli.group()
def nnodes():
    pass

@nnodes.command(name='reset')
@click.argument('events', nargs=-1)
@click.option('--level', type=str, default='inversion')
def nnodes_reset(events, level):
    from .bin.reset_inversion import bin
    bin(events, level)


@nnodes.command(name='reset-linesearch')
def nnodes_reset_linesearch():
    from .bin.reset_linesearch import bin
    bin()

@nnodes.command(name='reset-steps')
def nnodes_reset_steps():
    from .bin.reset_steps import bin
    bin()

@nnodes.command(name='fix-iter')
def nnodes_reset_steps():
    from .bin.adjust_iteration import bin
    bin()

@nnodes.command(name='check-events')
@click.argument('events', nargs=-1, default=None, type=str)
@click.option('--print-mode', type=str, default='all')
def nnodes_check_events(events, print_mode):
    from .bin.nnodes_check_events import bin
    bin(events, print_mode=print_mode)


@nnodes.command(name='events')
def nnodes_check_events():
    from .bin.nnodes_events import bin
    bin()


if __name__ == '__main__':

    sys.excepthook = mpiabort_excepthook
    cli()
    sys.excepthook = sys.__excepthook__