"""
cmt3d add-events
cmt3d check-events
cmt3d check-events-todo
cmt3d check-events-todownload
cmt3d check-status
cmt3d compute-weights
cmt3d create-event-dir
cmt3d create-inv-dir
cmt3d get-data
cmt3d get-data-mpi
cmt3d plot-wave
cmt3d cmt
cmt3d print-cost
cmt3d print-model
cmt3d print-model-names
cmt3d print-optvals
cmt3d print-status
cmt3d process-data
cmt3d process-dsdm
cmt3d process-synt
cmt3d window
"""

import click


@click.group()
def cli():
    pass


@cli.command(name='add-events')
def add_events():
    from .add_events import bin
    bin()


@cli.command(name='check-events')
def check_events():
    from .check_events import bin
    bin()


@cli.command(name='check-events-todo')
def check_events_todo():
    from .check_events_todo import bin
    bin()


@cli.command(name='check-status')
def check_status():
    from .check_status import bin
    bin()


@cli.command(name='compute-weights')
def compute_weights():
    from .compute_weights import bin
    bin()


@cli.command(name='create-event-dir')
def create_event_dir():
    from .create_event_dir import bin
    bin()


@cli.command(name='create-inv-dir')
def create_inv_dir():
    from .create_inversion_dir import bin
    bin()


@cli.command(name='get-data')
def get_data():
    from .get_data import bin
    bin()

def
get-data-mpi
plot-wave
cmt
print-cost
print-model
print-model-names
print-optvals
print-status
process-data
process-dsdm
process-synt
window
