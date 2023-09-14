from curses import meta
import imp
import os
import numpy as np
from .model import read_model, read_model_names
from .gaussian2d import g
from .utils import write_pickle, read_pickle
from obspy import Stream
from lwsspy.seismo.source import CMTSource
from .log import get_iter, get_step

def write_synt(synt: Stream, outdir, wavetype, it, ls=None):

        
    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'synt')

    # Get filename
    if ls is not None:
        fname = f'synt_{wavetype}_it{it:05d}_ls{ls:05d}.pkl'
    else:
        fname = f'synt_{wavetype}_it{it:05d}.pkl'

    # Get output file name
    file = os.path.join(syntdir, fname)

    # Write output
    write_pickle(file, synt)


def read_synt(outdir, wavetype, it, ls=None):

    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'synt')

    # Get filename
    if ls is not None:
        fname = f'synt_{wavetype}_it{it:05d}_ls{ls:05d}.pkl'
    else:
        fname = f'synt_{wavetype}_it{it:05d}.pkl'

    file = os.path.join(syntdir, fname)

    return read_pickle(file)


def update_cmt_synt(outdir):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get dirs
    metadir = os.path.join(outdir, 'meta')
    ssyndir = os.path.join(outdir, 'simu', 'synt')

    # Read metadata and model
    m = read_model(outdir, it, ls)
    model_names = read_model_names(outdir)

    # Read original CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt')
    )

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmtsource, _mname, _m)

    # Update half-duration afterwards.
    cmtsource.update_hdur()
    
    # Write CMTSOLUTION to simulation DATA directory
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(ssyndir, 'DATA', 'CMTSOLUTION'))
