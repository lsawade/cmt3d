import os
import typing as tp
import cmt3d
import obspy
from gf3d.seismograms import GFManager
from .model import read_model, read_model_names
from .log import get_iter, get_step
from .utils import cmt3d2gf3d


def write_synt_raw(raw: obspy.Stream, outdir):

    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'simu')

    # Get filename
    fname = 'synt.pkl'

    # Get output file name
    file = os.path.join(syntdir, fname)

    # Write output
    cmt3d.write_pickle(file, raw)


def read_synt_raw(outdir) -> obspy.Stream:

    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'simu')

    # Get filename
    fname = 'synt.pkl'

    # Get output file name
    file = os.path.join(syntdir, fname)

    # Write output
    raw = cmt3d.read_pickle(file)  # type: obspy.Stream

    return raw


def write_synt(synt: obspy.Stream, outdir, wavetype, it, ls=None):


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
    cmt3d.write_pickle(file, synt)


def read_synt(outdir, wavetype, it, ls=None) -> obspy.Stream:

    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'synt')

    # Get filename
    if ls is not None:
        fname = f'synt_{wavetype}_it{it:05d}_ls{ls:05d}.pkl'
    else:
        fname = f'synt_{wavetype}_it{it:05d}.pkl'

    file = os.path.join(syntdir, fname)

    synt = cmt3d.read_pickle(file)  # type: obspy.Stream

    return synt

def read_synt_all(outdir, wavetype) -> tp.List[obspy.Stream]:

    # Get the synthetics directory
    syntdir = os.path.join(outdir, 'synt')

    # Init list
    synts = []

    # Loop over synthetics
    for _sfile in sorted(os.listdir(syntdir)):

        if "ls00000" in _sfile and wavetype in _sfile:

            synt = cmt3d.read_pickle(os.path.join(syntdir, _sfile))  # type: obspy.Stream
            synts.append(synt)

    return synts


def forward(outdir, gfm: GFManager):

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Get dirs
    metadir = os.path.join(outdir, 'meta')

    # Read metadata and model
    m = read_model(outdir, it, ls)
    model_names = read_model_names(outdir)

    # Read original CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt')
    )

    # Update the CMTSOLUTION with the current model state
    for _m, _mname in zip(m, model_names):
        setattr(cmtsource, _mname, _m)

    # Update half-duration afterwards.
    cmtsource.update_hdur()

    # Convert to gf3d style source
    gf3d_source = cmt3d2gf3d(cmtsource)

    # Write CMTSOLUTION to simulation DATA directory
    st = gfm.get_seismograms(gf3d_source)

    # Write synthetics
    write_synt_raw(st, outdir)


