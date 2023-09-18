
import os
import numpy as np
from obspy import Stream
import cmt3d


def write_data(data: Stream, outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    # Write output
    cmt3d.write_pickle(file, data)


def read_data(outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    return cmt3d.read_pickle(file)


def write_data_windowed(data: Stream, outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_windowed_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    # Write output
    cmt3d.write_pickle(file, data)


def read_data_windowed(outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_windowed_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    return cmt3d.read_pickle(file)
