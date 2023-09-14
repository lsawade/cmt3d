
import os
import numpy as np
from obspy import Stream
from .utils import write_pickle, read_pickle


def write_data(data: Stream, outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    # Write output
    write_pickle(file, data)


def read_data(outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    return read_pickle(file)


def write_data_windowed(data: Stream, outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_windowed_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    # Write output
    write_pickle(file, data)


def read_data_windowed(outdir, wavetype):

    # Get the synthetics directory
    datadir = os.path.join(outdir, 'data')

    # Get filename
    fname = f'data_windowed_{wavetype}.pkl'

    # Full file name
    file = os.path.join(datadir, fname)

    return read_pickle(file)
