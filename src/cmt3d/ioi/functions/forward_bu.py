import os
import numpy as np
from .model import read_model
from .metadata import read_metadata
from .gaussian2d import g


def write_synt(synt, syntdir, it, ls=None):
    if ls is not None:
        fname = f"synt_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"synt_it{it:05d}.npy"
    file = os.path.join(syntdir, fname)
    np.save(file, synt)


def read_synt(syntdir, it, ls=None):
    if ls is not None:
        fname = f"synt_it{it:05d}_ls{ls:05d}.npy"
    else:
        fname = f"synt_it{it:05d}.npy"
    file = os.path.join(syntdir, fname)
    return np.load(file)


def forward(modldir, metadir, syntdir, it, ls=None):

    # Read metadata and model
    m = read_model(outdir, it, ls)
    X = read_metadata(metadir)

    # Forward modeling
    synt = g(m, X)

    # Write to disk
    write_synt(synt, syntdir, it, ls)
