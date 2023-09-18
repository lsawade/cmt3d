import os
import logging
import cmt3d
from .data import read_data
from .forward import read_synt
from .model import get_cmt
from .forward import read_synt


def make_measurements(outdir, it, ls):

    # Get dirs
    cmtsource = get_cmt(outdir, it=0, ls=0)

    # OUtfile dir
    measdir = os.path.join(outdir, 'meas')

    # Get process parameters
    processparams = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get wavetypes
    wavetypes = list(processparams.keys())

    for _wtype in wavetypes:

        # Read data and synt
        data = read_data(outdir, _wtype)
        synt = read_synt(outdir, _wtype, it, ls)

        # Logger
        logger = logging.getLogger("cmt3d")
        window_dict = cmt3d.get_all_measurements(
            data, synt, cmtsource, logger=logger)

        # Create output file
        filename = os.path.join(
            measdir, f"window_dict_{_wtype}_it{it:05d}_ls{ls:05d}.pkl")

        # Writing the measurement pickle
        cmt3d.write_pickle(filename, window_dict)

        return window_dict
