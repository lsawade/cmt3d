import os
from lwsspy.utils.io import read_yaml_file

from ..measurements import get_measurements_and_windows
from .utils import write_pickle
from .data import read_data
from .forward import read_synt


def make_measurements(outdir, it, ls):

    # Get process parameters
    processparams = read_yaml_file(os.path.join(outdir, 'process.yml'))

    # Get wavetypes
    wavetypes = list(processparams.keys())

    for _wtype in wavetypes:

        data = read_data(outdir)

        window_dict = get_all_measurements(
            data, synt, self.cmtsource, logger=self.logger)

        # Create output file
        filename = "measurements"
        if post_fix is not None:
            filename += "_" + post_fix
        filename += ".pkl"

        outfile = os.path.join(self.cmtdir, filename)
        with open(outfile, "wb") as f:
            cPickle.dump(window_dict, f)

        return window_dict
