"""
:Author:
    Lucas Sawade (lsawade-at-princeton.edu)

:Last Modified:
    2022.03.04 11.00
"""

from os import path
from sys import argv, exit
from ..functions.plot import plot_stream_pdf
from ..functions.log import get_iter, get_step


def bin():
    """

    Usage:

        gcmt3d-plot-wave /path/to/eventinvdir <wave>

    This script calls a python function that plots wavetype wave.
    
    """

    # Get args or print usage statement 
    if (len(argv) != 3) or (argv[1] == '-h') or (argv[1] == '--help'):
        print(bin.__doc__)
        exit()
    else:
        outdir, wave = argv[1:]

    # Run the initializer
    outfile = path.join(outdir, f'{wave}.pdf')
    plot_stream_pdf(
        outdir, outfile, it=get_iter(outdir), ls=get_step(outdir), 
        wavetype=wave)
