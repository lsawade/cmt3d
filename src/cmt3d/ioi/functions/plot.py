from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import obspy
import cmt3d
from matplotlib.ticker import MaxNLocator
from lwsspy.plot.axes_from_axes import axes_from_axes
from lwsspy.seismo.plot_seismogram import plot_seismogram_by_station
from lwsspy.seismo.inv2net_sta import inv2net_sta
from .data import read_data_windowed
from .forward import read_synt
from .model import get_cmt


def plot_stream_pdf(outdir, outfile, it=0, ls=0, wavetype='mantle'):

    # Get initial CMT source
    initcmt = get_cmt(outdir, it=0, ls=0)

    # Read data
    data = read_data_windowed(outdir, wavetype)

    # Reading synthetics
    init_synt = read_synt(outdir, wavetype, 0, 0)

    # Get new synthetics
    if (it != 0) or (ls != 0):
        cmt = get_cmt(outdir, it, ls)
        newsynt = read_synt(outdir, wavetype, it, ls)
    else:
        cmt = None
        newsynt = None

    # Get stations
    inv = cmt3d.read_inventory(os.path.join(outdir, 'meta', 'stations.xml'))

    # Get stations and networks
    networks, stations = inv2net_sta(inv)

    # Process Params
    processparams = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get period range from process parameters
    periodrange = 1 / np.array(processparams[wavetype]['process']['pre_filt'])

    # Sort if possible
    if 'distance' in data[0].stats:

        distances = []
        for _network, _station in zip(networks, stations):
            try:
                datatr = data.select(network=_network, station=_station)[0]
                distances.append(datatr.stats['distance'])

            except Exception as e:
                print(_network, _station, e)
                distances.append(180.0)

        idx = np.argsort(distances)
        networks = np.array(networks)[idx].tolist()
        stations = np.array(stations)[idx].tolist()

    with PdfPages(outfile) as pdf:
        for _network, _station in zip(networks, stations):
            try:
                # Checking whether there is at least 1 trace in the stream for
                # the station
                datatr = data.select(network=_network, station=_station)[0]
                # synttr = synt.select(network=_network, station=_station)[0]

                fig, _ = plot_seismogram_by_station(
                    _network, _station,
                    obsd=data,
                    synt=init_synt,
                    newsynt=newsynt,
                    obsdcmt=initcmt,
                    newsyntcmt=cmt,
                    inventory=inv,
                    timescale=3600.0,
                    windows=True,
                    annotations=True,
                    plot_beach=True,
                    map=True,
                    midpointmap=False,
                    pdfmode=True,
                    periodrange=periodrange[0:4:3][::-1],
                    eventdetails=True,
                    stationdetails=True,
                    instrumentdetails=True
                    #     xlim_in_seconds=[40*60,55*60],
                )

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close(fig)
            except Exception as e:
                print(_network, _station, e)
