from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
from lwsspy.plot.axes_from_axes import axes_from_axes
from lwsspy.utils.io import read_yaml_file
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.plot_seismogram import plot_seismogram_by_station
from lwsspy.seismo.read_inventory import flex_read_inventory as read_inventory
from lwsspy.seismo.inv2net_sta import inv2net_sta
from pandas import period_range
from .utils import read_pickle
from .data import read_data_windowed
from .forward import read_synt
from .model import get_cmt
# def cgh(costdir, gradir, hessdir, it, ls=None):
#     c = read_cost(costdir, it, ls)
#     g = read_gradient(graddir, it, ls)
#     H = read_hessian(hessdir, it, ls)

#     return c, g, H


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
    inv = read_inventory(os.path.join(outdir, 'meta', 'stations.xml'))

    # Get stations and networks
    networks, stations = inv2net_sta(inv)

    # Process Params
    processparams = read_yaml_file(os.path.join(outdir, 'process.yml'))

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


def plot_cost(outdir):

    # Cost dir
    costdir = os.path.join(outdir, 'cost')

    clist = []
    for _cfile in sorted(os.listdir(costdir)):
        # if "ls00000.npy" in _cfile:
        if "it00000" in _cfile:
            clist.append(np.load(os.path.join(costdir, _cfile)))

    plt.figure()
    ax = plt.axes()
    plt.plot(np.log10(clist/clist[0]))
    plt.xlabel("Iteration #")
    plt.ylabel("$\\log_{10}\\,C/C_0$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show(block=False)


def plot_hessians(outdir):
    hessdir = os.path.join(outdir, 'hess')
    s = np.load(os.path.join(outdir, 'meta', 'scaling.npy'))

    mlist = []
    for _mfile in sorted(os.listdir(hessdir)):
        # if "_ls00000.npy" in _mfile:
        if "it00000" in _mfile:
            mlist.append(np.load(os.path.join(hessdir, _mfile)))

    N = len(mlist)
    n = int(np.ceil(np.sqrt(N)))

    # Get number of rows and colums
    ncols = n
    if N/n < n:
        nrows = n - 1
    else:
        nrows = n
    print(N, ncols, nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2*ncols + 1.0, nrows*2+1.0))
    plt.subplots_adjust(hspace=0.4)

    counter = 0
    for _i in range(nrows):
        for _j in range(ncols):
            if nrows == 1:
                ax = axes[_j]
            else:
                ax = axes[_i][_j]

            if len(mlist) > counter:
                im = ax.imshow(
                    np.diag(s) @ mlist[counter] @ np.diag(s))
                cax = axes_from_axes(
                    ax, 99080+counter, [0., -.05, 1.0, .05])
                plt.colorbar(im, cax=cax, orientation='horizontal')
            ax.axis('equal')
            ax.axis('off')
            ax.set_title(f"{counter}")

            counter += 1
    plt.show(block=False)


def plot_model(outdir):

    modldir = os.path.join(outdir, 'modl')
    mlist = []
    for _mfile in sorted(os.listdir(modldir)):
        # if "_ls00000.npy" in _mfile:
        if "it00000" in _mfile:
            mlist.append(np.load(os.path.join(modldir, _mfile)))

    mlist = np.array(mlist)
    plt.figure()
    ax = plt.axes()
    plt.plot(mlist/mlist[0])
    plt.xlabel("Iteration #")
    plt.ylabel("$M/M_0$")
    # ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend([_i for _i in range(mlist[0].size)])

    plt.show(block=False)


def plot_cm():
    clist = []
    for _cfile in sorted(os.listdir(costdir)):
        clist.append(np.load(os.path.join(costdir, _cfile)))

    N = 200
    marray = np.zeros((m.size, N))
    for _i in range(N):
        marray[:, _i] = read_model(outdir, _i, ls=0)

    mnorm = 0.5*np.sum((marray - m_sol[:, np.newaxis])**2, axis=0)/m.size

    ax = axes()
    plot(clist/clist[0])
    plot(mnorm/mnorm[0])
    # ax.set_yscale('log')


def plot_gh():
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
    axes[0].plot(g)
    im = axes[1].imshow(H)
    plt.colorbar(im)


def plot_g(mnames, it, ls):

    data = read_data_processed(datadir)
    synt = read_synt(syntdir, it, ls)
    dsdm = []
    for _i, (_key, _m) in enumerate(mdict.items()):
        dsdm.append(read_frechet(_i, frecdir, it, ls))

    plt.figure(figsize=(10, 6))

    # Plot the plain data
    ax = plt.subplot(251)
    ax.axis('off')
    plt.imshow(data)
    plt.text(0, 1, 'Data', ha='left',
             va='bottom', transform=ax.transAxes)

    # Plot the difference between model and data
    ax = plt.subplot(252)
    ax.axis('off')
    plt.imshow(data-synt)
    plt.text(0, 1, 'Difference', ha='left',
             va='bottom', transform=ax.transAxes)
    plt.text(0, 0, f'Misfit: {0.5/synt.size*np.sum((synt-data)**2):.4f}', ha='left',
             va='top', transform=ax.transAxes)

    for _i, _key in enumerate(mnames):
        ax = plt.subplot(252 + 1 + _i)
        ax.axis('off')
        plt.imshow(dsdm[_i])
        plt.text(0, 1, _key, ha='left',
                 va='bottom', transform=ax.transAxes)
