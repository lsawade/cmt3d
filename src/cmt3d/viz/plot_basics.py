import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import cmt3d.ioi as ioi
import obsplotlib.plot as opl


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
                cax = opl.axes_from_axes(
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


def plot_cm(outdir: str):

    m = ioi.read_model(outdir, 0, 0)

    costdir = os.path.join(outdir, 'cost')

    clist = []
    for _cfile in sorted(os.listdir(costdir)):
        clist.append(np.load(os.path.join(costdir, _cfile)))

    N = 200
    marray = np.zeros((m.size, N))
    for _i in range(N):
        marray[:, _i] = ioi.read_model(outdir, _i, ls=0)

    mnorm = 0.5*np.sum((marray - m_sol[:, np.newaxis])**2, axis=0)/m.size

    ax = axes()
    plot(clist/clist[0])
    plot(mnorm/mnorm[0])
    # ax.set_yscale('log')


def plot_gh(outdir: str, it: int, ls: int):
    g = ioi.read_gradient(outdir, it, ls)
    H = ioi.read_hessian(outdir, it, ls)
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]})
    axes[0].plot(g)
    im = axes[1].imshow(H)
    plt.colorbar(im)


def plot_g(outdir, wavetype, it, ls):

    # Cost dir
    costdir = os.path.join(outdir, 'data')

    data = ioi.read_data(outdir, wavetype)
    synt = ioi.read_synt(outdir, wavetype, it, ls)
    model = ioi.read_model(outdir, it, ls)
    mnames = ioi.read_model_names(outdir)

    dsdm = []

    for _i, (_m) in enumerate(model):
        dsdm.append(ioi.read_dsdm(outdir, wavetype, _i, it, ls))

    plt.figure(figsize=(10, 6))

    # Plot the plain data
    ax = plt.subplot(251)
    ax.axis('off')
    plt.imshow(data)
    plt.text(0, 1, 'Data', ha='left',
             va='bottom', transform=ax.transAxes)

    # Plot the difference between synthetics and data
    ax = plt.subplot(252)
    ax.axis('off')
    plt.imshow(data-synt)
    plt.text(0, 1, 'Difference', ha='left',
             va='bottom', transform=ax.transAxes)
    # plt.text(0, 0, f'Misfit: {0.5/synt.size*np.sum((synt-data)**2):.4f}', ha='left',
    #          va='top', transform=ax.transAxes)

    for _i, _key in enumerate(mnames):
        ax = plt.subplot(252 + 1 + _i)
        ax.axis('off')
        plt.imshow(dsdm[_i])
        plt.text(0, 1, _key, ha='left',
                 va='bottom', transform=ax.transAxes)
