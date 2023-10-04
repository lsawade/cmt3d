# %%
import obsplotlib.plot as opl
import os
import obspy
from collections import OrderedDict
import numpy as np
import cmt3d
import cmt3d.ioi as ioi
import matplotlib.pyplot as plt
from cmt3d.viz.plot_inversion_section import plot_inversion_section

# %%

outdir = "/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/B010495E"


# %%
plt.close('all')
wave = "body"
component = "Z"
plot_inversion_section(outdir, wave, windows=False,
                       component=component)
plt.savefig(f"section_{wave}_{component}.pdf", dpi=300)

# %% get stuff from directory
cmt = ioi.get_cmt(outdir, 0)


# %%
# Plotting the frechet derivatives for a specific station
model_names = ioi.read_model_names(outdir)

# Selected station
network, station, component = "IU", "ANMO", "Z"
fdict = dict()

wave = "body"
for _i, _mname in enumerate(model_names):
    fdict[_mname] = ioi.read_dsdm(outdir, wave, _i, 0, 0)


synt = ioi.read_synt(outdir, wave, 0, 0)

# %%
opl.frechet(synt, fdict, network, station, component)
plt.savefig(f"frechet_{network}_{station}_{component}_{wave}.png", dpi=300)
# %%
cmts = ioi.get_cmt_all(outdir)
print((cmts[-1].M0-cmts[0].M0)/cmts[0].M0)
print(cmts[-1] - cmts[0])


# %%
def read_all_cmt(outdir, linesearches: bool = True):

    # Get directory
    modldir = os.path.join(outdir, 'modl')

    models = OrderedDict()

    for _mfile in sorted(os.listdir(modldir)):
        itstr, lsstr = os.path.basename(_mfile).strip(".npy").split("_")[1:]

        it = int(itstr[2:])
        ls = int(lsstr[2:])

        if not linesearches and (ls != 0):
            continue

        if it not in models:
            models[it] = OrderedDict()

        models[it][ls] = ioi.get_cmt(outdir, it, ls=ls)

    return models


def read_all_cost(outdir, linesearches: bool = True):

    # Get directory
    costdir = os.path.join(outdir, 'cost')

    costs = OrderedDict()

    for _cfile in sorted(os.listdir(costdir)):
        itstr, lsstr = os.path.basename(_cfile).strip(".npy").split("_")[1:]

        it = int(itstr[2:])
        ls = int(lsstr[2:])

        if not linesearches and (ls != 0):
            continue

        if it not in costs:
            costs[it] = OrderedDict()

        costs[it][ls] = ioi.read_cost(outdir, it, ls=ls)

    return costs


def plot_cm(cmts, costs, mpar='depth_in_m'):

    ax = plt.gca()

    # Get colors from rainbow
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(cmts), endpoint=True))

    for (it, _lscmts), (_, _lscosts) in zip(cmts.items(), costs.items()):

        m = []
        c = []
        iit = np.linspace(0, 1, len(_lscmts), endpoint=True)
        for _i, ((ls, _cmt), (_, _cost)) in enumerate(zip(_lscmts.items(), _lscosts.items())):
            c.append(_cost)
            if mpar == 'iteration':
                m.append(it + iit[_i])
            else:
                m.append(getattr(_cmt, mpar))

        plt.plot(m, c, 'o-', c=colors[it], label=f"{it}")
        plt.show(block=False)


# %%
cmts = read_all_cmt(outdir)
costs = read_all_cost(outdir)
plt.close('all')
plt.figure()
plot_cm(cmts, costs, mpar='iteration')

plt.savefig(f"costs_including_ls.png", dpi=300)
