#%%
import os
from collections import OrderedDict
import numpy as np
import cmt3d.ioi as ioi
import matplotlib.pyplot as plt
from cmt3d.viz.plot_inversion_section import plot_inversion_section

# %%
outdir = "/Users/lucassawade/database/nnodes/C201009071613A/"

# %%
plot_inversion_section(outdir, 'body', windows=False,
                       component='Z')



# %%
cmts = ioi.get_cmt_all(outdir)
print((cmts[-1].M0-cmts[0].M0)/cmts[0].M0)
print(cmts[-1] -cmts[0])


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

    colors = ['k', 'r', 'b', 'g',]
    for (it, _lscmts), (_, _lscosts) in zip(cmts.items(), costs.items()):

        m = []
        c = []
        for (ls, _cmt), (_, _cost) in zip(_lscmts.items(), _lscosts.items()):
            m.append(getattr(_cmt, mpar))
            c.append(_cost)

        plt.plot(m, c, 'o-', c=colors[it], label=f"{it}")
        plt.show(block=False)


# %%
cmts = read_all_cmt(outdir)
costs = read_all_cost(outdir)
plot_cm(cmts, costs, mpar='depth_in_m')


