# %%
import os
import typing as tp
from cmt3d.viz import utils
from cmt3d.viz import catalog_utils as cutils
from cmt3d.viz import catalog_plot_utils as cputils
from cmt3d.cmt_catalog import CMTCatalog
from cmt3d.viz.compare_catalogs import CompareCatalogs
from glob import glob
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import OrderedDict

utils.updaterc()
plt.ion()

# %%
# Load the CMT catalog
cat1 = CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))


# %%
# Download Crust1.0
dlat = 1.0
dlon = 1.0

minlat = -89.5
minlon = -179.5
maxlat = 89.5
maxlon = 179.5

lon, lat, thck = np.loadtxt("data/crust1.0/crsthk.xyz").T

lon = lon.reshape(180, 360)[0, :]
lat = lat.reshape(180, 360)[:, 0]
thck = thck.reshape(180, 360)


def get_thck(lat, lon, thck, xlat, xlon):
    lat_idx = np.argmin(np.abs(lat - xlat))
    lon_idx = np.argmin(np.abs(lon - xlon))

    return thck[lat_idx, lon_idx]


def get_moho_depth(lat, lon, depth, xlat, xlon):
    lat_idx = np.argmin(np.abs(lat - xlat))
    lon_idx = np.argmin(np.abs(lon - xlon))

    return depth[lat_idx, lon_idx]


thcks = []
for event in cat1:
    thcks.append(get_topo(lat, lon, thck, event.latitude, event.longitude))


lon, lat, depthtomoho = np.loadtxt("data/crust1.0/depthtomoho.xyz").T

lon = lon.reshape(180, 360)[0, :]
lat = lat.reshape(180, 360)[:, 0]
depthtomoho = depthtomoho.reshape(180, 360)


# %%
shallow_cat, _ = cat1.filter(maxdict={"depth_in_m": 20000})

# %%


from scipy import stats


def get_kde(x, y):
    kernel = stats.gaussian_kde(np.vstack([x, y]))
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


def get2Dhist(x, y):
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    Z, _, _ = np.histogram2d(
        x,
        y,
        bins=100,
        range=[[xmin, xmax], [ymin, ymax]],
    )
    return X, Y, Z


# %%


color_dict = {"normal": "blue", "thrust": "red", "strike-slip": "green"}

incrust = []
outsidecrust = []

for _event in shallow_cat:
    _depth = get_moho_depth(lat, lon, depthtomoho, _event.latitude, _event.longitude)
    if -_depth > _event.depth_in_m / 1000:
        incrust.append(_event)

    else:
        outsidecrust.append(_event)


incat = CMTCatalog(incrust)
outcat = CMTCatalog(outsidecrust)

for cat, label in zip([incat, outcat], ["Inside crust", "Outside Crust"]):
    plt.figure()
    split_cat = cat.split_to_mechanism()

    handles = []
    for _j, (mech, _cat) in enumerate(split_cat.items()):

        _thcks = []
        for _event in _cat:
            _thck = get_thck(lat, lon, thck, _event.latitude, _event.longitude)
            _thcks.append(_thck)

        _thcks = np.array(_thcks)
        idx = np.where(_thcks < 100)[0]
        gamma = _cat.getvals("gamma")

        # idx = np.where(_thcks < 15)[0]
        ax = plt.subplot(1, 3, _j + 1)

        X, Y, Z = get_kde(gamma[idx], np.array(_thcks)[idx])
        p1_range = (Z.max() - Z.min()) * 0.01
        Zc = np.where(Z < (Z.min() + p1_range), np.nan, Z)
        ax.contourf(X, Y, Zc, cmap="gray_r", levels=10)
        # X, Y, Z = get2Dhist(gamma[idx], np.array(_thcks)[idx])
        # Zc = np.where(Z < 1, np.nan, Z)
        # ax.pcolormesh(X, Y, Zc, cmap="gray_r")

        plt.scatter(
            gamma[idx],
            np.array(_thcks)[idx],
            s=0.25,
            alpha=0.1,
            label=mech,
            color=[
                color_dict[mech],
            ]
            * len(gamma[idx]),
        )

        R = np.corrcoef(gamma[idx], np.array(_thcks)[idx])[0, 1]

        plt.vlines(
            0.0,
            np.min(_thcks[idx]),
            np.max(_thcks[idx]),
            color="black",
            linestyle=":",
            lw=0.5,
            alpha=1.0,
        )

        if _j == 0:
            plt.ylabel("Crustal Thickness [km]")

        utils.plot_label(ax, f"R = {R:.2f}", location=2, box=False)
        plt.title(mech)
        plt.xlim(-np.pi / 6, np.pi / 6)
        plt.xlabel("$\\gamma$")

    plt.suptitle(label)


# %%


plt.figure()
color_dict = {"normal": "blue", "thrust": "red", "strike-slip": "green"}

for _i, mtcomp in enumerate(["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]):

    split_cat = shallow_cat.split_to_mechanism()

    for _j, (mech, _cat) in enumerate(split_cat.items()):

        _comp = _cat.getvals(mtcomp)
        _M0 = _cat.getvals("M0")

        _thcks = []
        for _event in _cat:
            _thcks.append(get_topo(lat, lon, thck, _event.latitude, _event.longitude))
        ax = plt.subplot(3, 6, _i + 1 + 6 * _j)

        _thcks = np.array(_thcks)

        idx = np.where(_thcks < 15)[0]
        plt.scatter(
            _comp[idx] / _M0[idx],
            np.array(_thcks)[idx],
            s=1,
            alpha=0.5,
            label=mech,
            color=[
                color_dict[mech],
            ]
            * len(_M0[idx]),
        )

        if _j == len(split_cat) - 1:
            plt.xlabel(mtcomp)

        if _i == 0:
            plt.ylabel("Crustal Thickness")

        if _i == 5:
            utils.plot_label(
                ax, mech, location=15, box=False, rotation=270, ha="center", va="center"
            )
