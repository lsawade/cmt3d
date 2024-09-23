# %%

import numpy as np
import typing as tp
from glob import glob
from copy import deepcopy
from cmt3d.cmt_catalog import CMTCatalog
import cmt3d.viz.utils as utils
import cmt3d.viz.catalog_utils as cutils
import cmt3d.viz.catalog_plot_utils as cputils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

utils.updaterc()
plt.ion()

# %% Create the Catalogs for the PCA and delete unused catalogs

# Load the CMT catalog
cat1 = CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))
cat2 = deepcopy(cat1)
cutils.make_catalog_dc(cat2)

# Actually used
shallowcat, _ = cat1.filter(maxdict=dict(depth_in_m=30000))
# shallowcat_dc, _ = cat2.filter(maxdict=dict(depth_in_m=12000))

# shallowcat = deepcopy(cat1)
# Remove the unused catalogs
del cat1, cat2

# Define coloring for the different mechanisms
mech = shallowcat.getvals(vtype="mtype")
# mech_dc = shallowcat_dc.getvals(vtype="mtype")

colordict = {
    "strike-slip": (0.8, 0.2, 0.2),
    "normal": (0.2, 0.8, 0.2),
    "thrust": (0.2, 0.2, 0.8),
    "unknown": (0.5, 0.5, 0.5),
}

colors = np.array([colordict[m] for m in mech])
colors_dc = np.array([colordict[m] for m in mech])

# %%
parameters = [
    # "moment_magnitude",
    "m_rr",
    "m_tt",
    "m_pp",
    "m_rt",
    "m_rp",
    "m_tp",
    "gamma",
    "lambda1",
    "lambda2",
    "lambda3",
    # "lune_gamma",
    # "longitude",
    # "latitude",
    "depth_in_m",
    "time_shift",
]


# %% Transfrom the

Acat, limits = cutils.cat2array(
    shallowcat,
    parameters=parameters,
    normalize=True,
)

# Acat_dc, limits = cutils.cat2array(
#     shallowcat_dc,
#     parameters=parameters,
#     normalize=True,
# )

# %%

# cputils.crossplot(Acat, parameters, colors, cutils.labeldict)

# %%
idx = np.where(mech == "strike-slip")[0]
# plot_scatter_kde(Acat[:, 6], Acat[:, 0], colors) # gamma vs Mrr
plot_scatter_kde(
    Acat[:, 6], 1 / 3 * (Acat[:, 1] + Acat[:, 2] - 2 * Acat[:, 0]), colors
)  # gamma vs. vertical CLVD
from matplotlib.lines import Line2D

handles, labels = plt.gca().get_legend_handles_labels()
for _mech, _color in colordict.items():
    point = Line2D(
        [0],
        [0],
        label=cutils.labeldict[_mech],
        marker="o",
        markersize=10,
        markeredgecolor="none",
        markerfacecolor=_color,
        linestyle="",
    )
    handles.append(point)

plt.legend(handles=handles, loc="upper left", frameon=False)
plt.xlabel(cutils.labeldict[parameters[6]])
# plt.ylabel(cutils.labeldict[parameters[0]])
plt.ylabel("Vertical CLVD")

# %%
import numpy as np
from dataclasses import dataclass


@dataclass
class PCA:

    reduced_data: np.ndarray
    sorted_eigenvalues: np.ndarray
    sorted_eigenvectors: np.ndarray
    loadings: np.ndarray
    explained_variance: np.ndarray
    total_explained_variance: float


def do_PCA(data):
    ### Representing the Data
    # data has shape (n, d)

    ### Step 1: Standardize the Data along the Features
    # This is the same as using the `StandardScaler` from `sklearn.preprocessing`
    standardized_data = (data - data.mean(axis=0)) / data.std(axis=0)

    ### Step 2: Calculate the Covariance Matrix
    # use `ddof = 1` if using sample data (default assumption) and use `ddof = 0` if using population data
    covariance_matrix = np.cov(standardized_data, ddof=1, rowvar=False)

    ### Step 3: Eigendecomposition on the Covariance Matrix
    # `sklearns` uses and eigh implmentation at least in the version I am using
    # it gives identical results.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    ### Step 4: Sort the Principal Components
    # np.argsort can only provide lowest to highest; use [::-1] to reverse the list
    # np.linalg.eigh gives the ordered but ascending eigenvalues and eigenvectors
    order_of_importance = np.argsort(eigenvalues)[::-1]

    # utilize the sort order to sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order_of_importance]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns

    # It is important to note here that the sorted eigen

    ### Step 5: Calculate the Explained Variance
    # use sorted_eigenvalues to ensure the explained variances correspond to the eigenvectors
    explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

    ### Step 6: Reduce the Data via the Principal Components
    k = data.shape[1]  # select the number of principal components
    reduced_data = np.matmul(
        standardized_data, sorted_eigenvectors[:, :k]
    )  # transform the original data

    ### Step 7: Determine the Explained Variance
    total_explained_variance = sum(explained_variance[:k])

    ### Potential Next Steps: Iterate on the Number of Principal Components
    # plt.plot(np.cumsum(explained_variance))
    # plt.ylim(0, 1)

    # Compute loadings
    loadings = sorted_eigenvectors * np.sqrt(sorted_eigenvalues)[None, :]

    return PCA(
        reduced_data=reduced_data,
        explained_variance=explained_variance,
        sorted_eigenvalues=sorted_eigenvalues,
        sorted_eigenvectors=sorted_eigenvectors,
        total_explained_variance=total_explained_variance,
        loadings=loadings,
    )


# %%


# # Make biplot
# cputils.biplot(
#     score=reduced_data[:, :2],
#     coeff=loadings[:, :2],
#     colors=colors,
#     labels=[cutils.labeldict[p] for p in parameters],
# )


def plot_loadings(loadings, labels):

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

    for _pc in range(6):
        i, j = _pc // 3, np.mod(_pc, 3)
        ax = fig.add_subplot(gs[i, j])
        ax.bar(
            range(6),
            height=loadings[:, _pc],
            bottom=0,
        )
        if i == 1:
            ax.set_xticks(range(6), labels)
        else:
            ax.tick_params(axis="x", which="both", labelbottom=False)
        ax.set_title(f"PC{_pc}")
        plt.minorticks_off()


def plot_explained_variance(explained_variance):
    plt.figure()
    x = range(1, len(explained_variance) + 1)
    plt.plot(x, np.cumsum(explained_variance) * 100)
    plt.xticks(x, [f"PC{i}" for i in x])
    plt.ylabel(f"Explained Variance [%]")
    plt.ylim(0, 100)


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


def plot_pc_v_var(data, var, colors, varlabel="$\\gamma$"):

    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4)

    for _pc in range(6):
        i, j = _pc // 3, np.mod(_pc, 3)
        ax = fig.add_subplot(gs[i, j])
        X, Y, Z = get_kde(var, data[:, _pc])

        p1_range = (Z.max() - Z.min()) * 0.01
        Zc = np.where(Z < (Z.min() + p1_range), np.nan, Z)
        ax.scatter(
            var,
            data[:, _pc],
            c=colors,
            s=1,
            alpha=0.1,
            marker=".",
            facecolor="none",
            zorder=10,
        )
        ax.contourf(X, Y, Zc, cmap="gray_r")
        ax.set_xlabel(varlabel)
        ax.set_ylabel(f"PC{_pc}")
        ax.vlines(
            0,
            ymin=data[:, _pc].min(),
            ymax=data[:, _pc].max(),
            color="k",
            linestyle="--",
            alpha=0.5,
        )
        ax.set_xlim(-np.pi / 6, np.pi / 6)


# plot_pc_v_var(reduced_data, gamma, colors, varlabel="$\\gamma$")

# %%
outdir = "pca_plots"
gamma = shallowcat.getvals(vtype="gamma")

pca = do_PCA(Acat)

# %%
cputils.crossplot(Acat, parameters, colors, cutils.labeldict)
# %%
plt.savefig(f"{outdir}/data_cross_all.png", dpi=300)

plot_loadings(pca.loadings, [cutils.labeldict[p] for p in parameters])
plt.suptitle(f"Loadings")
plt.savefig(f"{outdir}/pca_loadings_all.png", dpi=300)

plot_explained_variance(pca.explained_variance)
plt.title("All mechanisms")
plt.savefig(f"{outdir}/pca_variance_all.png", dpi=300)


plot_pc_v_var(pca.reduced_data, gamma, colors, varlabel="$\\gamma$")
plt.savefig(f"{outdir}/pca_v_gamma_all.png", dpi=300)


labels = {str(i): f"{var:.1f}" for i, var in enumerate(pca.explained_variance * 100)}
cputils.crossplot(pca.reduced_data, list(labels.keys()), colors, labels)
plt.savefig(f"{outdir}/pca_cross_all.png", dpi=300)

# %%


for _mech in ["strike-slip", "normal", "thrust"]:

    idx = np.where(mech == _mech)[0]

    pca = do_PCA(Acat[idx, :])

    cputils.crossplot(Acat[idx, :], parameters, colors[idx], cutils.labeldict)
    plt.savefig(f"{outdir}/data_cross_{_mech}.png", dpi=300)

    # Make labels
    labels = {
        str(i): f"{var:.1f}" for i, var in enumerate(pca.explained_variance * 100)
    }

    cputils.crossplot(pca.reduced_data, list(labels.keys()), colors[idx], labels)
    plt.savefig(f"{outdir}/pca_cross_{_mech}.png", dpi=300)

    # plot_loadings(pca.loadings, [cutils.labeldict[p] for p in parameters])

    # plt.suptitle(f"Loadings: {cutils.labeldict[_mech]}")
    # plt.savefig(f"{outdir}/pca_loadings_{_mech}.png", dpi=300)

    # plot_pc_v_var(pca.reduced_data, gamma[idx], colors[idx], varlabel="$\\gamma$")
    # plt.suptitle("Mechanism: " + _mech)
    # plt.savefig(f"{outdir}/pca_v_gamma_{_mech}.png", dpi=300)

    # plot_explained_variance(pca.explained_variance)
    # plt.title("Mechanism: " + _mech)
    # plt.savefig(f"{outdir}/pca_variance_{_mech}.png", dpi=300)


# %%


def plot_scatter_kde(x, y, colors):

    ax = plt.gca()

    # Get KDE
    X, Y, Z = get_kde(x, y)

    # Get parameter range
    p1_range = (Z.max() - Z.min()) * 0.01
    Zc = np.where(Z < (Z.min() + p1_range), np.nan, Z)

    # Plot KDE
    ax.contourf(X, Y, Zc, cmap="gray_r")

    # Scatter
    ax.scatter(
        x,
        y,
        c=colors,
        s=10,
        alpha=0.1,
        marker=".",
        facecolor="none",
        zorder=10,
    )

    # Center lines
    ax.vlines(
        0, ymin=y.min(), ymax=y.max(), color="k", linestyle="-", zorder=5, lw=0.75
    )
    ax.hlines(
        0, xmin=x.min(), xmax=x.max(), color="k", linestyle="-", zorder=5, lw=0.75
    )

    # lws = 0.5
    # cmean = "k"
    # cstd = "gray"
    # # Get dist mean lines
    # ax.vlines(
    #     x.mean(),
    #     ymin=y.min(),
    #     ymax=y.max(),
    #     linestyle="--",
    #     zorder=5,
    #     color=cmean,
    #     lw=lws,
    # )
    # ax.hlines(
    #     y.mean(),
    #     xmin=x.min(),
    #     xmax=x.max(),
    #     linestyle="--",
    #     zorder=5,
    #     color=cmean,
    #     lw=lws,
    # )

    # # Get std lines
    # ax.vlines(
    #     x.mean() + x.std(),
    #     ymin=y.min(),
    #     ymax=y.max(),
    #     linestyle="-.",
    #     zorder=5,
    #     color=cstd,
    #     lw=lws,
    # )
    # ax.hlines(
    #     y.mean() + y.std(),
    #     xmin=x.min(),
    #     xmax=x.max(),
    #     linestyle="-.",
    #     zorder=5,
    #     color=cstd,
    #     lw=lws,
    # )
    # ax.vlines(
    #     x.mean() - x.std(),
    #     ymin=y.min(),
    #     ymax=y.max(),
    #     linestyle="-.",
    #     zorder=5,
    #     color=cstd,
    #     lw=lws,
    # )
    # ax.hlines(
    #     y.mean() - y.std(),
    #     xmin=x.min(),
    #     xmax=x.max(),
    #     linestyle="-.",
    #     zorder=5,
    #     color=cstd,
    #     lw=lws,
    # )

    # Labels
    # ax.set_xlabel(cutils.labeldict[parameters[0]])
    # ax.set_ylabel(cutils.labeldict[parameters[1]])


# %%
# Do strikeslip only
idx = np.where(mech == "strike-slip")[0]
Mrr = Acat[idx, 0]
Mtt = Acat[idx, 1]
Mpp = Acat[idx, 2]

gamma = shallowcat.getvals(vtype="gamma")[idx]

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plt.sca(ax)
plot_scatter_kde(gamma, Mrr, colors[idx])
plt.ylabel(cutils.labeldict["m_rr"])
plt.xlabel(cutils.labeldict["gamma"])
# plt.sca(ax[1])
# plot_scatter_kde(Mrr, Mpp, colors[idx])
# plt.xlabel(cutils.labeldict["m_rr"])
# plt.ylabel(cutils.labeldict["m_rt"])
# plt.sca(ax[2])
# plot_scatter_kde(gamma, Mrr, colors[idx])
# plt.xlabel(cutils.labeldict["gamma"])
# plt.ylabel(cutils.labeldict["m_rr"])
# plt.sca(ax[3])
# plot_scatter_kde(gamma, Mtt, colors[idx])
# plt.xlabel(cutils.labeldict["gamma"])
# plt.ylabel(cutils.labeldict["m_rt"])
# plt.sca(ax[4])
# plot_scatter_kde(gamma, Mpp, colors[idx])
# plt.xlabel(cutils.labeldict["gamma"])
# plt.ylabel(cutils.labeldict["m_rp"])
utils.plot_label(
    plt.gca(),
    f"R: {np.corrcoef(Mrr, gamma)[0,1]:.2f}",
    location=1,
    box=False,
    fontsize="small",
)

plt.subplots_adjust(wspace=0.6, bottom=0.2, top=0.9, left=0.1, right=0.9)
plt.savefig(f"{outdir}/strike-slip_scatter_kde_iso.pdf", dpi=300)
plt.show()


# %%

idx = np.where(mech == "strike-slip")[0]
Mrr = Acat[idx, 0]
lat = shallowcat.getvals(vtype="latitude")[idx]
lon = shallowcat.getvals(vtype="longitude")[idx]

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sc = plt.scatter(
    lon, lat, c=Mrr, s=10, cmap="seismic", alpha=0.5, vmin=-1, vmax=1, edgecolors="none"
)
plt.colorbar(sc)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
# plt.title("Moment Magnitude")
plt.savefig(f"{outdir}/strike-slip_map_mrr.png", dpi=300)
plt.show()


# %%

DAcat = Acat - Acat_dc

# %%
cputils.crossplot(DAcat, parameters, colors, cutils.labeldict)

# %%

pca = do_PCA(DAcat[idx, :])

# %%
cputils.crossplot(pca.reduced_data, list(labels.keys()), colors[idx], labels)

# %%

plot_loadings(pca.loadings, [cutils.labeldict[p] for p in parameters])

# %%
plot_pc_v_var(pca.reduced_data, gamma, colors[idx], varlabel="$\\gamma$")


# %%
# Plot the three main components in a crossplot versus gamma
gamma = shallowcat.getvals(vtype="gamma")[idx]
dMrr = DAcat[idx, 0]
dMtt = DAcat[idx, 1]
dMpp = DAcat[idx, 2]


Dcat_gamma = np.vstack([dMrr, dMtt, dMpp, gamma]).T

cputils.crossplot(
    Dcat_gamma, ["m_rr", "m_tt", "m_pp", "gamma"], colors[idx], cutils.labeldict
)


# %% Make binned statistic
lat = shallowcat.getvals(vtype="latitude")
lon = shallowcat.getvals(vtype="longitude")
mech = shallowcat.getvals(vtype="mtype")
gamma = shallowcat.getvals(vtype="gamma")
Mrr = Acat[:, 0]
idx_mech = np.where(mech == "strike-slip")[0]

# %%
plt.figure(figsize=(8, 4))

for _i, _mech in enumerate(["normal", "thrust", "strike-slip"]):

    idx_mech = np.where(mech == _mech)[0]

    lat_mech = lat[idx_mech]
    lon_mech = lon[idx_mech]
    gamma_mech = gamma[idx_mech]
    Mrr_mech = Mrr[idx_mech]

    cond = (60 > lat_mech) & (lat_mech > -30) & (0 < lon_mech) & (lon_mech < 180)
    # cond = (60 < lon) & (lon < 180)
    geo_idx = np.where(cond)[0]
    geo_not_idx = np.where(~cond)[0]

    # bins = np.histogram_bin_edges(Mrr, bins="auto")
    bins = np.linspace(0, np.max(np.abs(Mrr_mech)), 15)

    ax = plt.subplot(1, 3, 1 + _i)

    plt.hist(
        np.abs(Mrr_mech),
        bins=bins,
        label="All",
        histtype="step",
        color="k",
        density=True,
    )

    plt.hist(
        np.abs(Mrr_mech[geo_not_idx]),
        bins=bins,
        label="West",
        histtype="step",
        color="b",
        density=True,
    )

    plt.hist(
        np.abs(Mrr_mech[geo_idx]),
        bins=bins,
        label="East",
        histtype="step",
        color="r",
        density=True,
    )

    plt.xlim(0, np.max(np.abs(Mrr_mech)))
    plt.legend(frameon=False, loc="upper right")
    plt.title(f"{cutils.labeldict[_mech]}")
    plt.xlabel("$|M_{rr}|$")
    plt.show()

    plt.savefig(f"{outdir}/hist_mrr_{_mech}.pdf", dpi=300)

    # %%


plt.figure()
plt.scatter(lon_mech[geo_idx], lat_mech[geo_idx], c="r", s=1, alpha=0.5)
plt.scatter(lon_mech[geo_not_idx], lat_mech[geo_not_idx], c="k", s=1, alpha=0.5)
plt.show()


# %%
print(np.mean(gamma[geo_idx]), np.std(gamma[geo_idx]))
print(np.mean(gamma[geo_not_idx]), np.std(gamma[geo_not_idx]))

# %%

# Calculate binned 2d statistic of correlation values between gamma and Mrr in geolocated 5x5 degree bins

from scipy.stats import binned_statistic_2d


# Setup bins using degree
ddeg = 10
lonbins = np.arange(-180, 180, ddeg)
latbins = np.arange(-90, 90, ddeg)


def corr2d(x, y, z1, z2, xbins, ybins, mincount=4):

    # Calculate the binned statistic
    mean_z1z2, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z1 * z2, statistic="mean", bins=[xbins, ybins]
    )
    mean_z1, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z1, statistic="mean", bins=[xbins, ybins]
    )
    mean_z2, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z2, statistic="mean", bins=[xbins, ybins]
    )
    std_z1, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z1, statistic="std", bins=[xbins, ybins]
    )
    std_z2, x_edge, y_edge, binnumber = binned_statistic_2d(
        x, y, z2, statistic="std", bins=[xbins, ybins]
    )

    # Compute normal histogram of locations
    hist, x_edge, y_edge = np.histogram2d(x, y, bins=[xbins, ybins])

    # Get bin centers
    xc = (x_edge[1:] + x_edge[:-1]) / 2
    yc = (y_edge[1:] + y_edge[:-1]) / 2
    YY, XX = np.meshgrid(yc, xc)

    # Combine to get the correlation
    nonnan = np.where(~np.isnan(mean_z1z2) & (hist > mincount))
    corr = (mean_z1z2[nonnan] - mean_z1[nonnan] * mean_z2[nonnan]) / (
        std_z1[nonnan] * std_z2[nonnan]
    )

    xp = XX[nonnan]
    yp = YY[nonnan]

    return xp, yp, corr


# %%

idx_mech = np.where(mech == "strike-slip")[0]
lonp, latp, corr = corr2d(
    lon[idx_mech], lat[idx_mech], Mrr[idx_mech], gamma[idx_mech], lonbins, latbins
)

# lonp, latp, corr = corr2d(lon, lat, Mrr, gamma, lonbins, latbins)

# %%
# Plot the values on a map
plt.figure(figsize=(8, 4))
sc = plt.scatter(
    lonp,
    latp,
    c=corr,
    s=10,
    cmap="seismic",
    # alpha=0.5,
    vmin=-1,
    vmax=1,
    edgecolors="none",
)
cb = plt.colorbar(sc)
cb.set_label("R")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
plt.title(f"{ddeg}x{ddeg}dg binned correlation between $M_{{rr}}$ and $\\gamma$")
plt.subplots_adjust(bottom=0.125, top=0.9, left=0.1, right=1.0)
plt.show()
plt.savefig(f"{outdir}/corr_map.pdf", dpi=300)
# The same plot but with pygmt


# %%
plt.figure(figsize=(8, 4))
sc = plt.scatter(
    lon[idx_mech],
    lat[idx_mech],
    c=np.abs(Mrr[idx_mech]),
    s=10,
    cmap="gray_r",
    alpha=0.5,
    vmin=0,
    vmax=np.max(np.abs(Mrr)),
    edgecolors="none",
)
# Plot rectangle around the east region
plt.plot(
    [0, 0, 180, 180, 00],
    [-30, 60, 60, -30, -30],
    "r-",
    lw=1,
    alpha=0.5,
)

cb = plt.colorbar(sc)
cb.set_label("$|M_{rr}|$")
plt.ylabel("Latitude")
plt.xlabel("Longitude")
# plt.title(f"{ddeg}x{ddeg}dg binned correlation between $M_{{rr}}$ and $\\gamma$")
plt.subplots_adjust(bottom=0.125, top=0.9, left=0.1, right=0.9)
plt.show()

# %%
# The same plot but with pygmt
import pygmt

region = "d"
projection = "W12c"

absMrr = np.abs(Mrr[idx_mech])

fig = pygmt.Figure()
# fig.basemap(region=[-180, 180, -90, 90], projection="W15c", frame=True)
pygmt.makecpt(cmap="gray", reverse=True, series=[0, absMrr.max(), absMrr.max() / 20])
fig.coast(
    region=region,
    projection=projection,
    land="grey90",
    water="white",
    shorelines=True,
    frame=["ag"],
)
fig.plot(
    x=lon[idx_mech],
    y=lat[idx_mech],
    fill=np.abs(Mrr[idx_mech]),
    size=np.ones_like(Mrr[idx_mech]) * 0.1,
    style="cc",
    pen="black",
    cmap=True,
    # alpha=0.5,
)
fig.plot(
    x=[
        0,
        0,
        179.0,
    ],
    y=[
        -30,
        60,
        60,
    ],
    pen="1p,red",
    straight_line=True,
)
fig.plot(
    x=[179.0, 0],
    y=[-30, -30],
    pen="1p,red",
    straight_line=True,
)
fig.plot(
    x=[
        179.0,
        179.0,
    ],
    y=[60, -30],
    pen="1p,red",
)
fig.colorbar(position="JMR+o0.5c/0c+w5c/0.3c", frame=["a2f1", "x+l|Mrr|"])
fig.show()


# %%
# Find mechanisms with relative Mrr's larger than 2.5
idx_large = np.where(np.abs(Mrr) > 2.5)[0]

largeMrrCat = CMTCatalog([shallowcat[_i] for _i in idx_large])
