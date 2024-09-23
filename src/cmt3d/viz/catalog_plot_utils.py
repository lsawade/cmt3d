import numpy as np
import typing as tp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmt3d.viz.utils as utils
from . import catalog_utils as cutils
import pygmt

if tp.TYPE_CHECKING:
    from cmt3d.cmt_catalog import CMTCatalog


def crossplot(data, parameters, colors, labels):

    N = len(parameters)

    # Create figure Grid
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrows=N, ncols=N, figure=fig, hspace=0.1, wspace=0.1)

    # Set up the labels
    letters = [str(i) for i in range(1, 200)]

    for _j, _param1 in enumerate(parameters):
        for _i, _param2 in enumerate(parameters):

            ax = fig.add_subplot(gs[_j, _i])

            if _i != _j:

                plt.scatter(
                    data[:, _i],
                    data[:, _j],
                    c=colors,
                    s=1.0,
                    alpha=0.25,
                    edgecolors="none",
                )

            if _j == N - 1:
                ax.set_xlabel(labels[_param2])
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if _i == 0:
                ax.set_ylabel(labels[_param1])
            else:
                ax.tick_params(axis="y", labelleft=False)

            # Turn of axis
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Remove ticks
            ax.tick_params(
                axis="both",
                which="both",
                left=False,
                bottom=False,
                right=False,
                top=False,
                labelleft=False,
                labelbottom=False,
                labelright=False,
                labeltop=False,
            )

            # # Add plot letter
            # utils.plot_label(
            #     ax,
            #     letters[counter],
            #     location=1,
            #     box=False,
            #     fontdict=dict(fontsize="x-small"),
            # )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)


def biplot(score, coeff, colors, labels):
    """
    Author: Serafeim Loukas, serafeim.loukas@epfl.ch
    Inputs:
       score: the projected data
       coeff: the eigenvectors (PCs)
       y: the class labels
    """
    xs = score[:, 0]  # projection on PC1
    ys = score[:, 1]  # projection on PC2
    n = coeff.shape[0]  # number of variables
    plt.figure(figsize=(10, 8), dpi=100)
    plt.scatter(xs, ys, c=colors, s=3)  # color based on group

    for i in range(n):
        # plot as arrows the variable scores (each variable has a score for PC1 and one for PC2)
        plt.arrow(
            0,
            0,
            coeff[i, 0],
            coeff[i, 1],
            color="k",
            alpha=0.9,
            linestyle="-",
            linewidth=1.5,
            overhang=0.2,
        )
        plt.text(
            coeff[i, 0] * 1.15,
            coeff[i, 1] * 1.15,
            labels[i],
            color="k",
            ha="center",
            va="center",
            fontsize=10,
        )

    plt.xlabel("PC{}".format(1), size=14)
    plt.ylabel("PC{}".format(2), size=14)
    limx = int(xs.max()) + 1
    limy = int(ys.max()) + 1
    plt.xlim([-limx, limx])
    plt.ylim([-limy, limy])
    plt.grid()
    plt.tick_params(axis="both", which="both", labelsize=14)


def plot_split_gamma_compare(
    split_cat,
    compare_cat=None,
    label="GCMT",
    compare_label="CMT3D+",
    mechlabel_offset=0.125,
):
    # Get number of depth ranges
    nrows = len(split_cat)

    # Set up figure
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.1, wspace=0.2)

    # Set up axes
    axes = []
    for i in range(nrows):
        axesrow = []
        for j in range(3):
            axesrow.append(fig.add_subplot(gs[i, j]))
        axes.append(axesrow)

    # Set up the labels
    letters = "abcdefghijklmnopqrstuvwxyz"

    counter = 0

    for i, (dname, rangd) in enumerate(split_cat.items()):
        dlow, dhigh, dd = rangd["range"]

        # Get the
        osplit_cat = rangd["catalogs"]

        if compare_cat is not None:
            nsplit_cat = compare_cat[dname]["catalogs"]

        # Setup the bins
        # bins = np.linspace(-np.pi / 6, np.pi / 6, 25, endpoint=True)

        for j, (name, ocat) in enumerate(osplit_cat.items()):
            print(f"Plotting {dname}-{name}")

            # Get the right axes instance
            ax = axes[i][j]

            # Get gamma for the catalog
            ogamma = ocat.getvals(vtype="decomp", dtype="gamma")

            bins = np.histogram_bin_edges(
                ogamma, bins="auto", range=(-np.pi / 6, np.pi / 6)
            )

            if compare_cat is not None:
                ncat = nsplit_cat[name]
                ngamma = ncat.getvals(vtype="decomp", dtype="gamma")

            # Plot histogram GCMT3D
            ovals, _, _ = ax.hist(
                ogamma,
                bins=bins,
                facecolor="lightgrey",
                linewidth=0.75,
                histtype="stepfilled",
                edgecolor="none",
                label=label,
                density=True,
            )

            if compare_cat is not None:
                # Get gamma for the catalog
                ngamma = ncat.getvals(vtype="decomp", dtype="gamma")

                # Plot histogram GCMT3D
                nvals, _, _ = ax.hist(
                    ngamma,
                    bins=bins,
                    edgecolor="k",
                    linewidth=0.75,
                    histtype="step",
                    label=compare_label,
                    density=True,
                )
            else:
                nvals = ovals

            ax.legend(
                loc="upper left",
                frameon=False,
                fancybox=False,
                numpoints=1,
                scatterpoints=1,
                fontsize="x-small",
                borderaxespad=0.0,
                borderpad=0.5,
                handletextpad=0.2,
                labelspacing=0.2,
                handlelength=1.0,
                bbox_to_anchor=(0.0, 1.0),
            )

            # Plot stats label
            statlabel = (
                f"{label} N={len(ogamma):d}\n"
                f"$\\mu$ = {np.mean(ogamma):7.4f}\n"
                f"$\\sigma$ = {np.std(ogamma):7.4f}\n"
            )

            if compare_cat is not None:
                statlabel += (
                    f"{compare_label} N={len(ngamma):d}\n"
                    f"$\\mu$ = {np.mean(ngamma):7.4f}\n"
                    f"$\\sigma$ = {np.std(ngamma):7.4f}\n"
                )

            utils.plot_label(
                ax,
                statlabel,
                location=2,
                box=False,
                fontdict=dict(fontsize="xx-small", fontfamily="monospace"),
            )

            # Plot figure letter label
            utils.plot_label(
                ax,
                f"({letters[counter]})",
                location=21,
                box=False,
                fontdict=dict(fontsize="x-small", fontfamily="monospace"),
            )

            counter += 1

            if i == 0:
                utils.plot_label(
                    ax, "CLVD-", location=6, box=False, fontdict=dict(fontsize="small")
                )
                utils.plot_label(
                    ax, "CLVD+", location=7, box=False, fontdict=dict(fontsize="small")
                )
                utils.plot_label(
                    ax, "DC", location=14, box=False, fontdict=dict(fontsize="small")
                )
                utils.plot_label(
                    ax,
                    name.capitalize(),
                    fontsize="small",
                    dist=mechlabel_offset,
                    location=14,
                    box=False,
                    fontdict=dict(fontsize="small"),
                )

            if j == 2:
                axes[i][j].yaxis.set_label_position("right")
                axes[i][j].set_ylabel(
                    dname.capitalize(), fontsize="small", rotation=270, va="bottom"
                )

            # Axes limits
            ymax = np.max((np.max(nvals), np.max(ovals))) * 1.1
            axes[i][j].set_ylim(0, ymax)
            axes[i][j].set_xlim(-np.pi / 6, np.pi / 6)
            axes[i][j].vlines([0], 0, ymax, "k", ls=":", lw=0.5)

            # Label locators
            axes[i][j].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
            axes[i][j].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 12))
            major = utils.Multiple(12, number=np.pi, latex="\pi")
            axes[i][j].xaxis.set_major_formatter(major.formatter)

            if i == nrows - 1:
                axes[i][j].set_xlabel(r"$\gamma$")
            else:
                ax.tick_params(axis="x", labelbottom=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.925, bottom=0.1)
    plt.show(block=False)


def plot_parameter_correlations_mech(
    cat,  #: CMTCatalog,
    parameters: tp.List[str] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    colors=[(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8)],
):

    N = len(parameters)

    # Get parameters from the catalog into a dict
    paramd = dict()

    # Split catalog into mechanisms
    split_cat = cat.split_to_mechanism()

    # Mechanism order
    order = ["strike-slip", "normal", "thrust"]

    labeldict = {
        "thrust": "Thrust",
        "normal": "Normal",
        "strike-slip": "Strike-Slip",
        "m_rr": "$M_{rr}$",
        "m_tt": "$M_{\\theta\\theta}$",
        "m_pp": "$M_{\\phi\\phi}$",
        "m_rt": "$M_{r\\theta}$",
        "m_rp": "$M_{r\\phi}$",
        "m_tp": "$M_{\\theta\\phi}$",
        "depth_in_m": "z",
        "longitude": "Lon",
        "latitude": "Lat",
        "time_shift": "$T_s$",
        "lmd0": "$\\lambda_0$",
        "lmd1": "$\\lambda_1$",
        "lmd2": "$\\lambda_2$",
        "lune_gamma": "$\\gamma$",
        "lune_kappa": "$\\kappa$ St",
        "lune_theta": "$\\theta$ Dp",
        "lune_sigma": "$\\sigma$ Rk",
        "lune_M0": "$M_0$",
        "moment_magnitude": "$M_w$",
    }

    # Get the right axes instance
    for param in parameters:
        paramd[param] = dict()

        for _mech, _subcat in split_cat.items():

            paramd[param][_mech] = dict()

            # The eigenvalues are special
            if param in ["lmd1", "lmd2", "lmd3"]:
                idx = int(param[-1]) - 1
                lmd = np.array([cmt.tnp[0] for cmt in _subcat])
                lmd_idx = lmd[:, idx]
                lmd_idx /= np.max(np.abs(lmd), axis=1)
                paramd[param][_mech]["values"] = lmd_idx

            else:
                paramd[param][_mech]["values"] = _subcat.getvals(vtype=param)

            if param in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
                paramd[param][_mech]["values"] /= _subcat.getvals(vtype="M0")
                paramd[param][_mech]["limits"] = (-1.1, 1.1)

            if param == "depth_in_m":
                vals = _subcat.getvals(vtype="depth_in_m")
                minmw = np.min(vals)
                maxmw = np.max(vals)
                paramd[param][_mech]["values"] = (vals - minmw) / (maxmw - minmw)
                paramd[param][_mech]["limits"] = (0, 1)

            if param == "longitude":
                paramd[param][_mech]["values"] /= 180
                paramd[param][_mech]["limits"] = (-1, 1)

            if param == "latitude":
                paramd[param][_mech]["values"] /= 90
                paramd[param][_mech]["limits"] = (-1, 1)

            if param == "time_shift":
                vals = _subcat.getvals(vtype="time_shift")
                minmw = np.min(vals)
                maxmw = np.max(vals)
                paramd[param][_mech]["values"] = (vals - minmw) / (maxmw - minmw)
                paramd[param][_mech]["limits"] = (0, 1)

            if param == "moment_magnitude":
                vals = _subcat.getvals(vtype="moment_magnitude")
                minmw = np.min(vals)
                maxmw = np.max(vals)
                paramd[param][_mech]["values"] = (
                    paramd[param][_mech]["values"] - minmw
                ) / (maxmw - minmw)
                paramd[param][_mech]["limits"] = (0, 1)

            if "lmd" in param:
                idx = int(param[-1]) - 1
                if idx == 0:
                    paramd[param][_mech]["limits"] = (0.5, 1)
                if idx == 1:
                    paramd[param][_mech]["limits"] = (0.0, 0.5)
                if idx == 2:
                    paramd[param][_mech]["limits"] = (-1, 0.5)

            if param == "lune_gamma":
                paramd[param][_mech]["limits"] = (-30, 30)
            if param == "lune_kappa":
                paramd[param][_mech]["limits"] = (0, 180)
            if param == "lune_theta":
                paramd[param][_mech]["limits"] = (0, 90)
            if param == "lune_sigma":
                paramd[param][_mech]["limits"] = (-90, 90)
            if param == "lune_M0":
                paramd[param][_mech]["limits"] = (0, 1)
                paramd[param][_mech]["values"] /= np.max(cat.getvals(vtype="M0"))

    # Create figure Grid
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrows=N, ncols=N, figure=fig, hspace=0.1, wspace=0.1)

    # Set up the labels
    letters = [str(i) for i in range(1, 200)]

    # Loop over the parameters
    counter = 0
    for _i, _param1 in enumerate(parameters):
        for _j, _param2 in enumerate(parameters[: _i + 1]):

            # Get the right axes instance
            ax = fig.add_subplot(gs[_i, _j])

            for _k, _mech in enumerate(order):
                print(
                    _param2,
                    _param1,
                    _mech,
                    len(paramd[_param2][_mech]["values"]),
                    len(paramd[_param1][_mech]["values"]),
                )
                sc = ax.scatter(
                    paramd[_param2][_mech]["values"],
                    paramd[_param1][_mech]["values"],
                    s=0.25,
                    alpha=0.5,
                    color=colors[_k],
                    label=_mech,
                )

            ax.set_xlim(paramd[_param2][_mech]["limits"])
            ax.set_ylim(paramd[_param1][_mech]["limits"])

            if _i == N - 1:
                ax.set_xlabel(labeldict[_param2])
            else:
                ax.tick_params(axis="x", labelbottom=False)

            if _j == 0:
                ax.set_ylabel(labeldict[_param1])
            else:
                ax.tick_params(axis="y", labelleft=False)

            # Add plot letter
            utils.plot_label(
                ax,
                letters[counter],
                location=1,
                box=False,
                fontdict=dict(fontsize="x-small"),
            )

            # Increase counter
            counter += 1

            # Turn of axis
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            # Remove ticks
            ax.tick_params(
                axis="both",
                which="both",
                left=False,
                bottom=False,
                right=False,
                top=False,
                labelleft=False,
                labelbottom=False,
                labelright=False,
                labeltop=False,
            )

    ax = fig.add_subplot(gs[1, N - 2])
    for _k, key in enumerate(order):
        ax.scatter([], [], s=5, c=(colors[_k],), alpha=0.5, label=labeldict[key])
    plt.legend(
        loc="upper right", scatterpoints=1, frameon=False
    )  # , bbox_to_anchor=(1, 0.5))
    ax.axis("off")
    plt.show(block=False)


def plot_parameter_correlations(
    cat,  #: CMTCatalog,
    cat2,  #: CMTCatalog | None = None,
    parameters: tp.List[str] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    coloringparam: str | None = "depth_in_m",
):

    N = len(parameters)

    # Get parameters from the catalog into a dict
    paramd = dict()

    for param in parameters:
        paramd[param] = dict()
        paramd[param]["values"] = cat.getvals(vtype=param)

        if param in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            paramd[param]["values"] /= cat.getvals(vtype="M0")
            paramd[param]["limits"] = (-1.1, 1.1)
        if param == "depth_in_m":
            paramd[param]["values"] /= np.max(cat.getvals(vtype="depth_in_m"))
            paramd[param]["limits"] = (0, 1)
        if param == "longitude":
            paramd[param]["values"] /= 180
            paramd[param]["limits"] = (-1, 1)
        if param == "latitude":
            paramd[param]["values"] /= 90
            paramd[param]["limits"] = (-1, 1)
        if param == "time_shift":
            paramd[param]["values"] /= np.max(np.abs(cat.getvals(vtype="time_shift")))
            paramd[param]["limits"] = (-1, 1)

    # Get the coloring parameter
    if coloringparam is not None:
        cparam = cat.getvals(vtype=coloringparam)

        if coloringparam == "depth_in_m":
            cparam = cparam / 1000
        if coloringparam in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            cparam = cparam / cat.getvals(vtype="M0")
    else:
        cparam = None

    # Create figure Grid
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrows=N, ncols=N, figure=fig, hspace=0.1, wspace=0.1)

    # Set up the labels
    letters = "abcdefghijklmnopqrstuvwxyz"

    # Loop over the parameters
    for _i, _param1 in enumerate(parameters):
        for _j, _param2 in enumerate(parameters[: _i + 1]):
            # Get the right axes instance
            ax = fig.add_subplot(gs[_i, _j])
            sc = ax.scatter(
                paramd[_param2]["values"],
                paramd[_param1]["values"],
                c=cparam,
                s=0.25,
                alpha=0.25,
                cmap="rainbow",
            )

            ax.set_xlim(paramd[_param2]["limits"])
            ax.set_ylim(paramd[_param1]["limits"])

            if _i == N - 1:
                ax.set_xlabel(_param2)

            if _j == 0:
                ax.set_ylabel(_param1)

    if cparam is not None:
        # fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.8, 0.5, 0.05, 0.3])
        fig.colorbar(sc, cax=cbar_ax)

    plt.show(block=False)


def plot_parameter_correlations_compare(
    cat1,  #: CMTCatalog,
    cat2,  #: CMTCatalog | None = None,
    parameters: tp.List[str] = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    coloringparam: str | None = "depth_in_m",
):

    N = len(parameters)

    # Get parameters from the catalog into a dict
    paramd1 = dict()
    paramd2 = dict()

    for param in parameters:
        paramd1[param] = dict()
        paramd1[param]["values"] = cat1.getvals(vtype=param)

        paramd2[param] = dict()
        paramd2[param]["values"] = cat2.getvals(vtype=param)

        if param in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            paramd1[param]["values"] /= cat1.getvals(vtype="M0")
            paramd1[param]["limits"] = (-1.1, 1.1)
            paramd2[param]["values"] /= cat2.getvals(vtype="M0")
            paramd2[param]["limits"] = (-1.1, 1.1)
        if param == "depth_in_m":
            paramd1[param]["values"] /= np.max(cat1.getvals(vtype="depth_in_m"))
            paramd1[param]["limits"] = (0, 1)
            paramd2[param]["values"] /= np.max(cat2.getvals(vtype="depth_in_m"))
            paramd2[param]["limits"] = (0, 1)
        if param == "longitude":
            paramd1[param]["values"] /= 180
            paramd1[param]["limits"] = (-1, 1)
            paramd2[param]["values"] /= 180
            paramd2[param]["limits"] = (-1, 1)
        if param == "latitude":
            paramd1[param]["values"] /= 90
            paramd1[param]["limits"] = (-1, 1)
            paramd2[param]["values"] /= 90
            paramd2[param]["limits"] = (-1, 1)
        if param == "time_shift":
            paramd1[param]["values"] /= np.max(np.abs(cat1.getvals(vtype="time_shift")))
            paramd1[param]["limits"] = (-1, 1)
            paramd2[param]["values"] /= np.max(np.abs(cat2.getvals(vtype="time_shift")))
            paramd2[param]["limits"] = (-1, 1)

    # Get the coloring parameter
    if coloringparam is not None:
        cparam = cat1.getvals(vtype=coloringparam)

        if coloringparam == "depth_in_m":
            cparam = cparam / 1000
        if coloringparam in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            cparam = cparam / cat1.getvals(vtype="M0")
    else:
        cparam = None

    # Create figure Grid
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrows=N, ncols=N, figure=fig, hspace=0.1, wspace=0.1)

    # Set up the labels
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMONPQRSTUVWXYZ"

    # Loop over the parameters
    for _i, _param1 in enumerate(parameters):
        for _j, _param2 in enumerate(parameters[: _i + 1]):
            # Get the right axes instance
            ax = fig.add_subplot(gs[_i, _j])
            sc = ax.scatter(
                paramd1[_param2]["values"],
                paramd2[_param1]["values"],
                c=cparam,
                s=0.25,
                alpha=0.25,
                cmap="rainbow",
            )

            ax.set_xlim(paramd1[_param2]["limits"])
            ax.set_ylim(paramd1[_param1]["limits"])

            if _i == N - 1:
                ax.set_xlabel(_param2)

            if _j == 0:
                ax.set_ylabel(_param1)

    if cparam is not None:
        # fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.8, 0.5, 0.05, 0.3])
        fig.colorbar(sc, cax=cbar_ax)

    plt.show(block=False)


def plot_split_cat(split_cat, projection="M9c", region=(-120, -20, -60, 20)):
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="magma", series=[0, 800, 10])

    with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("30c", "15c"),
        sharex="b",
        sharey="l",
    ):
        # pygmt.makecpt(cmap="batlow")

        for i, (name, cat) in enumerate(split_cat.items()):
            focal_mechanisms = cutils.cat2meca(cat)
            with fig.set_panel(panel=i):  # sets the current panel
                projection = "M9c"
                region = [minlon, maxlon, minlat, maxlat]
                fig.coast(
                    region=region,
                    projection=projection,
                    land="grey",
                    water="lightblue",
                    shorelines=True,
                    frame=["ag", f"+t{name.capitalize()}"],
                )

                # Pass the focal mechanism data through the spec parameter. In addition provide
                # scale, event location, and event depth

                fig.meca(
                    spec=focal_mechanisms,
                    scale=".05c",  # in centimeters
                    # Fill compressive quadrants with color "red"
                    # [Default is "black"]
                    convention="mt",
                    # compressionfill="red",
                    # Fill extensive quadrants with color "cornsilk"
                    # [Default is "white"]
                    # extensionfill="cornsilk",
                    # Draw a 0.5 points thick dark gray ("gray30") solid outline via
                    # the pen parameter [Default is "0.25p,black,solid"]
                    pen="0.5p,gray30,solid",
                    region=region,
                    projection=projection,
                    cmap=True,
                )
                # fig.text(
                #     x=0,
                #     y=0,
                #     text=name,
                #     font="Helvetica-Bold",
                #     justify="LM",
                #     offset="0.5c",
                #     pen="black",
                # )

    fig.colorbar()
    fig.show()


def plot_split_cat_depth(
    depth_split_cat,
    projection="M?",
    region=(-120, -20, -60, 20),
):
    fig = pygmt.Figure()

    # Get region
    minlon, maxlon, minlat, maxlat = region

    # Define label and title font sizes
    pygmt.config(FONT_TITLE="20p", MAP_TITLE_OFFSET="-5p")

    # xlabel "xaf+lx-axis"
    # ylabel "yaf+ly-axis"
    nrows = len(depth_split_cat)

    with fig.subplot(
        nrows=nrows,
        ncols=3,
        figsize=("30c", f"{4+9*nrows}c"),
        sharex="b",
        sharey="l",
        margins=["0.5c", "0.33c"],
        autolabel=True,
    ):
        counter = 0
        for i, (dname, _vals) in enumerate(depth_split_cat.items()):
            dlow, dhigh, dd = _vals["range"]
            split_cat = _vals["catalogs"]

            for j, (name, _scat) in enumerate(split_cat.items()):
                focal_mechanisms = cutils.cat2meca(_scat)[::-1, :]

                with fig.set_panel(panel=counter):
                    counter += 1
                    # Map stuff
                    projection = "M9c"
                    region = [minlon, maxlon, minlat, maxlat]
                    frame = [
                        "ag",
                    ]

                    # Make colorbar
                    pygmt.makecpt(
                        cmap="magma", series=[dlow - 1, dhigh + 1, dd], reverse="z"
                    )

                    # # Add title to top row
                    if i == 0:
                        frame.append(f"+t{name.capitalize()}")

                    # Add y-axis label to left most plot
                    if j == 0:
                        frame.append(f"yaf+l{dname.capitalize()}")

                    fig.coast(
                        region=region,
                        projection=projection,
                        land="lightgrey",
                        water="white",
                        shorelines=True,
                        frame=frame,
                    )

                    # Pass the focal mechanism data through the spec parameter. In addition provide
                    # scale, event location, and event depth
                    fig.meca(
                        spec=focal_mechanisms,
                        scale=".06c",  # in centimeters
                        # Fill compressive quadrants with color "red"
                        # [Default is "black"]
                        convention="mt",
                        # compressionfill="red",
                        # Fill extensive quadrants with color "cornsilk"
                        # [Default is "white"]
                        # extensionfill="cornsilk",
                        # Draw a 0.5 points thick dark gray ("gray30") solid outline via
                        # the pen parameter [Default is "0.25p,black,solid"]
                        pen="0.25p,black,solid",
                        region=region,
                        projection=projection,
                        cmap=True,
                    )
                    if j == 2:
                        fig.colorbar(
                            position="n1.025/-0.025+w8c/0.3c+v",
                            frame=f"a{dd}f{int(dd/2):d}+lDepth [km]",
                        )
    fig.show()

    return fig


def plot_gmt_cat(
    cat: "CMTCatalog",
    projection="M9c",
    region: tuple | list | None = None,
    topography: bool = False,
    resolution: str = "30s",
    outfile: str = "cat_plot.png",
):

    if region is None:
        region = "d"
        projection = "W12c"
    else:
        projection = "M9c"

    depth = cat.getvals(vtype="depth_in_m") / 1000.0
    print(np.linspace(depth.min(), depth.max(), 20))

    focal_mechanisms = cutils.cat2meca(cat)
    fig = pygmt.Figure()
    pygmt.makecpt(
        cmap="magma",
        series=[depth.min(), depth.max(), (depth.max() - depth.min()) / 20],
    )

    if topography:

        if region == "d":
            dregion = [-180, 180, -90, 90]
        else:
            dregion = region
        grid = pygmt.datasets.load_earth_relief(
            resolution=resolution,
            region=dregion,
            registration="gridline",
        )
        fig.grdimage(grid=grid, frame="a", projection=projection, cmap="oleron")
        fig.colorbar(frame=["a1000", "x+lElevation", "y+lm"])
    else:
        fig.coast(
            region=region,
            projection=projection,
            land="grey80",
            water="white",
            shorelines=True,
            frame=["ag"],
        )

    # Pass the focal mechanism data through the spec parameter. In addition provide
    # scale, event location, and event depth

    fig.meca(
        spec=focal_mechanisms,
        scale=".075c",  # in centimeters
        # Fill compressive quadrants with color "red"
        # [Default is "black"]
        convention="mt",
        # compressionfill="red",
        # Fill extensive quadrants with color "cornsilk"
        # [Default is "white"]
        # extensionfill="cornsilk",
        # Draw a 0.5 points thick dark gray ("gray30") solid outline via
        # the pen parameter [Default is "0.25p,black,solid"]
        pen="0.5p,gray30,solid",
        region=region,
        projection=projection,
        cmap=False,
    )
    # fig.text(
    #     x=0,
    #     y=0,
    #     text=name,
    #     font="Helvetica-Bold",
    #     justify="LM",
    #     offset="0.5c",
    #     pen="black",
    # )

    fig.colorbar()
    fig.savefig(outfile, transparent=False)
    fig.show()
