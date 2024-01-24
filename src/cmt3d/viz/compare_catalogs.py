import os
from typing import Optional
from copy import copy, deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize, BoundaryNorm
from matplotlib.patches import Rectangle
from cartopy.crs import PlateCarree, Mollweide
from numpy.core.fromnumeric import size
from numpy.lib.function_base import quantile
from obspy import Inventory
from . import utils
from .binned_stats import plot_binnedstats
from .scatter_hist import scatter_hist

from ..cmt_catalog import CMTCatalog
from .plot_quakes import plot_quakes
from .plot_quakes import get_level_norm_cmap


class CompareCatalogs:
    # Old and new parameters
    olatitude: np.ndarray
    nlatitude: np.ndarray
    olongitude: np.ndarray
    nlongitude: np.ndarray
    omoment: np.ndarray
    nmoment: np.ndarray
    odepth_in_m: np.ndarray
    ndepth_in_m: np.ndarray
    oeps_nu: np.ndarray
    neps_nu: np.ndarray

    # Map parameters
    central_longitude = 180.0

    # Plot style parameters
    cmt_cmap: ListedColormap = ListedColormap(
        [
            (0.9, 0.9, 0.9),
            (0.7, 0.7, 0.7),
            (0.5, 0.5, 0.5),
            (0.3, 0.3, 0.3),
            (0.1, 0.1, 0.1),
        ]
    )
    depth_dict = {
        0: (0.8, 0.2, 0.2),
        70: (0.2, 0.6, 0.8),
        300: (0.35, 0.35, 0.35),
        800: (0.35, 0.35, 0.35),
    }
    depth_cmap: ListedColormap = ListedColormap(list(depth_dict.values()))
    depth_norm: Normalize = BoundaryNorm(list(depth_dict.keys()), depth_cmap.N)

    def __init__(
        self,
        old: CMTCatalog,
        new: CMTCatalog,
        oldlabel: str = "Old",
        newlabel: str = "New",
        stations: Inventory = None,
        nbins: int = 40,
    ):
        # Assign
        self.oldlabel = oldlabel
        self.newlabel = newlabel

        # Fix up so they can be compared (if both catalogs are very large this
        # can take some time)
        old = old.unique(ret=True)
        self.old, self.new = old.check_ids(new)
        self.N = len(self.old)

        # Number of bins
        self.nbins = nbins

        # Populate attributes
        self.populate()

        # Stations to be plotted alongside if defined
        self.stations = stations

    def populate(self):
        # Old and new values
        self.olatitude = self.old.getvals("latitude")
        self.nlatitude = self.new.getvals("latitude")
        self.olongitude = self.old.getvals("longitude")
        self.nlongitude = self.new.getvals("longitude")
        self.oM0 = self.old.getvals("M0")
        self.nM0 = self.new.getvals("M0")
        self.omoment_magnitude = self.old.getvals("moment_magnitude")
        self.nmoment_magnitude = self.new.getvals("moment_magnitude")
        self.odepth_in_m = self.old.getvals("depth_in_m")
        self.ndepth_in_m = self.new.getvals("depth_in_m")
        self.oeps_nu = self.old.getvals("decomp", "eps_nu")
        self.neps_nu = self.new.getvals("decomp", "eps_nu")
        self.otime_shift = self.old.getvals("time_shift")
        self.ntime_shift = self.new.getvals("time_shift")
        self.odip = self.old.getvals("time_shift")
        self.ndip = self.new.getvals("time_shift")

        self.labeldict = dict(
            latitude="Lat [$^\circ$]",
            longitude="Lon [$^\circ$]",
            M0="M0 [%]",
            moment_magnitude="$M_W$",
            time_shift="$t_{cmt} [s]$",
            eps_nu="$\epsilon$",  # Nu is unused for GCMTs
            depth_in_m="Z [km]",
            location="Location [km]",
        )

        # Number of bins
        # Min max ddepth for cmt plotting
        self.ddepth = (self.ndepth_in_m - self.odepth_in_m) / 1000.0
        self.maxddepth = np.max(self.ddepth)
        self.minddepth = np.min(self.ddepth)
        self.dd_absmax = np.max(
            np.abs(
                [
                    np.quantile(np.min(self.ddepth), 0.30),
                    np.quantile(np.min(self.ddepth), 0.70),
                ]
            )
        )
        self.maxdepth = np.max(self.ndepth_in_m) / 1000.0
        self.mindepth = np.min(self.ndepth_in_m) / 1000.0
        self.dmbins = np.linspace(-0.5, 0.5 + 0.5 / self.nbins, self.nbins)
        self.ddegbins = np.linspace(-0.1, 0.1 + 0.1 / self.nbins, self.nbins)
        self.dzbins = np.linspace(
            -self.dd_absmax, 2 * self.dd_absmax / self.nbins, self.nbins
        )
        self.dtbins = np.linspace(-10, 10 + 10 / self.nbins, self.nbins)

        # Get depth dependent cmap and norm
        self.depth_cmap, self.depth_norm, levels = get_level_norm_cmap(
            depth=self.odepth_in_m / 1000.0, cmap="rainbow_r", levels=None
        )

    def plot_cmts(self, legend=True):
        # Get axes (must be map axes)
        ax = plt.gca()

        minm = np.min(self.nmoment_magnitude)
        maxm = np.max(self.nmoment_magnitude)

        def sizefunc(x):
            return np.pi * (0.4 * (x - minm) / (maxm - minm) + 1) ** 8 + 8

        # Plot events
        scatter, ax, l1, l2 = plot_quakes(
            self.nlatitude,
            self.nlongitude,
            self.ndepth_in_m / 1000.0,
            self.nmoment_magnitude,
            ax=ax,
            cmap="rainbow_r",
            legend=legend,
            yoffsetlegend2=0.09,
            sizefunc=sizefunc,
        )
        ax.set_global()
        utils.plot_map(zorder=0, fill=True)
        utils.plot_label(ax, f"N: {self.N}", location=1, box=False, dist=0.0)

        return scatter, ax, l1, l2

    def plot_eps_nu(self):
        # Get axes
        ax = plt.gca()

        bins = np.arange(-0.5, 0.50001, 0.025)

        # Plot histogram GCMT3D
        plt.hist(
            self.oeps_nu[:, 0],
            bins=bins,
            edgecolor="k",
            facecolor=(0.3, 0.3, 0.8, 0.75),
            linewidth=0.75,
            label=self.oldlabel,
            histtype="stepfilled",
        )

        # Plot histogram GCMT3D+
        plt.hist(
            self.neps_nu[:, 0],
            bins=bins,
            edgecolor="k",
            facecolor=(0.3, 0.8, 0.3, 0.75),
            linewidth=0.75,
            label=self.newlabel,
            histtype="stepfilled",
        )
        plt.legend(
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
        label = (
            f"{self.oldlabel}\n"
            f"$\\mu$ = {np.mean(self.oeps_nu[:,0]):7.4f}\n"
            f"$\\sigma$ = {np.std(self.oeps_nu[:,0]):7.4f}\n"
            f"{self.newlabel}\n"
            f"$\\mu$ = {np.mean(self.neps_nu[:,0]):7.4f}\n"
            f"$\\sigma$ = {np.std(self.neps_nu[:,0]):7.4f}\n"
        )
        utils.plot_label(
            ax,
            label,
            location=2,
            box=False,
            fontdict=dict(fontsize="xx-small", fontfamily="monospace"),
        )
        utils.plot_label(
            ax, "CLVD-", location=6, box=False, fontdict=dict(fontsize="small")
        )
        utils.plot_label(
            ax, "CLVD+", location=7, box=False, fontdict=dict(fontsize="small")
        )
        utils.plot_label(
            ax, "DC", location=14, box=False, fontdict=dict(fontsize="small")
        )
        plt.xlabel(r"$\epsilon$")

    def omega_angle(self):
        omt = self.old.getvals(vtype="fulltensor")
        nmt = self.new.getvals(vtype="fulltensor")

        norm_omt = np.sqrt(np.sum(omt * omt, axis=tuple((1, 2))))
        norm_nmt = np.sqrt(np.sum(nmt * nmt, axis=tuple((1, 2))))

        mt_dot = np.sum(omt * nmt, axis=tuple((1, 2)))

        mt_angle = np.arccos(mt_dot / (norm_nmt * norm_omt))

        return mt_angle / np.pi * 180

    def gamma(self):
        ogamma = self.old.getvals(vtype="decomp", dtype="gamma")
        ngamma = self.new.getvals(vtype="decomp", dtype="gamma")

        return ogamma, ngamma

    def plot_omega_gamma_figure(self):
        angles = self.omega_angle()
        ogamma, ngamma = self.gamma()

        # Get numbers in segments
        N_10 = len(np.where(angles <= 10)[0])
        N10_20 = len(np.where((10 < angles) & (angles <= 20))[0])
        N20_30 = len(np.where((20 < angles) & (angles <= 30))[0])
        N30_ = len(np.where(30 < angles)[0])

        _, axes = plt.subplots(2, 1, figsize=(5, 4))
        plt.subplots_adjust(hspace=0.65)
        vals, bins, bc = axes[0].hist(angles, bins=150, color="darkgray")

        # Axes limits
        xmin, xmax = 0, 49.99
        ymin, ymax = 0, np.max(vals) * 1.25
        yann = np.max(vals) * 1.05
        axes[0].set_ylim(ymin, ymax)
        axes[0].set_xlim(xmin, xmax)

        # Annotations
        axes[0].vlines([10, 20, 30], 0, ymax, "k", lw=0.75, ls="--")
        axes[0].annotate(f"{N_10}", (5, yann), ha="center")
        axes[0].annotate(f"{N10_20}", (15, yann), ha="center")
        axes[0].annotate(f"{N20_30}", (25, yann), ha="center")
        axes[0].annotate(f"{N30_}", (40, yann), ha="center")
        axes[0].set_xlabel("$\\omega$")
        axes[0].set_ylabel("N")

        # Spine changes
        axes[0].tick_params(direction="inout", which="both", top=False, right=False)
        axes[0].spines["right"].set_visible(False)
        axes[0].spines["top"].set_visible(False)

        # manual arrowhead width and length
        hw = 1.0 / 15.0 * (ymax - ymin)
        hl = 1.0 / 40.0 * (xmax - xmin)
        lw = 1.0  # axis line width
        ohg = 0.5  # arrow overhang

        axes[0].arrow(
            xmax - 0.5 * hl,
            0.0,
            hl,
            0.0,
            fc="k",
            ec="k",
            lw=lw,
            head_width=hw,
            head_length=hl,
            overhang=ohg,
            length_includes_head=True,
            clip_on=False,
            head_starts_at_zero=False,
        )

        # Labels
        lplt.plot_label(
            axes[0],
            "a)",
            location=18,
            box=False,
            dist=0.025,
            fontdict=dict(fontsize="small"),
        )
        lplt.plot_label(
            axes[0],
            "Angular Change in Moment Tensor",
            location=14,
            box=False,
            dist=0.025,
            fontdict=dict(fontsize="small"),
        )

        # Plot Gamma subplot
        bins = np.linspace(-np.pi / 6, np.pi / 6, 100)
        oldlabel = (
            f"{self.oldlabel}\n"
            f"$\\mu$ = {np.mean(ogamma):7.4f}\n"
            f"$\\sigma$ = {np.std(ogamma):7.4f}"
        )
        newlabel = (
            f"{self.newlabel}\n"
            f"$\\mu$ = {np.mean(ngamma):7.4f}\n"
            f"$\\sigma$ = {np.std(ngamma):7.4f}"
        )

        # Plot histograms with labels
        ovals, _, _ = axes[1].hist(ogamma, bins=bins, color="lightgray", label=oldlabel)
        nvals, _, _ = axes[1].hist(
            ngamma, bins=bins, color="black", histtype="step", label=newlabel
        )

        # Legend
        leg = plt.legend(
            loc="upper right",
            frameon=False,
            labelspacing=1.0,
            prop=dict(size="x-small", family="monospace"),
            borderaxespad=0.0,
        )

        # Axes limits
        ymax = np.max((np.max(nvals), np.max(ovals))) * 1.1
        axes[1].set_ylim(0, ymax)
        axes[1].set_xlim(-np.pi / 6, np.pi / 6)
        axes[1].vlines([0], 0, ymax, "k", ls=":", lw=0.5)

        # Label locators
        axes[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 24))
        axes[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 12))
        major = lplt.Multiple(12, number=np.pi, latex="\pi")
        axes[1].xaxis.set_major_formatter(major.formatter)

        # Plot Axes, DC, CLVD labels
        lplt.plot_label(
            axes[1],
            "b)",
            location=18,
            box=False,
            dist=0.025,
            fontdict=dict(fontsize="small"),
        )
        lplt.plot_label(
            axes[1],
            "CLVD-",
            location=6,
            box=False,
            dist=0.01,
            fontdict=dict(fontsize="small"),
        )
        lplt.plot_label(
            axes[1],
            "CLVD+",
            location=7,
            box=False,
            dist=0.01,
            fontdict=dict(fontsize="small"),
        )
        lplt.plot_label(
            axes[1],
            "DC",
            location=14,
            box=False,
            dist=0.01,
            fontdict=dict(fontsize="small"),
        )
        axes[1].set_xlabel(r"$\gamma$")
        axes[1].set_ylabel("N")

    def plot_depth_v_ddepth(self):
        # Get axis
        ax = plt.gca()
        msize = 15

        # Sort the depth
        isort = np.argsort(self.odepth_in_m)[::-1]

        plt.scatter(
            self.ddepth[isort],
            self.odepth_in_m[isort] / 1000,
            c=self.depth_cmap(self.depth_norm(self.odepth_in_m[isort] / 1000.0)),
            s=msize,
            marker="o",
            alpha=0.5,
            edgecolors="none",
        )

        # Custom legend
        # classes = ['  <70 km', ' ', '>300 km']
        # colors = [(0.8, 0.2, 0.2), (0.2, 0.6, 0.8), (0.35, 0.35, 0.35)]
        # for cla, col in zip(classes, colors):
        #     plt.scatter([], [], c=[col], s=msize, label=cla, alpha=0.5,
        #                 edgecolors='none')
        # plt.legend(loc='lower left', frameon=False, fancybox=False,
        #            numpoints=1, scatterpoints=1, fontsize='x-small',
        #            borderaxespad=0.0, borderpad=0.5, handletextpad=0.2,
        #            labelspacing=0.2, handlelength=0.5,
        #            bbox_to_anchor=(0.0, 0.0))

        # Zero line
        plt.plot([0, 0], [0, np.max(self.odepth_in_m / 1000.0)], "-", lw=0.5, c="k")

        # Stats plots
        plotdict = dict(
            blines=dict(lw=1.0, color="k"),
            # median=dict(ls='', marker='o', c='k', markersize=2.5),
            # quantile=dict(ls='', markersize=3, c='k', marker='|'),
            mean=dict(ls="", marker="o", c="k", markersize=2.0),
            std=dict(ls="", markersize=4, c="k", marker="|"),
        )
        bins = np.logspace(1, 2.903, 8)
        plot_binnedstats(
            self.odepth_in_m[isort] / 1000,
            self.ddepth[isort],
            bins=bins,
            plotdict=plotdict,
            orientation="vertical",
            quantile=[0.25, 0.75],  # quantilemarkers=[9, 8]
            log=True,
        )

        # Axes properties
        plt.ylim(([10, np.max(self.odepth_in_m / 1000.0)]))
        plt.xlim(([np.min(self.ddepth), np.max(self.ddepth)]))
        ax.invert_yaxis()
        ax.set_yscale("log")
        plt.xlabel("Depth Change [km]")
        plt.ylabel("Depth [km]")

    def plot_spatial_distribution(
        self, parameter: str, outfile: Optional[str] = None, extent=None
    ):
        """Plots a 3x3 plots of distributions of changes at different depth
        ranges for a provided parameter.

        Parameters
        ----------
        parameter : str
            parameter to plot the changes of
        outfile : Optional[str], optional
            outputfile, by default None
        extent : Iterable, optional
            arraylike of 4 elements describing the map bounds. Just an input
            for cartopy's ax.set_extent, by default None, which makes the map
            a global map.
        """

        # Change backend if output is pdf
        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend("pdf")

        aspect = 10 / 9.0
        size = 8
        fig = plt.figure(figsize=(size * aspect, size))

        # Raise error if parameter doesnt exist.

        # Define levels on where to loo
        levels = [0.0, 10.0, 12.5, 15.0, 20.0, 30.0, 70.0, 120.0, 300.0, 800.0]

        if parameter == "location":
            # Get data for parameter in question
            olat = copy(self.olatitude)
            nlat = copy(self.nlatitude)
            olon = copy(self.olongitude)
            nlon = copy(self.nlongitude)
            dlat = nlat - olat
            dlon = nlon - olon
            dparam = utils.haversine(olon, olat, nlon, nlat)
            b = 90 - utils.bearing(olon, olat, nlon, nlat)
            print(np.min(b), np.mean(b), np.max(b))

        else:
            # Get data for parameter in question
            oparam = copy(getattr(self, "o" + parameter))
            nparam = copy(getattr(self, "n" + parameter))
            dparam = nparam - oparam

        if parameter == "eps_nu":
            oparam = oparam[:, 0]
            nparam = nparam[:, 0]
            dparam = dparam[:, 0]

        if parameter == "depth_in_m":
            oparam /= 1000.0
            nparam /= 1000.0
            dparam /= 1000.0

        if parameter in ["M0"]:
            dparam /= oparam
            dparam *= 100.0

        if parameter == "eps_nu":
            vmin, vmax = -0.2, 0.2
        else:
            dparam_absmax = np.max(
                np.abs([np.quantile(dparam, 0.05), np.quantile(dparam, 0.95)])
            )

            vmin = -dparam_absmax
            vmax = dparam_absmax

        # Location, we only have positive values so they change.
        if parameter == "location":
            vmin = 0
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap("inferno")
            quivers = []
        else:
            vcenter = 0
            norm = utils.MidpointNormalize(vmin=vmin, midpoint=vcenter, vmax=vmax)
            cmap = plt.get_cmap("seismic")

        # list of pos that have the given depth
        individualpos = []
        idx_sort = np.argsort(self.ndepth_in_m)
        pos = np.arange(0, len(self.ndepth_in_m))[idx_sort]

        def split(a, n):
            k, m = divmod(len(a), n)
            return (
                a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
            )

        individualpos = split(pos, 9)

        # Create size function with normalization
        def sizefunc(x):
            size = np.abs(x) / vmax
            wlsize = np.where((size < 0.2), 0.01, size)
            wlsize = np.where((size > 1.0), 1.0, wlsize)
            return 10 * ((1 + wlsize) ** 2)

        axes = []

        individualpos = []
        for _i in range(len(levels) - 1):
            pos = np.where(
                (levels[_i] < self.ndepth_in_m / 1000.0)
                & (self.ndepth_in_m / 1000.0 < levels[_i + 1])
            )
            if len(pos[0]) != 0:
                individualpos.append(pos[0])

        for _i, pos in enumerate(individualpos):
            # Get the eq's that are in the certain range
            dmin = np.min(self.ndepth_in_m[pos] / 1000.0)
            dmax = np.max(self.ndepth_in_m[pos] / 1000.0)

            # Create axes
            axes.append(plt.subplot(3, 3, _i + 1))
            axes[_i].set_title(
                f"{int(np.round(dmin)):3d} - {int(np.round(dmax)):3d} km",
            )
            # y=0.925)
            axes[_i].axis("off")

            # Create subaaxes
            mapinset = utils.axes_from_axes(
                axes[_i],
                _i,
                [0.0, 0.2, 1.0, 0.8],
                projection=Mollweide(central_longitude=self.central_longitude),
            )
            mapinset.set_global()

            # Plot map
            utils.plot_map(ax=mapinset, outline=False, borders=False)

            # DO scatter stuff
            if parameter == "location":
                # set displayed arrow length for longest arrow
                displayed_arrow_length = 25.0

                # calculate scale factor for quiver
                scale_factor = np.max(dparam) / displayed_arrow_length

                Q = mapinset.quiver(
                    self.olongitude[pos],
                    self.olatitude[pos],
                    dlon[pos],
                    dlat[pos],
                    dparam[pos],
                    angles=b[pos],
                    pivot="tail",
                    cmap=cmap,
                    norm=norm,
                    scale=2.0,
                    transform=PlateCarree(),  # units='xy', width=100,
                    width=0.005,
                    linewidth=0.1,
                    edgecolor="k",
                )
                quivers.append(Q)
            else:
                mapinset.scatter(
                    self.nlongitude[pos],
                    self.nlatitude[pos],
                    s=sizefunc(dparam[pos]),
                    c=dparam[pos],
                    transform=PlateCarree(),
                    cmap=cmap,
                    alpha=1.0,
                    norm=norm,
                    edgecolor="k",
                    linewidth=0.175,
                    zorder=10,
                )

            if extent is not None:
                mapinset.set_extent(extent)

            # Histogram axis
            inset = utils.axes_from_axes(axes[_i], _i, [0.2, 0.05, 0.6, 0.1])
            inset.spines["right"].set_visible(False)
            inset.spines["top"].set_visible(False)
            inset.spines["left"].set_visible(False)
            inset.tick_params(top=False, left=False, right=False, labelleft=False)
            inset.minorticks_off()
            if parameter == "eps_nu":
                xlim = [vmin, vmax]
            else:
                xlim = [np.min(dparam[pos]), np.max(dparam[pos])]
            inset.set_xlim(xlim)

            # Plot histogram
            bins = np.linspace(xlim[0], xlim[1], 20)
            self.plot_histogram(
                dparam[pos],
                bins,
                facecolor="darkgray",
                outline=False,
                ax=inset,
                stats=False,
            )

            # Automatically set eight of the histograms and plot corresponding
            # Vertical line at zero
            ylim = inset.get_ylim()
            inset.plot([0, 0], ylim, "k", lw=1.0)
            inset.plot(xlim, [0, 0], "k", lw=1.0)
            utils.plot_label(
                inset,
                f"#: {len(pos)}",
                fontdict=dict(fontsize="x-small"),
                box=False,
                dist=0.0,
                location=6,
            )

        plt.subplots_adjust(
            left=0.01, right=0.99, bottom=0.175, top=0.95, hspace=0.25, wspace=0.02
        )

        # Parameter is location use the normal colorbar
        if parameter == "location":
            # qk = mapinset.quiverkey(
            #     Q, 0.5, .5, 10.0, r'10 km', labelpos='N',
            #     coordinates="figure")

            cax = axes[-2].inset_axes([-0.5, -0.175, 2.0, 0.05])
            cbar = plt.colorbar(
                cax=cax,
                mappable=ScalarMappable(norm=norm, cmap=cmap),
                orientation="horizontal",
            )

            cbar.set_label(f"Change in {self.labeldict[parameter]}")

        else:
            # Axes for the scatter legend
            cax = utils.axes_from_axes(axes[-2], 100, [-0.25, -0.15, 1.5, 0.05])
            cax.axis("off")

            # Boundaries
            boundaries = np.linspace(vmin, vmax, 9)

            # Legend done
            legend = utils.scatterlegend(
                boundaries,
                cmap=cmap,
                norm=norm,
                sizefunc=sizefunc,
                loc="upper center",
                frameon=False,
                title=f"Change in {self.labeldict[parameter]}",
                title_fontsize="small",
                fontsize="x-small",
                lkw=dict(marker="o", markeredgewidth=0.175, markeredgecolor="k"),
                yoffset=-12.5 if outfile is not None else -50,  # For PDF output!
            )

        if outfile is not None:
            plt.savefig(outfile)
            plt.close(fig)
            plt.switch_backend(backend)

    def plot_summary(self, outfile: Optional[str] = None):
        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend("pdf")

        # Create figure handle
        fig = plt.figure(figsize=(11, 6))

        # Create subplot layout
        GS = GridSpec(3, 3)
        plt.subplots_adjust(wspace=0.3, hspace=0.75)
        # Plot events
        ax = fig.add_subplot(GS[:2, :2])
        ax.axis("off")
        pad, w, h = 0.025, 0.95, 0.825
        a = ax.get_position()
        iax_pos = [
            a.x1 - (w + pad) * a.width,
            a.y1 - (h + pad) * a.height,
            w * a.width,
            h * a.height,
        ]
        iax = fig.add_axes(
            iax_pos, projection=Mollweide(central_longitude=self.central_longitude)
        )
        iax.set_global()
        self.plot_cmts()
        utils.plot_label(ax, "a)", location=6, box=False)

        # Plot eps_nu change
        ax = fig.add_subplot(GS[0, 2])
        self.plot_eps_nu()
        utils.plot_label(ax, "b)", location=17, box=False)

        # Plot Depth v dDepth
        ax = fig.add_subplot(GS[1, 2])
        self.plot_depth_v_ddepth()
        utils.plot_label(ax, "c)", location=6, box=False)

        # Plot tshift histogram
        tbins = self.nbins
        # tbins = np.linspace(-0.5, 0.5, 100)
        ax = fig.add_subplot(GS[2, 0])
        self.plot_histogram(
            self.ntime_shift - self.otime_shift, tbins, facecolor="lightgray"
        )
        utils.remove_topright()
        plt.xlabel("Centroid Time Change [sec]")
        plt.ylabel("N", rotation=0, horizontalalignment="right")
        utils.plot_label(ax, "d)", location=6, box=False)

        # Plot Scalar Moment histogram
        Mbins = self.nbins
        # Mbins = np.linspace(-10, 10, 100)
        ax = fig.add_subplot(GS[2, 1])
        self.plot_histogram(
            (self.nM0 - self.oM0) / self.oM0 * 100,
            Mbins,
            facecolor="lightgray",
            statsleft=False,
        )
        utils.remove_topright()
        plt.xlabel("Scalar Moment Change [%]")
        plt.ylabel("N", rotation=0, horizontalalignment="right")
        utils.plot_label(ax, "e)", location=6, box=False)

        # Plot ddepth histogram
        zbins = self.nbins
        # zbins = np.linspace(-2.5, 2.5, 100)
        ax = fig.add_subplot(GS[2, 2])
        self.plot_histogram(self.ddepth, zbins, facecolor="lightgray", statsleft=True)
        utils.remove_topright()
        plt.xlabel("Depth Change [km]")
        plt.ylabel("N", rotation=0, horizontalalignment="right")
        utils.plot_label(ax, "f)", location=6, box=False)

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)

    @staticmethod
    def get_change(o, n, d: bool, f: bool):
        """Getting the change values, if ``d`` is ``False`` the function just
        returns (o, n).

        Parameters
        ----------
        o : arraylike
            old values
        n : arraylike
            new values
        d : bool
            compute change
        f : bool
            compute fractional change

        Returns
        -------
        tuple
            old, new values
        """

        if d:
            if f:
                o_out = (n - o) / o
            else:
                o_out = n - o
            n_out = o_out

        else:
            o_out = o
            n_out = n
        return o_out, n_out

    def plot_2D_scatter(
        self,
        param1="depth",
        param2="M0",
        d1: bool = True,
        d2: bool = True,
        f1: bool = False,
        f2: bool = False,
        xlog: bool = False,
        ylog: bool = False,
        xrange: Optional[list] = None,
        yrange: Optional[list] = None,
        xinvert: bool = False,
        yinvert: bool = False,
        nbins: int = 40,
        outfile: Optional[str] = None,
    ):
        """
        latitude = lat
        longitude = lon
        depth = depth
        M0 = M0
        moment_magnitude = moment_mag
        eps_nu = eps_nu

        d? meaning the change in the parameter, boolean

        f? meaning fractional change, boolean, only used of d? True,
        """

        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend("pdf")
            plt.figure(figsize=(4.5, 3))

        # Get first parameters
        old1 = copy(getattr(self, "o" + param1))
        new1 = copy(getattr(self, "n" + param1))

        # Get second parameters
        old2 = copy(getattr(self, "o" + param2))
        new2 = copy(getattr(self, "n" + param2))

        # Compute the values to be plotted
        o1p, n1p = self.get_change(old1, new1, d1, f1)
        o2p, n2p = self.get_change(old2, new2, d2, f2)

        if param1 == "depth_in_m":
            o1p /= 1000.0
            n1p /= 1000.0
        if param2 == "depth_in_m":
            o2p /= 1000.0
            n2p /= 1000.0

        # Plot 2D scatter histograms
        if d1 and d2:
            label = " "
            axscatter, _, _ = scatter_hist(
                n1p,
                n2p,
                nbins,
                label=label,
                histc=(0.4, 0.4, 1.0),
                fraction=0.85,
                mult=False,
            )

        elif (d1 and not d2) or (not d1 and d2):
            label = " "
            axscatter, _, _ = scatter_hist(
                n1p,
                n2p,
                nbins,
                label=label,
                histc=(0.4, 0.4, 1.0),
                fraction=0.85,
                mult=False,
            )

        else:
            labels = ["O", "N"]
            axscatter, _, _ = scatter_hist(
                [o1p, n1p],
                [o2p, n2p],
                nbins,
                label=labels,
                histc=[(0.3, 0.3, 0.9), (0.9, 0.3, 0.3)],
                fraction=0.85,
                mult=True,
            )

        # Possibly do stuff with axes
        if d1 and not d2:
            xlabel = "d" + self.labeldict[param1]
            ylabel = self.labeldict[param2]
        elif not d1 and d2:
            xlabel = self.labeldict[param1]
            ylabel = "d" + self.labeldict[param2]
        elif not d1 and not d2:
            xlabel = self.labeldict[param1]
            ylabel = self.labeldict[param2]
        else:
            xlabel = "d" + self.labeldict[param1]
            ylabel = "d" + self.labeldict[param2]

        # add labels to plot
        axscatter.set_xlabel(xlabel)
        axscatter.set_ylabel(ylabel)

        if ylog:
            axscatter.set_yscale("log")
        if xlog:
            axscatter.set_xscale("log")

        if xrange is not None:
            axscatter.set_xlim(xrange)
        if yrange is not None:
            axscatter.set_ylim(yrange)

        if xinvert:
            axscatter.invert_xaxis()
        if yinvert:
            axscatter.invert_yaxis()

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)
            plt.close()

    def plot_depth_v_eps_nu(self, outfile=None):
        if outfile is not None:
            backend = plt.get_backend()
            plt.switch_backend("pdf")
            plt.figure(figsize=(4.5, 3))

        axscatter, _, _ = scatter_hist(
            [self.oeps_nu[:, 0], self.neps_nu[:, 0]],
            [self.odepth_in_m / 1000.0, self.ndepth_in_m / 1000.0],
            self.nbins,
            label=[self.oldlabel, self.newlabel],
            histc=[(0.4, 0.4, 1.0), (1.0, 0.4, 0.4)],
            fraction=0.85,
            ylog=True,
            mult=True,
        )
        axscatter.invert_yaxis()
        axscatter.set_xlim((-0.5, 0.5))
        ylim = axscatter.get_ylim()
        axscatter.set_ylim(
            (ylim[0], 2.5)
        )  # 0.95*np.min((np.min(self.ndepth_in_m/1000.0),
        #        np.min(self.odepth_in_m/1000.0)))))
        axscatter.set_yscale("log")

        # Binned stats
        gray = 0.35 * np.ones(3)
        plotdict_before = dict(
            blines=dict(lw=2.0, color=gray),
            # median=dict(ls='', marker='o', c='k', markersize=2.5),
            # quantile=dict(ls='', markersize=3, c='k', marker='|'),
            mean=dict(ls="", marker="o", c=gray, markersize=4.0),
            std=dict(ls="", markersize=15, c=gray, marker="|"),
        )
        plotdict_after = dict(
            blines=dict(lw=3.0, color="k"),
            # median=dict(ls='', marker='o', c='k', markersize=2.5),
            # quantile=dict(ls='', markersize=3, c='k', marker='|'),
            mean=dict(ls="", marker="o", c="k", markersize=6.0),
            std=dict(ls="", markersize=20, c="k", marker="|"),
        )

        bins = np.logspace(0, 2.903, 8)
        plot_binnedstats(
            self.odepth_in_m / 1000.0,
            self.oeps_nu[:, 0],
            bins=bins,
            plotdict=plotdict_before,
            orientation="vertical",
            quantile=[0.25, 0.75],  # quantilemarkers=[9, 8]
            log=True,
        )
        plot_binnedstats(
            self.ndepth_in_m / 1000.0,
            self.neps_nu[:, 0],
            bins=bins,
            plotdict=plotdict_after,
            orientation="vertical",
            quantile=[0.25, 0.75],  # quantilemarkers=[9, 8]
            log=True,
        )

        # Plot clvd labels
        utils.plot_label(
            axscatter, "CLVD-", location=11, box=False, fontdict=dict(fontsize="small")
        )
        utils.plot_label(
            axscatter, "CLVD+", location=10, box=False, fontdict=dict(fontsize="small")
        )
        utils.plot_label(
            axscatter, "DC", location=16, box=False, fontdict=dict(fontsize="small")
        )
        axscatter.tick_params(labelbottom=False)
        plt.ylabel("Depth [km]")

        if outfile is not None:
            plt.savefig(outfile)
            plt.switch_backend(backend)
            plt.close()

    def plot_histogram(
        self,
        ddata,
        n_bins,
        facecolor=(0.7, 0.2, 0.2),
        alpha=1,
        chi=False,
        wmin=None,
        statsleft: bool = False,
        label: str = None,
        stats: bool = True,
        ax=None,
        outline: bool = True,
        CI: bool = False,
    ):
        """Plots histogram of input data."""

        if wmin is not None:
            print(f"Datamin: {np.min(ddata)}")
            ddata = ddata[np.where(ddata >= wmin)]
            print(f"Datamin: {np.min(ddata)}")

        # The histogram of the data
        if ax is None:
            ax = plt.gca()

        n, bins, _ = ax.hist(
            ddata,
            n_bins,
            facecolor=facecolor,
            edgecolor=facecolor,
            alpha=alpha,
            label=label,
        )
        if outline:
            _, _, _ = ax.hist(ddata, n_bins, color="k", histtype="step")
        text_dict = {
            "fontsize": "x-small",
            "verticalalignment": "top",
            "horizontalalignment": "right",
            "transform": ax.transAxes,
            "zorder": 100,
            "family": "monospace",
        }

        if stats:
            # Get datastats
            datamean = np.mean(ddata)
            datastd = np.std(ddata)

            # Check if mean closer to right edge or left edge and putt stats
            # wherever there is more room
            xmin, xmax = ax.get_xlim()
            if np.abs(datamean - xmin) > np.abs(xmax - datamean):
                statsleft = True
            else:
                statsleft = False

            if statsleft:
                text_dict["horizontalalignment"] = "left"
                posx = 0.03
            else:
                posx = 0.97

            ax.text(
                posx,
                0.97,
                f"$\\mu$ = {datamean:5.2f}\n" f"$\\sigma$ = {datastd:5.2f}",
                **text_dict,
            )

        if CI:
            ci_norm = {
                "80": 1.282,
                "85": 1.440,
                "90": 1.645,
                "95": 1.960,
                "99": 2.576,
                "99.5": 2.807,
                "99.9": 3.291,
            }
            if chi:
                Zval = ci_norm["90"]
            else:
                Zval = ci_norm["95"]

            mean = np.mean(ddata)
            pmfact = Zval * np.std(ddata)
            nCI = [mean - pmfact, mean + pmfact]
            # if we are only concerned about the lowest values the more
            # the better:
            if wmin is not None:
                nCI[1] = np.max(ddata)
                if nCI[0] < wmin:
                    nCI[0] = wmin
            minbox = [np.min(bins), 0]
            minwidth = (nCI[0]) - minbox[0]
            maxbox = [nCI[1], 0]
            maxwidth = np.max(bins) - maxbox[0]
            height = np.max(n) * 1.05

            boxdict = {
                "facecolor": "w",
                "edgecolor": None,
                "alpha": 0.6,
            }
            minR = Rectangle(minbox, minwidth, height, **boxdict)
            maxR = Rectangle(maxbox, maxwidth, height, **boxdict)
            ax.add_patch(minR)
            ax.add_patch(maxR)

            return nCI
        else:
            return None

    def filter(self, maxdict: dict = dict(), mindict: dict = dict()):
        """This uses two dictionaries as inputs. One dictionary for
        maximum values and one dictionary that contains min values of the
        elements to filter. To do that we create a dictionary containing
        the attributes and properties of
        :class:``lwsspy.seismo.source.CMTSource``.

        List of Attributes and Properties
        -------------------------

        .. literal::

            origin_time
            pde_latitude
            pde_longitude
            pde_depth_in_m
            mb
            ms
            region_tag
            eventname
            cmt_time
            half_duration
            latitude
            longitude
            depth_in_m
            m_rr
            m_tt
            m_pp
            m_rt
            m_rp
            m_tp
            M0
            moment_magnitude
            time_shift

        Example
        -------

        Let's filter the catalog to only contain events with a maximum depth
        of 20km.

        >>> maxfilterdict = dict(depth_in_m=20000.0)
        >>> cmtcat = CMTCatalog.from_files("CMTfiles/*")
        >>> filtered_cat = cmtcat.filter(maxdict=maxfilterdict)

        will returns a catalog with events shallower than 20.0 km.
        """

        # Create new list of cmts
        oldlist, newlist = deepcopy(self.old.cmts), deepcopy(self.new.cmts)

        # percentage parameters
        pparams = [
            "m_rr",
            "m_tt",
            "m_pp",
            "m_rt",
            "m_rp",
            "m_tp",
            "M0",
            "moment_magnitude",
        ]

        oldpoppedlist = []
        newpoppedlist = []

        # First maxvalues
        for key, value in maxdict.items():
            # Create empty pop set
            popset = set()
            print(key, value)
            # Check CMTs that are below threshold for key
            for _i, (_ocmt, _ncmt) in enumerate(zip(oldlist, newlist)):
                oldval = getattr(_ocmt, key)
                newval = getattr(_ncmt, key)
                if key in pparams:
                    if np.abs((newval - oldval) / oldval) > value:
                        popset.add(_i)
                else:
                    if np.abs(newval - oldval) > value:
                        popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                oldpoppedlist.append(oldlist.pop(_i))
                newpoppedlist.append(newlist.pop(_i))

        # First minvalues
        for key, value in mindict.items():
            # Create empty pop set
            popset = set()

            # Check CMTs that are below threshold for key
            for _i, (_ocmt, _ncmt) in enumerate(zip(oldlist, newlist)):
                oldval = getattr(_ocmt, key)
                newval = getattr(_ncmt, key)
                if key in pparams:
                    if np.abs((newval - oldval) / oldval) < value:
                        popset.add(_i)
                else:
                    if np.abs(newval - oldval) < value:
                        popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                oldpoppedlist.append(oldlist.pop(_i))
                newpoppedlist.append(newlist.pop(_i))

        return (
            CompareCatalogs(
                CMTCatalog(oldlist),
                CMTCatalog(newlist),
                oldlabel=self.oldlabel,
                newlabel=self.newlabel,
                nbins=self.nbins,
            ),
            CompareCatalogs(
                CMTCatalog(oldpoppedlist),
                CMTCatalog(newpoppedlist),
                oldlabel=self.oldlabel,
                newlabel=self.newlabel,
                nbins=self.nbins,
            ),
        )

    def print_paramater_change(self, param="depth_in_m"):
        # percentage parameters
        pparams = [
            "m_rr",
            "m_tt",
            "m_pp",
            "m_rt",
            "m_rp",
            "m_tp",
            "M0",
            "moment_magnitude",
        ]

        for _o, _n in zip(self.old, self.new):
            oval = getattr(_o, param)
            nval = getattr(_n, param)

            if param in pparams:
                dval = 100 * np.abs((nval - oval) / oval)
                mod = "%"
                if param == "M0":
                    nval = nval / oval
                    oval = 1.0

            else:
                dval = nval - oval
                mod = ""

            string = f'{_o.eventname + ":":20} {oval:10.2f} -> {nval:10.2f}   delta: {dval:10.2f} {mod}'
            print(string)


def bin():
    import argparse

    utils.updaterc()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--old", dest="old", help="Old Catalog", required=True, type=str
    )
    parser.add_argument(
        "-n", "--new", dest="new", help="New Catalog", required=True, type=str
    )
    parser.add_argument(
        "-d",
        "--outdir",
        dest="outdir",
        help="Directory to place outputs in",
        type=str,
        default=".",
    )
    parser.add_argument(
        "-w",
        "--write-cats",
        dest="write",
        help="Write catalogs to file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--spatial",
        dest="spatial",
        help="Create plots for spatial distribution",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-ol",
        "--old-label",
        dest="oldlabel",
        help="Old label",
        required=True,
        type=str or None,
    )
    parser.add_argument(
        "-nl",
        "--new-label",
        dest="newlabel",
        help="New label",
        required=True,
        type=str or None,
    )
    args = parser.parse_args()

    # Get catalogs
    old = CMTCatalog.load(args.old)
    new = CMTCatalog.load(args.new)
    new, newp = new.filter(mindict=dict(depth_in_m=5000.0))

    print("Old:", len(old.cmts))
    print("New:", len(new.cmts))
    print("New:", len(newp.cmts))

    # Get overlaps
    ocat, ncat = old.check_ids(new)

    print("After checkid:")
    print("  Old:", len(ocat.cmts))
    print("  New:", len(ncat.cmts))

    # Writing
    if args.write:
        ocat.save(os.path.join(args.outdir, args.oldlabel + ".pkl"))
        ncat.save(os.path.join(args.outdir, args.newlabel + ".pkl"))

    # Compare Catalog

    CC = CompareCatalogs(
        old=ocat, new=ncat, oldlabel=args.oldlabel, newlabel=args.newlabel, nbins=25
    )
    # plt.figure(figsize=(4.5, 3))
    # CC.plot_2D_scatter(param1="moment_magnitude", param2="depth_in_m", d1=False,
    #                    d2=False, xlog=False, ylog=True, yrange=[3, 800],
    #                    yinvert=True)
    # plt.figure(figsize=(4.5, 3))
    # CC.plot_2D_scatter(param1="depth_in_m", param2="depth_in_m", d1=True,
    #                    d2=False, xlog=False, ylog=False, yrange=[0, 700],
    #                    yinvert=True)
    # plt.figure(figsize=(4.5, 3))
    # CC.plot_2D_scatter(param1="depth_in_m", param2="time_shift", d1=True,
    #                    d2=True)
    # plt.figure(figsize=(4.5, 3))
    # CC.plot_2D_scatter(param1="time_shift", param2="depth_in_m", d1=True,
    #                    d2=False, ylog=False, yrange=[0, 700],
    #                    yinvert=True)
    # plt.figure(figsize=(4.5, 3))
    # CC.plot_depth_v_eps_nu()

    # plt.show(block=True)

    # Filter for a minimum depth larger than zero
    CC, CC_pop = CC.filter(
        maxdict={"M0": 1.5, "latitude": 0.4, "longitude": 0.4, "depth_in_m": 30000.0}
    )
    # for ocmt, ncmt in zip(CC.old, CC.new):
    #     print(f"\n\nOLD: {(ncmt.depth_in_m - ocmt.depth_in_m)/1000.0}")
    #     print(ocmt)
    #     print(" ")
    #     print("NEW")
    #     print(ncmt)
    # print(len(CC.new))
    # extent = -80, -60, -10, -30
    # extent = None

    # Comparison figures
    # CC.plot_slab_map(extent=extent)
    # outfile=os.path.join(
    #     args.outdir, "catalog_slab_map.pdf"))

    CC.plot_summary(outfile=os.path.join(args.outdir, "catalog_comparison.pdf"))
    CC.plot_depth_v_eps_nu(outfile=os.path.join(args.outdir, "depth_v_sourcetype.pdf"))

    if args.spatial:
        spatial_dir = os.path.join(os.path.join(args.outdir, "spatial_changes"))

        if os.path.exists(spatial_dir) is False:
            os.mkdir(spatial_dir)

        CC.plot_spatial_distribution(
            "depth_in_m", outfile=os.path.join(spatial_dir, "spatial_depth.pdf")
        )
        CC.plot_spatial_distribution(
            "time_shift", outfile=os.path.join(spatial_dir, "spatial_time_shift.pdf")
        )
        CC.plot_spatial_distribution(
            "M0", outfile=os.path.join(spatial_dir, "spatial_M0.pdf")
        )
        CC.plot_spatial_distribution(
            "eps_nu", outfile=os.path.join(spatial_dir, "spatial_eps.pdf")
        )
        CC.plot_spatial_distribution(
            "latitude", outfile=os.path.join(spatial_dir, "spatial_lat.pdf")
        )

        CC.plot_spatial_distribution(
            "longitude", outfile=os.path.join(spatial_dir, "spatial_lon.pdf")
        )

        CC.plot_spatial_distribution(
            "location", outfile=os.path.join(spatial_dir, "spatial_location.pdf")
        )
