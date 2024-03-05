# %%

import os
from glob import glob
import pygmt
import typing as tp
from copy import deepcopy
import numpy as np
import cmt3d
import cmt3d.ioi as ioi
import cmt3d.viz as viz
import cmt3d.viz.utils as utils
from cmt3d.viz.compare_catalogs import CompareCatalogs
from cmt3d import CMTCatalog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

utils.updaterc()
# %%
# Make the catalogs to compare the events
# gcmt_files = glob(os.path.join("events", "gcmt", "*"))
# gcmt_cat = cmt3d.CMTCatalog.from_file_list(gcmt_files)

# %%

# gcmt = obspy.read_events("gcmtcatalog.ndk")

# %%
# Convert to cmt3d catalog
# gcmt_cat = CMTCatalog([cmt3d.CMTSource.from_event(ev) for ev in gcmt])

# %%
# gcmt_cat.save("gcmtcatalog.pkl")
gcmt_cat = CMTCatalog.from_file_list(glob("events/gcmt/*"))
cmt3d_cat = cmt3d.CMTCatalog.from_file_list(glob("events/cmt3d/*"))
cmt3dp_cat = cmt3d.CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))


# %%


# Choose minimimum and maximum latitude and longitude around south america
# to filter the catalog
minlat = -60
maxlat = 20
minlon = -120
maxlon = -20

# Choose minium and maximum latitude and longitude around fiji tonga subduction
# zone to filter the catalog
# minlat = -30
# maxlat = 0
# minlon = 160
# maxlon = 190

mindict = dict(latitude=minlat, longitude=minlon)
maxdict = dict(latitude=maxlat, longitude=maxlon)

sa_cat, _ = gcmt_cat.filter(maxdict=maxdict, mindict=mindict)

# %%
# Split the catalog into the different mechanisms with the null value threshold
# set to 1.0 to include all NDC events
split_cat = sa_cat.split_to_mechanism(
    thrust_null_value_threshold=1.0,
    normal_null_value_threshold=1.0,
    strike_slip_null_value_threshold=1.0,
)

# %%
# Store focal mechanism parameters in a dictionary based on the Aki & Richards
# convention


def cat2meca(cat: CMTCatalog):
    focal_mechanisms = []

    for ev in cat:
        # Get exponent to scale moment tensor
        exp = np.ceil(np.log10(ev.M0))

        # Append the focal mechanism parameters to the list
        focal_mechanisms.append(
            [
                ev.longitude,
                ev.latitude,
                ev.depth_in_m / 1000.0,
                ev.m_rr / exp,
                ev.m_tt / exp,
                ev.m_pp / exp,
                ev.m_rt / exp,
                ev.m_rp / exp,
                ev.m_tp / exp,
                exp,
            ]
        )

    # Make numpy array
    focal_mechanisms = np.array(focal_mechanisms)

    return focal_mechanisms


# %%


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
            focal_mechanisms = cat2meca(cat)
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


# %%


def split_cat_mech_depth(
    cat,
    ranges: tp.Dict[str, tp.Tuple[int, int, int]] = dict(
        shallow=(0, 70, 10), intermediate=(70, 200, 20), deep=(200, 900, 100)
    ),
):
    from collections import OrderedDict

    splitcat = OrderedDict()

    for i, (_name, (dlow, dhigh, dd)) in enumerate(ranges.items()):
        # Set range for this depth and name
        if _name not in splitcat:
            splitcat[_name] = dict()
            splitcat[_name]["range"] = (dlow, dhigh, dd)

        # Filter catalog by depth
        _tcat, _ = cat.filter(
            maxdict=dict(depth_in_m=dhigh * 1000),
            mindict=dict(depth_in_m=dlow * 1000),
        )

        # Split the catalog into the different mechanisms with the null value threshold
        # set to 1.0 to include all NDC events
        splitcat[_name]["catalogs"] = _tcat.split_to_mechanism(
            thrust_null_value_threshold=1.0,
            normal_null_value_threshold=1.0,
            strike_slip_null_value_threshold=1.0,
        )

        # Sort events from shallow to deep for each catalog
        for name, _scat in splitcat[_name]["catalogs"].items():
            # Sort the catalog by depth
            _scat.sort(key="depth_in_m")

    return splitcat


depth_split_cat = split_cat_mech_depth(gcmt_cat)
depth_split_cat_cmt3d = split_cat_mech_depth(cmt3d_cat)
depth_split_cat_cmt3dp = split_cat_mech_depth(cmt3dp_cat)


# %%

import pygmt


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
                focal_mechanisms = cat2meca(_scat)[::-1, :]

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


# %%
fig = plot_split_cat_depth(depth_split_cat)
fig.savefig("gcmt_depth_map_subselection.pdf")
# %%
fig = plot_split_cat_depth(depth_split_cat_cmt3dp)
fig.savefig("cmt3d_depth_map_subselection.pdf")
# %%
# Use the catalog with the depth split to plot distributions of the DC vs. NDC
# distributions, use gamma here


def plot_split_gamma(depth_split_cat):
    """Takes in the catalog organized by depth and mechanism to plot the
    distributions in terms of gamma and depth

    Parameters
    ----------
    depth_split_cat : _type_
        _description_
    """
    utils.updaterc()

    # Get number of depth ranges
    nrows = len(depth_split_cat)

    # Set up figure
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(nrows, 3, figure=fig, hspace=0.1, wspace=0.2)

    # Set up axes
    axes = []
    for i in range(nrows):
        axesrow = []
        for j in range(3):
            axesrow.append(fig.add_subplot(gs[i, j]))
        axes.append(axesrow)

    # Set up the labels
    labels = "abcdefghi"

    counter = 0

    for i, (dname, rangd) in enumerate(depth_split_cat.items()):
        dlow, dhigh, dd = rangd["range"]

        # Get the
        split_cat = rangd["catalogs"]

        # Setup the bins
        bins = np.linspace(-np.pi / 6, np.pi / 6, 50, endpoint=True)

        # Label
        label = "GCMT"

        for j, (name, cat) in enumerate(split_cat.items()):
            # Get the right axes instance
            ax = axes[i][j]

            # Get gamma for the catalog
            gamma = cat.getvals(vtype="decomp", dtype="gamma")

            # Plot histogram GCMT3D
            nvals, _, _ = ax.hist(
                gamma,
                bins=bins,
                facecolor="lightgrey",
                linewidth=0.75,
                label=label,
            )

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
                f"{label}\n"
                f"$\\mu$ = {np.mean(gamma):7.4f}\n"
                f"$\\sigma$ = {np.std(gamma):7.4f}\n"
            )
            utils.plot_label(
                ax,
                statlabel,
                location=2,
                box=False,
                fontdict=dict(fontsize="xx-small", fontfamily="monospace"),
            )

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
                plt.title(name.capitalize(), fontsize="small")

            # Axes limits
            ymax = np.max(np.max(nvals)) * 1.1
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

    plt.show()


plot_split_gamma(depth_split_cat)

# %%

ocat, ncat = gcmt_cat.check_ids(cmt3d_cat)

# %%
# ocat_split = split_cat_mech_depth(ocat)
# ncat_split = split_cat_mech_depth(ncat)

# %%
x = """Takes in the catalog organized by depth and mechanism to plot the
distributions in terms of gamma and depth

Parameters
----------
depth_split_cat : _type_
    _description_
"""


def plot_split_gamma_compare(
    split_cat, compare_cat=None, label="GCMT", compare_label="CMT3D+"
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
                    dist=0.125,
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


# %%
plot_split_gamma_compare(
    depth_split_cat,
    compare_cat=depth_split_cat_cmt3d,
    label="GCMT",
    compare_label="CMT3D",
)
plt.savefig("gcmt_cmt3d_gamma_compare.pdf", transparent=True)

plot_split_gamma_compare(
    depth_split_cat,
    compare_cat=depth_split_cat_cmt3dp,
    label="GCMT",
    compare_label="CMT3D+",
)
plt.savefig("gcmt_cmt3d+_gamma_compare.pdf", transparent=True)


plot_split_gamma_compare(
    depth_split_cat_cmt3d,
    compare_cat=depth_split_cat_cmt3dp,
    label="CMT3D",
    compare_label="CMT3D+",
)
plt.savefig("cmt3d_cmt3d+_gamma_compare.pdf", transparent=True)


# %%
plot_split_gamma_compare(depth_split_cat, label="GCMT")
plt.savefig("gcmt_gamma.pdf", transparent=True)


# %%
# Setup new ranges for the depth split to compare only shallow strike slip events

ranges = {
    "10-20 km": (0, 20, 2),
    "20-30 km": (20, 30, 1),
    "30-50 km": (30, 50, 2),
    # "50-70 km": (50, 70, 2),
}

ranges = {
    "5-12 km": (5, 15, 1),
    "12-20 km": (15, 20, 0.5),
    "20-30 km": (20, 30, 1),
    # "50-70 km": (50, 70, 2),
}

# Setup the new ranges
shallow_split_gcmt = split_cat_mech_depth(gcmt_cat, ranges=ranges)
shallow_split_cat_cmt3d = split_cat_mech_depth(cmt3d_cat, ranges=ranges)
shallow_split_cat_cmt3dp = split_cat_mech_depth(cmt3dp_cat, ranges=ranges)

# %%

plot_split_gamma_compare(
    shallow_split_gcmt,
    compare_cat=shallow_split_cat_cmt3d,
    label="GCMT",
    compare_label="CMT3D",
)
plt.savefig("gcmt_cmt3d_gamma_compare_shallow.pdf", transparent=True)

plot_split_gamma_compare(
    shallow_split_gcmt,
    compare_cat=shallow_split_cat_cmt3dp,
    label="GCMT",
    compare_label="CMT3D+",
)
plt.savefig("gcmt_cmt3d+_gamma_compare_shallow.pdf", transparent=True)


plot_split_gamma_compare(
    shallow_split_cat_cmt3d,
    compare_cat=shallow_split_cat_cmt3dp,
    label="CMT3D",
    compare_label="CMT3D+",
)
plt.savefig("cmt3d_cmt3d+_gamma_compare_shallow.pdf", transparent=True)


# %%

# ranges = {
#     "5-10 km": (5, 10, 1),
#     "10-15 km": (10, 15, 1),
#     "15-20 km": (15, 20, 1),
#     # "50-70 km": (50, 70, 2),
# }

# # Setup the new ranges
# shallow_split_cat_cmt3dp = split_cat_mech_depth(cmt3dp_cat, ranges=ranges)


# %%


# Plotting geographical distribution of shallow gamma

# Get gamma for the catalog
for range, _ in ranges.items():
    ss_shallow_cat = shallow_split_cat_cmt3dp[range]["catalogs"]["strike-slip"]
    gamma = ss_shallow_cat.getvals(vtype="decomp", dtype="gamma")
    latitudes = ss_shallow_cat.getvals(vtype="latitude")
    longitudes = ss_shallow_cat.getvals(vtype="longitude")
    magnitudes = ss_shallow_cat.getvals(vtype="moment_magnitude")

    def hist2d_scatter(
        longitude, latitude, data, dlat=3, dlon=3, region=None, mincount=3
    ):

        if not region:
            region = [-180, 180, -90, 90]

        # Get the sparate values
        minlon, maxlon, minlat, maxlat = region

        # Get the bin edges
        lonbins = np.arange(minlon, maxlon + dlon, dlon)
        latbins = np.arange(minlat, maxlat + dlon, dlat)

        # Create the 2D histogram
        hist, xedges, yedges = np.histogram2d(
            longitude, latitude, bins=(lonbins, latbins), weights=data
        )

        # Counts the number of events in each bin
        counts, _, _ = np.histogram2d(longitude, latitude, bins=(lonbins, latbins))

        # Divide the sum by the counts to get the mean, but check whether counts are zero
        a = np.empty(hist.shape)
        a[:] = np.nan
        means = np.divide(hist, counts, out=a, where=counts >= mincount)

        # Get bin centers
        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2

        # Create the meshgrid
        Y, X = np.meshgrid(yc, xc)

        x, y, h = X.flatten(), Y.flatten(), means.flatten()

        idx = ~np.isnan(h)

        return x[idx], y[idx], h[idx]

    lon, lat, m = hist2d_scatter(longitudes, latitudes, gamma)

    #  plot scatter map of gamma awith a diverging colormap of the gamma values as
    #  a function of longitude and latitude using pygmt
    fig = pygmt.Figure()
    pygmt.makecpt(cmap="magma", series=[0, 800, 10])

    projection = "W12c"
    region = "g"
    pygmt.makecpt(cmap="polar", series=[-np.pi / 6, np.pi / 6, np.pi / 27])
    fig.coast(
        region=region,
        projection=projection,
        land="gray90",
        water="white",
        shorelines=True,
        frame=["ag"],  # , "+t" + range + " Strike-Slip"],
    )
    fig.plot(
        x=lon,
        y=lat,
        size=np.ones_like(lon) * 0.1,
        fill=m,
        cmap=True,
        style="cc",
        pen="black",
    )
    fig.colorbar(frame=["af+l@~g", "-Bx1pi8f1pi16"])
    # Pass the focal mechanism data through the spec parameter. In addition provide
    # scale, event location, and event depth
    # fig.text(
    #     x=0,
    #     y=0,
    #     text=name,
    #     font="Helvetica-Bold",
    #     justify="LM",
    #     offset="0.5c",
    #     pen="black",
    # )

    # fig.colorbar()
    fig.show()

    fig.savefig(f"shallow_gamma_{range.replace(' ', '_')}.pdf")

# %%
