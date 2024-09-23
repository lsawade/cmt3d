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
import cmt3d.viz.catalog_utils as cutils
import cmt3d.viz.catalog_plot_utils as cputils
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


depth_split_cat = cutils.split_cat_mech_depth(gcmt_cat)
depth_split_cat_cmt3d = cutils.split_cat_mech_depth(cmt3d_cat)
depth_split_cat_cmt3dp = cutils.split_cat_mech_depth(cmt3dp_cat)


# %%
fig = cputils.plot_split_cat_depth(depth_split_cat)
fig.savefig("gcmt_depth_map_subselection.pdf")
# %%
fig = cputils.plot_split_cat_depth(depth_split_cat_cmt3dp)
fig.savefig("cmt3d_depth_map_subselection.pdf")

# %%

ocat, ncat = gcmt_cat.check_ids(cmt3d_cat)

# %%
# ocat_split = split_cat_mech_depth(ocat)
# ncat_split = split_cat_mech_depth(ncat)


# %%
cputils.plot_split_gamma_compare(
    depth_split_cat,
    compare_cat=depth_split_cat_cmt3d,
    label="GCMT",
    compare_label="CMT3D",
)
plt.savefig("gcmt_cmt3d_gamma_compare.pdf", transparent=True)

cputils.plot_split_gamma_compare(
    depth_split_cat,
    compare_cat=depth_split_cat_cmt3dp,
    label="GCMT",
    compare_label="CMT3D+",
)
plt.savefig("gcmt_cmt3d+_gamma_compare.pdf", transparent=True)

cputils.plot_split_gamma_compare(
    depth_split_cat_cmt3d,
    compare_cat=depth_split_cat_cmt3dp,
    label="CMT3D",
    compare_label="CMT3D+",
)
plt.savefig("cmt3d_cmt3d+_gamma_compare.pdf", transparent=True)


# %%
cputils.plot_split_gamma_compare(depth_split_cat, label="GCMT")
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
shallow_split_gcmt = cutils.split_cat_mech_depth(gcmt_cat, ranges=ranges)
shallow_split_cat_cmt3d = cutils.split_cat_mech_depth(cmt3d_cat, ranges=ranges)
shallow_split_cat_cmt3dp = cutils.split_cat_mech_depth(cmt3dp_cat, ranges=ranges)

# %%

cputils.plot_split_gamma_compare(
    shallow_split_gcmt,
    compare_cat=shallow_split_cat_cmt3d,
    label="GCMT",
    compare_label="CMT3D",
)
plt.savefig("gcmt_cmt3d_gamma_compare_shallow.pdf", transparent=True)

cputils.plot_split_gamma_compare(
    shallow_split_gcmt,
    compare_cat=shallow_split_cat_cmt3dp,
    label="GCMT",
    compare_label="CMT3D+",
)
plt.savefig("gcmt_cmt3d+_gamma_compare_shallow.pdf", transparent=True)


cputils.plot_split_gamma_compare(
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
