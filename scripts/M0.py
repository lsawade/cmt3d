# %%
# import os
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
# gcmt_cat.save("gcmtcatalog.pkl")
gcmt_cat = CMTCatalog.from_file_list(glob("events/gcmt/*"))
cmt3dp_cat = cmt3d.CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))

# %%

cc = CompareCatalogs(gcmt_cat, cmt3dp_cat)

# %%
cc.plot_spatial_distribution_binned(parameter="M0")
plt.savefig("spatial_M0.pdf")

# %%
