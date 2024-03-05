# %%
from cmt3d.cmt_catalog import CMTCatalog
from cmt3d.viz.compare_catalogs import CompareCatalogs
from glob import glob
from copy import deepcopy
import numpy as np

# Load the CMT catalog
cat1 = CMTCatalog.from_file_list(glob("events/gcmt3d_fix/*"))
cat2 = deepcopy(cat1)


# %%


def make_catalog_dc(cat: CMTCatalog):
    for cmt in cat:

        # Get eignevalues and eigenvectors
        lmd, ev = cmt.tnp

        # Store scalar moment
        M0 = cmt.M0

        # Init new eigenvalues
        lmd_dc = np.zeros(3)

        # Reorient the moment tensor
        mt = ev @ np.diag(lmd_dc) @ np.linalg.inv(ev)

        # Set moment tensor
        cmt.fulltensor = mt

        # Update M0
        cmt.M0 = M0


make_catalog_dc(cat2)

# %%

cc = CompareCatalogs(cat1, cat2, oldlabel="CMT3D+", newlabel="CMT3D+ (DC)")
cc.plot_summary(outfile="cmt3d_dc.pdf")
