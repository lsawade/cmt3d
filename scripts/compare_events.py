# %%

import os
from glob import glob
import cmt3d
import cmt3d.ioi as ioi
import cmt3d.viz as viz
from cmt3d.viz.compare_catalogs import CompareCatalogs

# %%
# Make the catalogs to compare the events
gcmt____files = glob(os.path.join('events', 'gcmt', '*'))
cmt3d___files = glob(os.path.join('events', 'cmt3d', '*'))
gcmt3d__files = glob(os.path.join('events', 'gcmt3d', '*'))
gcmt3df_files = glob(os.path.join('events', 'gcmt3d_fix', '*'))

# Make catalog
gcmt____cat = cmt3d.CMTCatalog.from_file_list(gcmt____files)
cmt3d___cat = cmt3d.CMTCatalog.from_file_list(cmt3d___files)
gcmt3d__cat = cmt3d.CMTCatalog.from_file_list(gcmt3d__files)
gcmt3df_cat = cmt3d.CMTCatalog.from_file_list(gcmt3df_files)


# %%

gcmt_cmt3d = CompareCatalogs(old=gcmt____cat, new=cmt3d___cat,
                         oldlabel='GCMT', newlabel='CMT3D',
                         nbins=25)
gcmt_gcmt3df = CompareCatalogs(old=gcmt____cat, new=gcmt3df_cat,
                         oldlabel='GCMT', newlabel='GCMT3D',
                         nbins=25)

# %%
if not os.path.exists('plots'):
    os.makedirs('plots')

gcmt_cmt3d.plot_summary(outfile=os.path.join(
        'plots', "gcmt_cmt3d_comparison.pdf"))
gcmt_gcmt3df.plot_summary(outfile=os.path.join(
        'plots', "gcmt_gcmt3d_comparison.pdf"))