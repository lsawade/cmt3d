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
                               oldlabel='GCMT', newlabel='CMT3D+',
                               nbins=25)
cmt3d_gcmt3df = CompareCatalogs(old=cmt3d___cat, new=gcmt3df_cat,
                                oldlabel='CMT3D', newlabel='CMT3D+',
                                nbins=25)

# %%
if not os.path.exists('plots'):
    os.makedirs('plots')

gcmt_cmt3d.plot_summary(outfile=os.path.join(
    'plots', "gcmt_cmt3d_comparison.pdf"))
gcmt_gcmt3df.plot_summary(outfile=os.path.join(
    'plots', "gcmt_cmt3d+_comparison.pdf"))
cmt3d_gcmt3df.plot_summary(outfile=os.path.join(
    'plots', "cmt3d_cmt3d+_comparison.pdf"))


ranges = dict(
    shallow=(0, 70),
    intermediate=(70, 300),
    deep=(300, 1000),
    all=(0, 700)
    )

for rname, r in ranges.items():
    mindict = dict(depth_in_m=r[0]*1000.0)
    maxdict = dict(depth_in_m=r[1]*1000.0)

    fgcmt____cat, _ = gcmt____cat.filter(maxdict=maxdict, mindict=mindict)

    gcmt_cmt3d = CompareCatalogs(old=fgcmt____cat, new=cmt3d___cat,
                        oldlabel='GCMT', newlabel='CMT3D',
                        nbins=25)
    gcmt_gcmt3df = CompareCatalogs(old=fgcmt____cat, new=gcmt3df_cat,
                                oldlabel='GCMT', newlabel='CMT3D+',
                                nbins=25)
    gcmt_cmt3d.plot_summary(outfile=os.path.join(
        'plots', f"gcmt_cmt3d_comparison_{rname}.pdf"))
    gcmt_gcmt3df.plot_summary(outfile=os.path.join(
        'plots', f"gcmt_cmt3d+_comparison_{rname}.pdf"))
