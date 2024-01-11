#%%
import os
import cmt3d
import obsplotlib.plot as opl
import cmt3d.ioi as ioi
import matplotlib.pyplot as plt
import numpy as np
# %%
outdir = '/lustre/orion/geo111/scratch/lsawade/gcmt/nnodes/B070895A/'
it = 0
ls = 0
im = 0
wave = 'body'

# Read params
stations = cmt3d.read_inventory(os.path.join(outdir, 'meta', 'stations.xml'))
cmt = ioi.get_cmt(outdir, it, ls=ls)
mnames = ioi.read_model_names(outdir)


for wave in ['body', 'mantle', 'surface']:
    for im in range(10):
        print(im, mnames[im])

        # dsdm_raw = ioi.read_dsdm_raw(outdir, im)
        synt = ioi.read_synt(outdir, wave, it, ls)
        dsdm_processed = ioi.read_dsdm(outdir, wave, im, it, ls)

    # % Attach station info
        opl.attach_geometry(synt,
                            event_latitude=cmt.latitude,
                            event_longitude=cmt.longitude,
                            inv=stations)
        opl.copy_geometry(synt, dsdm_processed)

        plotstreams = opl.select_intersection([synt, dsdm_processed])

        # % Plot
        # Make directory
        if not os.path.exists(f"section/{cmt.eventname}/dsdm/{wave}"):
            os.makedirs(f"section/{cmt.eventname}/dsdm/{wave}")

        plt.close('all')
        plt.figure(figsize=(10, 10))
        opl.section(plotstreams,
                    origin_time=cmt.origin_time,
                    labels=['raw', 'processed'])

        plt.savefig(f"section/{cmt.eventname}/dsdm/{wave}/dsdm{im:05d}.pdf", dpi=300)


# %%

np.set_printoptions(linewidth=np.inf)
for it in range(10):
    print("ITERATION:", it)
    try:
        dm = ioi.read_descent(outdir, it)
        print("  DESC:", dm)
    except:
        break
    for ls in range(10):

        try:
            m = ioi.read_model(outdir, it, ls)
            c = ioi.read_cost(outdir, it, ls)
            opt = ioi.read_optvals(outdir, it, ls)
            print("  STEP:", ls)
            print("    MODL:", m)
            print("    OPT: ", opt)
        except:
            break
