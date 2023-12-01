# %%
# Import things
import os
import obspy
import cmt3d
import cmt3d.ioi
import obsplotlib.plot as opl
import obswinlib as owl
import matplotlib.pyplot as plt
from cmt3d.viz.filter_windows import make_plot_windows

# %%
# Load things
outdir = "/gpfs/alpine/geo111/scratch/lsawade/gcmt/nnodes/B010800D"
wave = 'body'
component = 'Z'
obs = cmt3d.ioi.read_data(outdir, wave)
syn = cmt3d.ioi.read_synt(outdir, wave, 0, 0)
cmt = cmt3d.ioi.get_cmt_all(outdir)[0]
xml_event = obspy.read_events(os.path.join(outdir, 'meta', 'init_model.cmt'))
stations = cmt3d.read_inventory(os.path.join(outdir, 'meta', 'stations.xml'))

# %%

# Window stuff
def make_window_dict(outdir, process_file, wave):
    stations = cmt3d.read_inventory(os.path.join(outdir, 'meta', 'stations.xml'))
    xml_event = obspy.read_events(os.path.join(outdir, 'meta', 'init_model.cmt'))
    config = cmt3d.read_yaml(process_file)
    config_dict = config[wave]['window'][0]
    return {'station': stations, 'event': xml_event,
            'config_dict': config_dict, '_verbose': True}

process_file = "/ccs/home/lsawade/gcmt/cmt3d/src/cmt3d/ioi/process_testing.yml"
wd = make_window_dict(outdir, process_file, wave)

#%%
# Add parameters
data = obs.copy().select(component=component)

# %%
if True:
    pdata = data.copy()
    psynt = syn.copy()
    opl.attach_geometry(pdata, cmt.latitude, cmt.longitude, stations)
    opl.copy_geometry(pdata, psynt)
    pdata, psynt = opl.select_intersection([pdata, psynt], components=component)

    scale = 20.0
    size = (10, 8)

    # Plot section
    plt.figure(figsize=size)
    ax, _ = opl.section([pdata,psynt], labels=["Obs", "Syn"],
                        origin_time=cmt.origin_time, lw=0.25,
                        comp=component,
                        plot_amplitudes=False,
                        plot_geometry=True,
                        scale=scale,
                        legendargs=dict(loc='lower right', ncol=4,
                                    frameon=False,
                                    bbox_to_anchor=(1, 1)),
                    )
    ax.tick_params(axis='y', labelsize='xx-small')

    plt.savefig("per_window_testsection.png", dpi=300)

# %%


data = owl.window_on_stream(data, syn, **wd)

# After each trace has windows attached continue
owl.add_tapers(data, taper_type="tukey",
                alpha=0.25, verbose=False)


# Plot stuff
opl.attach_geometry(data, cmt.latitude, cmt.longitude, stations)
opl.copy_geometry(data, syn)

# Make plotting windows
pdata, psynt = make_plot_windows(data, [syn,], ["syn"])


# Get intersection
pstreams = [pdata,*psynt]

scale = 1.0
size = (10, 8)

# Plot section
plt.figure(figsize=size)
ax, _ = opl.section(pstreams, labels=["Obs", "Syn"],
                    origin_time=cmt.origin_time, lw=0.25,
                    comp=component,
                    plot_amplitudes=False,
                    plot_geometry=True,
                    scale=scale,
                    legendargs=dict(loc='lower right', ncol=4,
                                frameon=False,
                                bbox_to_anchor=(1, 1)),
                window=True)
ax.tick_params(axis='y', labelsize='xx-small')

plt.savefig("testsection.png", dpi=300)
