
import os
import obsplotlib.plot as opl
import cmt3d
import cmt3d.ioi as ioi
import matplotlib.pyplot as plt

from .filter_windows import make_plot_windows

def plot_inversion_section(outdir, wtype, windows: bool, component='Z'):
    """
    Plots all steps into a section.
    """

    # Load the data
    data = ioi.read_data_windowed(outdir, wtype)

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(outdir, 'meta',
                                                 'stations.xml'))

    # Load the model
    modls = ioi.read_model_all(outdir)
    cmts = ioi.get_cmt_all(outdir)
    costs = ioi.read_cost_all(outdir)
    grads = ioi.read_gradient_all(outdir)
    hesss = ioi.read_hessian_all(outdir)

    # Get initial CMT
    cmt0 = cmts[0]

    # Read synthetics
    synts = ioi.read_synt_all(outdir, wtype)[0:1]

    # Attach geometry
    opl.attach_geometry(data, cmt0.latitude, cmt0.longitude, stations)
    opl.copy_geometry(data, synts)

    # Filter windows
    synt_labels= [f"C: {_c:.4f}" for _c in costs[0:1]]
    data, synts = make_plot_windows(data, synts, synt_labels)

    # Get intersection
    pstreams = opl.select_intersection([data, *synts], components=component)

    # Plot section
    opl.section(pstreams, labels=["Data", *synt_labels],
                origin_time=cmt0.origin_time,
                comp=component,
                plot_amplitudes=False,
                legendargs=dict(loc='lower right', ncol=4, frameon=False,
                                bbox_to_anchor=(1,1)),
                window=True)

    plt.show()