import typing as tp
import numpy as np
import matplotlib.pyplot as plt

from .plot_quakes import plot_quakes
from .utils import plot_label

if tp.TYPE_CHECKING:
    from cmt3d.cmt_catalog import CMTCatalog


def plot(catalog: CMTCatalog, ax=None, filename: tp.Union[str, None] = None):
    """Plots events on a map"""

    # Get values
    latitude = catalog.getvals("latitude")
    longitude = catalog.getvals("longitude")
    moment = catalog.getvals("moment_magnitude")
    depth = catalog.getvals("depth_in_m")/1000.0
    N = len(depth)

    # Scatter sizefunc
    def sizefunc(x): return np.pi*(0.25*(x-np.min(moment)) /
                                    (np.max(moment)-np.min(moment)) + 1)**8
    # Plot events
    scatter, ax, l1, l2 = plot_quakes(
        latitude, longitude, depth, moment, sizefunc=sizefunc,
        cmap='rainbow_r', ax=ax, yoffsetlegend2=0.02)
    ax.set_global()
    plot_label(ax, f"N: {N}", location=2, box=False, dist=0.0)

    # Save or plot...
    if filename is not None:
        plt.savefig(filename)