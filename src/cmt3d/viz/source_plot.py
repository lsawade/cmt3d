import sys
import typing as tp
import numpy as np
from obspy.imaging.beachball import beach as obspy_beach
from matplotlib import transforms
import matplotlib.pyplot as plt
from .utils import axes_from_axes
from .utils import plot_label
from .utils import get_aspect
from .utils import updaterc
from .utils import MidpointNormalize


if tp.TYPE_CHECKING:
    from ..source import CMTSource


def beach(source: CMTSource):

    updaterc()

    plt.figure(figsize=(2, 2))
    ax = plt.axes()

    # Plot beach ball
    bb = obspy_beach(source.tensor,
                     linewidth=2,
                     facecolor='k',
                     bgcolor='w',
                     edgecolor='k',
                     alpha=1.0,
                     xy=(0.5, 0.5),
                     width=200,
                     size=100,
                     nofill=False,
                     zorder=100,
                     axes=ax)
    ax.add_collection(bb)

    # This fixes pdf output issue
    bb.set_transform(transforms.Affine2D(np.identity(3)))

    ax.axis('off')


def axbeach(
        source: CMTSource, ax, x, y, width=50, facecolor='k', linewidth=2,
        alpha=1.0, clip_on=False, **kwargs):
    """Plots beach ball into given axes.
    Note that width heavily depends on the given screen size/dpi. Therefore
    often does not work."""

    # Plot beach ball
    bb = obspy_beach(source.tensor,
                     linewidth=linewidth,
                     facecolor=facecolor,
                     bgcolor='w',
                     edgecolor='k',
                     alpha=alpha,
                     xy=(x, y),
                     width=width,
                     size=100,  # Defines number of interpolation points
                     axes=ax,
                     **kwargs)
    bb.set(clip_on=clip_on)

    # This fixes pdf output issue
    bb.set_transform(transforms.Affine2D(np.identity(3)))

    ax.add_collection(bb)


def beachfig(source: CMTSource):
    """
    SDR
    M0
    location
    origin time
    centroid time shift
    half duration
    3x3 image black 1, white 0
    """
    updaterc()
    plt.figure(figsize=(5.25, 1.75))
    ax = plt.axes()
    ax.axis('off')

    # Plot beach ball
    bb = obspy_beach(source.tensor,
                     linewidth=1,
                     facecolor='k',
                     bgcolor='w',
                     edgecolor='k',
                     alpha=1.0,
                     xy=(0.625, 0.4),
                     width=145,
                     size=100,
                     nofill=False,
                     zorder=100,
                     axes=ax)
    ax.add_collection(bb)

    # This fixes pdf output issue
    bb.set_transform(transforms.Affine2D(np.identity(3)))

    # Base info string
    # title_string = f'{source.eventname}'
    header_topleft = 'PDE:'

    topleft = '\n'
    topleft += 'Magnitudes: '
    topleft += f'Mw {source.moment_magnitude:4.2f}, '
    topleft += f'mb {source.mb:4.2f}, '
    topleft += f'ms {source.ms:4.2f}\n'
    topleft += f'Origin: ' \
        f'{source.origin_time.strftime("%d-%m-%y %H:%M:%S"):>19s}\n'
    topleft += f'Lat, Lon: {"":>1}{source.pde_latitude:>7.2f}, ' \
        f'{source.pde_longitude:>7.2f}\n'
    topleft += f'Depth: {source.pde_depth_in_m/1000:>17.1f} km\n'

    bottomleft = ''
    bottomleft += f'Time Shift: {source.time_shift:>13} s\n'
    bottomleft += f'Lat, Lon: {"":>1}{source.latitude:>7.2f}, ' \
        f'{source.longitude:>7.2f}\n'
    bottomleft += f'Depth: {source.depth_in_m/1000:>17.1f} km\n'
    bottomleft += f'hdur: {source.half_duration:>19} s'

    ss, ds, rs = source.sdr
    bottomright = ''
    bottomright += 'S/D/R:\n'
    bottomright += f'{ss[0]:3.0f}/{ds[0]:3.0f}/{rs[0]:4.0f}\n'
    bottomright += f'{ss[1]:3.0f}/{ds[1]:3.0f}/{rs[1]:4.0f}'

    # Topleft text
    plot_label(ax, header_topleft, location=1, dist=0.0, box=False,
               fontdict=dict(family='monospace', size='x-small',
                             fontweight='bold'))

    plot_label(ax, topleft, location=1, dist=0.0, box=False,
               fontdict=dict(family='monospace', size='x-small'))

    # Bottom left text
    header_bottomleft = 'CMT:' + bottomleft.count('\n') * '\n' + '\n'

    plot_label(ax, header_bottomleft, location=3, dist=0.0, box=False,
               fontdict=dict(family='monospace', size='x-small',
                             fontweight='bold'))
    plot_label(ax, bottomleft, location=3, dist=0.0, box=False,
               fontdict=dict(family='monospace', size='x-small'))

    # Bottom left text
    # header_bottomleft = 'CMT:' + bottomleft.count('\n') * '\n' + '\n'

    # plot_label(ax, header_bottomleft, location=3, dist=0.0, box=False,
    #            fontdict=dict(family='monospace', size='x-small',
    #                          fontweight='bold'))
    plot_label(ax, bottomright, location=4, dist=0.0, box=False,
               fontdict=dict(family='monospace', size='x-small'))

    # Diverging Red to White to Black
    cmapname = 'RdGy'
    cmap = plt.get_cmap(cmapname)  # type: ignore
    norm = MidpointNormalize(vmin=-1.0, midpoint=0.0, vmax=1.0)

    # Make the tensor scaled to between -1 to 1
    absmax = np.max(np.abs(source.tensor))
    mt = source.fulltensor/absmax

    # Get the aspect of the original and the new axes
    fraction = 0.3
    asp = get_aspect(ax)
    subax = axes_from_axes(
        ax, 123,
        extent=[1-fraction*asp, 1-fraction, fraction*asp, fraction])
    im = plt.imshow(mt, cmap=cmap, norm=norm)
    subax.axis('off')

    # Create axes for colorbar
    cfrac = 0.1
    cax = axes_from_axes(
        ax, 123, extent=[
            1 - fraction*asp*(1 + cfrac), 1-fraction,
            fraction*asp*cfrac, fraction])
    plt.colorbar(im, cax=cax)
    cax.axis('off')


def plot_beach():
    cmtsource = CMTSource.from_CMTSOLUTION_file(sys.argv[1])
    beach(cmtsource)
    plt.show(block=True)


def plot_beachfig():
    cmtsource = CMTSource.from_CMTSOLUTION_file(sys.argv[1])
    beachfig(cmtsource)
    plt.show(block=True)
