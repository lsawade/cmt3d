import os
import glob
import typing as tp
import numpy as np
import matplotlib
import matplotlib.ft2font as ft
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def axes_from_axes(
        ax: Axes, n: int,
        extent: tp.Iterable = [0.2, 0.2, 0.6, 1.0],
        **kwargs) -> Axes:
    """Uses the location of an existing axes to create another axes in relative
    coordinates. IMPORTANT: Unlike ``inset_axes``, this function propagates
    ``*args`` and ``**kwargs`` to the ``pyplot.axes()`` function, which allows
    for the use of the projection ``keyword``.

    Parameters
    ----------
    ax : Axes
        Existing axes
    n : int
        label, necessary, because matplotlib will replace nonunique axes
    extent : list, optional
        new position in axes relative coordinates,
        by default [0.2, 0.2, 0.6, 1.0]


    Returns
    -------
    Axes
        New axes


    Notes
    -----

    DO NOT CHANGE THE INITIAL POSITION, this position works DO NOT CHANGE!

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.07.13 18.30

    """

    # Create new axes DO NOT CHANGE THIS INITIAL POSITION
    newax = plt.axes([0.0, 0.0, 0.25, 0.1], label=str(n), **kwargs)

    # Get new position
    ip = InsetPosition(ax, extent)

    # Set new position
    newax.set_axes_locator(ip)

    # return new axes
    return newax


def get_aspect(ax: Axes) -> float:
    """Returns the aspect ratio of an axes in a figure. This works around the
    problem of matplotlib's ``ax.get_aspect`` returning strings if set to
    'equal' for example

    Parameters
    ----------
    ax : Axes
        Matplotlib Axes object

    Returns
    -------
    float
        aspect ratio

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.20 11.30

    """

    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()

    # Axis size on figure
    _, _, w, h = ax.get_position().bounds

    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)

    return disp_ratio


def plot_label(ax: Axes, label: str, aspect: float = 1,
               location: int = 1, dist: float = 0.025,
               box: tp.Union[bool, dict] = True, fontdict: dict = {},
               **kwargs):
    """Plots label one of the corners of the plot.
    Plot locations are set as follows::

        17  6  14  7  18
            --------
         5 |1  22  2| 8
        13 |21  0 23| 15
        12 |3  24  4| 9
            --------
        20  11 16 10  19

    Tee dist parameter defines the distance between the axes and the text.

    Parameters
    ----------
    label : str
        label
    aspect : float, optional
        aspect ratio length/height, by default 1.0
    location : int, optional
        corner as described by above code figure, by default 1
    aspect : float, optional
        aspect ratio length/height, by default 0.025
    box : bool
        plots bounding box st. the label is on a background, default true
    Notes
    -----
    :Author:
        Lucas Sawade (lsawade@princeton.edu)
    :Last Modified:
        2021.01.26 18.30
    """
    if type(box) is bool:
        if box:
            boxdict = {'facecolor': 'w', 'edgecolor': 'k'}
        else:
            boxdict = {'facecolor': 'none', 'edgecolor': 'none'}
    else:
        boxdict = box

    # Get aspect of the axes
    aspect = 1.0/get_aspect(ax)

    # Inside
    if location == 0:
        ax.text(0.5, 0.5, label,
                horizontalalignment='center',
                verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 1:
        ax.text(dist, 1.0 - dist * aspect, label, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 2:
        ax.text(1.0 - dist, 1.0 - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 3:
        ax.text(dist, dist * aspect, label, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 4:
        ax.text(1.0 - dist, dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    # Outside
    elif location == 5:
        ax.text(-dist, 1.0, label, horizontalalignment='right',
                verticalalignment='top', transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 6:
        ax.text(0, 1.0 + dist * aspect, label, horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 7:
        ax.text(1.0, 1.0 + dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 8:
        ax.text(1.0 + dist, 1.0, label,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 9:
        ax.text(1.0 + dist, 0.0, label,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 10:
        ax.text(1.0, - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 11:
        ax.text(0.0, -dist * aspect, label, horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 12:
        ax.text(-dist, 0.0, label, horizontalalignment='right',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 13:
        ax.text(-dist, 0.5, label, horizontalalignment='right',
                verticalalignment='center_baseline', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 14:
        ax.text(0.5, 1.0 + dist * aspect, label, horizontalalignment='center',
                verticalalignment='bottom', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 15:
        ax.text(1 + dist, 0.5, label, horizontalalignment='left',
                verticalalignment='center_baseline', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 16:
        ax.text(0.5, -dist * aspect, label, horizontalalignment='center',
                verticalalignment='top', transform=ax.transAxes,
                bbox=boxdict, fontdict=fontdict, **kwargs)
    elif location == 17:
        ax.text(- dist, 1.0 + dist * aspect, label,
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 18:
        ax.text(1.0 + dist, 1.0 + dist * aspect, label,
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 19:
        ax.text(1.0 + dist, 0.0 - dist * aspect, label,
                horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 20:
        ax.text(0.0 - dist, 0.0 - dist * aspect, label,
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 21:
        ax.text(0.0 + dist, 0.5, label,
                horizontalalignment='left',
                verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 22:
        ax.text(0.5, 1.0 - dist * aspect, label,
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 23:
        ax.text(1.0 - dist, 0.5, label,
                horizontalalignment='right',
                verticalalignment='center_baseline',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    elif location == 24:
        ax.text(0.5, 0.0 + dist * aspect, label,
                horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes, bbox=boxdict,
                fontdict=fontdict, **kwargs)
    else:
        raise ValueError("Other corners not defined.")


FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')


def updaterc(rebuild=False):
    """Updates the rcParams to something generic that looks ok good out of
    the box.

    Args:

        rebuild (bool):
            Rebuilds fontcache incase it needs it.

    Last modified: Lucas Sawade, 2020.09.15 01.00 (lsawade@princeton.edu)
    """

    add_fonts()

    params = {
        'font.family': 'sans-serif',
        'font.style':   'normal',
        'font.variant': 'normal',
        'font.weight':  'normal',
        'font.stretch': 'normal',
        'font.size':    12.0,
        'font.serif':     [
            'Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif',
            'Computer Modern Roman',
            'New Century Schoolbook', 'Century Schoolbook L', 'Utopia',
            'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L',
            'Times', 'Palatino', 'Charter', 'serif'
        ],
        'font.sans-serif': [
            'Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans',
            'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana',
            'Geneva', 'Lucid', 'Avant Garde', 'sans-serif'
        ],
        'font.cursive':    [
            'Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT',
            'Felipa', 'Comic Neue', 'Comic Sans MS', 'cursive'
        ],
        'font.fantasy':    [
            'Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans', 'xkcd',
            'fantasy'
        ],
        'font.monospace':  [
            'Roboto Mono', 'Monaco', 'DejaVu Sans Mono',
            'Bitstream Vera Sans Mono',  'Computer Modern Typewriter',
            'Andale Mono', 'Nimbus Mono L', 'Courier New', 'Courier', 'Fixed',
            'Terminal', 'monospace'
        ],
        'font.size': 12,
        # 'pdf.fonttype': 3,
        'figure.dpi': 140,
        'font.weight': 'normal',
        # 'pdf.fonttype': 42,
        # 'ps.fonttype': 42,
        # 'ps.useafm': True,
        # 'pdf.use14corefonts': True,
        'axes.unicode_minus': False,
        'axes.labelweight': 'normal',
        'axes.labelsize': 'small',
        'axes.titlesize': 'medium',
        'axes.linewidth': 1,
        'axes.grid': False,
        'grid.color': "k",
        'grid.linestyle': ":",
        'grid.alpha': 0.7,
        'xtick.labelsize': 'small',
        'xtick.direction': 'out',
        'xtick.top': True,  # draw label on the top
        'xtick.bottom': True,  # draw label on the bottom
        'xtick.minor.visible': True,
        'xtick.major.top': True,  # draw x axis top major ticks
        'xtick.major.bottom': True,  # draw x axis bottom major ticks
        'xtick.major.size': 4,  # draw x axis top major ticks
        'xtick.major.width': 1,  # draw x axis top major ticks
        'xtick.minor.top': True,  # draw x axis top minor ticks
        'xtick.minor.bottom': True,  # draw x axis bottom minor ticks
        'xtick.minor.width': 1,  # draw x axis top major ticks
        'xtick.minor.size': 2,  # draw x axis top major ticks
        'ytick.labelsize': 'small',
        'ytick.direction': 'out',
        'ytick.left': True,  # draw label on the top
        'ytick.right': True,  # draw label on the bottom
        'ytick.minor.visible': True,
        'ytick.major.left': True,  # draw x axis top major ticks
        'ytick.major.right': True,  # draw x axis bottom major ticks
        'ytick.major.size': 4,  # draw x axis top major ticks
        'ytick.major.width': 1,  # draw x axis top major ticks
        'ytick.minor.left': True,  # draw x axis top minor ticks
        'ytick.minor.right': True,  # draw x axis bottom minor ticks
        'ytick.minor.size': 2,  # draw x axis top major ticks
        'ytick.minor.width': 1,  # draw x axis top major ticks
        'legend.fancybox': False,
        'legend.frameon': True,
        'legend.loc': 'best',
        'legend.numpoints': 1,
        'legend.fontsize': 'small',
        'legend.framealpha': 1,
        'legend.scatterpoints': 3,
        'legend.edgecolor': 'inherit',
        'legend.facecolor': 'w',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'sans',
        'mathtext.it': 'sans:italic',
        'mathtext.bf': 'sans:bold',
        'mathtext.cal': 'cursive',
        'mathtext.tt':  'monospace',
        'mathtext.default': 'it'
    }

    matplotlib.rcParams.update(params)


def add_fonts(verbose: bool = False):

    # Remove fontlist:
    for file in glob.glob('~/.matplotlib/font*.json'):
        os.remove(file)

    # Fonts
    fontfiles = glob.glob(os.path.join(FONTS, "*.tt?"))

    # for name, fname in fontdict.items():
    for fname in fontfiles:

        font = ft.FT2Font(fname)

        # Just to verify what kind of fonts are added verifiably
        if verbose:
            print(fname, "Scalable:", font.scalable)
            for style in ('Italic',
                          'Bold',
                          'Scalable',
                          'Fixed sizes',
                          'Fixed width',
                          'SFNT',
                          'Horizontal',
                          'Vertical',
                          'Kerning',
                          'Fast glyphs',
                          'Multiple masters',
                          'Glyph names',
                          'External stream'):
                bitpos = getattr(ft, style.replace(' ', '_').upper()) - 1
                print(
                    f"{style+':':17}", bool(font.style_flags & (1 << bitpos)))

        # Actually adding the fonts
        fe = fm.ttfFontProperty(font)
        fm.fontManager.ttflist.insert(0, fe)
