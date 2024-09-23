import os
import glob
import typing as tp
from typing import Callable, Optional, Union, List
import numpy as np
import matplotlib
import matplotlib.ft2font as ft
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from numpy import arctan2, sin, cos, degrees, radians
import cartopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, Colormap
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
import matplotlib.font_manager as fm
import matplotlib.ft2font as ft
from cartopy.crs import PlateCarree, Mollweide, UTM
import matplotlib.pyplot as plt


FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def pick_colors_from_cmap(N: int, cmap: str = "viridis") -> List[tuple]:
    """Picks N uniformly distributed colors from a given colormap.

    Parameters
    ----------
    N : int
        Number of wanted colors
    cmap : str, optional
        name of the colormap to pick from, by default 'viridis'


    Returns
    -------
    List[tuple]
        List of color tuples.


    See Also
    --------
    lwsspy.plot.update_colorcycler.update_colorcycler : Updates the colors
        used in new lines/scatter points etc.

    """

    # Get cmap
    colormap = plt.get_cmap(cmap)

    # Pick
    colors = colormap(np.linspace(0, 1, N))

    return colors


def axes_from_axes(
    ax: Axes, n: int, extent: tp.Iterable = [0.2, 0.2, 0.6, 1.0], **kwargs
) -> Axes:
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


def plot_label(
    ax: Axes,
    label: str,
    aspect: float = 1,
    location: int = 1,
    dist: float = 0.025,
    box: tp.Union[bool, dict] = True,
    fontdict: dict = {},
    **kwargs,
):
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
            boxdict = {"facecolor": "w", "edgecolor": "k"}
        else:
            boxdict = {"facecolor": "none", "edgecolor": "none"}
    else:
        boxdict = box

    # Get aspect of the axes
    aspect = 1.0 / get_aspect(ax)

    # Inside
    if location == 0:
        ax.text(
            0.5,
            0.5,
            label,
            horizontalalignment="center",
            verticalalignment="center_baseline",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 1:
        ax.text(
            dist,
            1.0 - dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 2:
        ax.text(
            1.0 - dist,
            1.0 - dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 3:
        ax.text(
            dist,
            dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 4:
        ax.text(
            1.0 - dist,
            dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    # Outside
    elif location == 5:
        ax.text(
            -dist,
            1.0,
            label,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 6:
        ax.text(
            0,
            1.0 + dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 7:
        ax.text(
            1.0,
            1.0 + dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 8:
        ax.text(
            1.0 + dist,
            1.0,
            label,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 9:
        ax.text(
            1.0 + dist,
            0.0,
            label,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 10:
        ax.text(
            1.0,
            -dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 11:
        ax.text(
            0.0,
            -dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 12:
        ax.text(
            -dist,
            0.0,
            label,
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 13:
        ax.text(
            -dist,
            0.5,
            label,
            horizontalalignment="right",
            verticalalignment="center_baseline",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 14:
        ax.text(
            0.5,
            1.0 + dist * aspect,
            label,
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 15:
        ax.text(
            1 + dist,
            0.5,
            label,
            horizontalalignment="left",
            verticalalignment="center_baseline",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 16:
        ax.text(
            0.5,
            -dist * aspect,
            label,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 17:
        ax.text(
            -dist,
            1.0 + dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 18:
        ax.text(
            1.0 + dist,
            1.0 + dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 19:
        ax.text(
            1.0 + dist,
            0.0 - dist * aspect,
            label,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 20:
        ax.text(
            0.0 - dist,
            0.0 - dist * aspect,
            label,
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 21:
        ax.text(
            0.0 + dist,
            0.5,
            label,
            horizontalalignment="left",
            verticalalignment="center_baseline",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 22:
        ax.text(
            0.5,
            1.0 - dist * aspect,
            label,
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 23:
        ax.text(
            1.0 - dist,
            0.5,
            label,
            horizontalalignment="right",
            verticalalignment="center_baseline",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    elif location == 24:
        ax.text(
            0.5,
            0.0 + dist * aspect,
            label,
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
            bbox=boxdict,
            fontdict=fontdict,
            **kwargs,
        )
    else:
        raise ValueError("Other corners not defined.")


FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


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
        "font.family": "sans-serif",
        "font.style": "normal",
        "font.variant": "normal",
        "font.weight": "normal",
        "font.stretch": "normal",
        "font.size": 12.0,
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Bitstream Vera Serif",
            "Computer Modern Roman",
            "New Century Schoolbook",
            "Century Schoolbook L",
            "Utopia",
            "ITC Bookman",
            "Bookman",
            "Nimbus Roman No9 L",
            "Times",
            "Palatino",
            "Charter",
            "serif",
        ],
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Bitstream Vera Sans",
            "Computer Modern Sans Serif",
            "Lucida Grande",
            "Verdana",
            "Geneva",
            "Lucid",
            "Avant Garde",
            "sans-serif",
        ],
        "font.cursive": [
            "Apple Chancery",
            "Textile",
            "Zapf Chancery",
            "Sand",
            "Script MT",
            "Felipa",
            "Comic Neue",
            "Comic Sans MS",
            "cursive",
        ],
        "font.fantasy": [
            "Chicago",
            "Charcoal",
            "Impact",
            "Western",
            "Humor Sans",
            "xkcd",
            "fantasy",
        ],
        "font.monospace": [
            "Roboto Mono",
            "Monaco",
            "DejaVu Sans Mono",
            "Bitstream Vera Sans Mono",
            "Computer Modern Typewriter",
            "Andale Mono",
            "Nimbus Mono L",
            "Courier New",
            "Courier",
            "Fixed",
            "Terminal",
            "monospace",
        ],
        "font.size": 12,
        # 'pdf.fonttype': 3,
        "figure.dpi": 140,
        "font.weight": "normal",
        # 'pdf.fonttype': 42,
        # 'ps.fonttype': 42,
        # 'ps.useafm': True,
        # 'pdf.use14corefonts': True,
        "axes.unicode_minus": False,
        "axes.labelweight": "normal",
        "axes.labelsize": "small",
        "axes.titlesize": "medium",
        "axes.linewidth": 1,
        "axes.grid": False,
        "grid.color": "k",
        "grid.linestyle": ":",
        "grid.alpha": 0.7,
        "xtick.labelsize": "small",
        "xtick.direction": "out",
        "xtick.top": True,  # draw label on the top
        "xtick.bottom": True,  # draw label on the bottom
        "xtick.minor.visible": True,
        "xtick.major.top": True,  # draw x axis top major ticks
        "xtick.major.bottom": True,  # draw x axis bottom major ticks
        "xtick.major.size": 4,  # draw x axis top major ticks
        "xtick.major.width": 1,  # draw x axis top major ticks
        "xtick.minor.top": True,  # draw x axis top minor ticks
        "xtick.minor.bottom": True,  # draw x axis bottom minor ticks
        "xtick.minor.width": 1,  # draw x axis top major ticks
        "xtick.minor.size": 2,  # draw x axis top major ticks
        "ytick.labelsize": "small",
        "ytick.direction": "out",
        "ytick.left": True,  # draw label on the top
        "ytick.right": True,  # draw label on the bottom
        "ytick.minor.visible": True,
        "ytick.major.left": True,  # draw x axis top major ticks
        "ytick.major.right": True,  # draw x axis bottom major ticks
        "ytick.major.size": 4,  # draw x axis top major ticks
        "ytick.major.width": 1,  # draw x axis top major ticks
        "ytick.minor.left": True,  # draw x axis top minor ticks
        "ytick.minor.right": True,  # draw x axis bottom minor ticks
        "ytick.minor.size": 2,  # draw x axis top major ticks
        "ytick.minor.width": 1,  # draw x axis top major ticks
        "legend.fancybox": False,
        "legend.frameon": True,
        "legend.loc": "best",
        "legend.numpoints": 1,
        "legend.fontsize": "small",
        "legend.framealpha": 1,
        "legend.scatterpoints": 3,
        "legend.edgecolor": "inherit",
        "legend.facecolor": "w",
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "cursive",
        "mathtext.tt": "monospace",
        "mathtext.default": "it",
    }

    matplotlib.rcParams.update(params)


def add_fonts(verbose: bool = False):
    # Remove fontlist:
    for file in glob.glob("~/.matplotlib/font*.json"):
        os.remove(file)

    # Fonts
    fontfiles = glob.glob(os.path.join(FONTS, "*.tt?"))

    # for name, fname in fontdict.items():
    for fname in fontfiles:
        font = ft.FT2Font(fname)

        # Just to verify what kind of fonts are added verifiably
        if verbose:
            print(fname, "Scalable:", font.scalable)
            for style in (
                "Italic",
                "Bold",
                "Scalable",
                "Fixed sizes",
                "Fixed width",
                "SFNT",
                "Horizontal",
                "Vertical",
                "Kerning",
                "Fast glyphs",
                "Multiple masters",
                "Glyph names",
                "External stream",
            ):
                bitpos = getattr(ft, style.replace(" ", "_").upper()) - 1
                print(f"{style+':':17}", bool(font.style_flags & (1 << bitpos)))

        # Actually adding the fonts
        fe = fm.ttfFontProperty(font)
        fm.fontManager.ttflist.insert(0, fe)


def map_axes(
    proj: str = "moll",
    central_longitude=0.0,
    zone: int = None,
    southern_hemisphere: bool = False,
) -> plt.Axes:
    """Creates matplotlib axes with map projection taken from cartopy.

    Parameters
    ----------
    proj: str, optional
        shortname for mapprojection
        'moll', 'carr', 'utm', by default "moll"
    central_longitude: float, optional
        What the name suggests default 0.0
    zone: int, optional
        if proj is 'utm', this value must be specified and refers to the UTM
        projection zone
    southern_hemisphere: bool, optional
        if proj is UTM please specify whether you want the southern or Northern
        Hemisphere by setting this flag. Default is False, which sets the option
        to Northern Hemisphere.


    Returns
    -------
    plt.Axes
        Matplotlib axes with projection

    Raises
    ------
    ValueError
        If non supported shortname for axes is given

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.01.13 20.30

    Examples
    --------

    >>> from lwsspy.plot import map_axes
    >>> map_axes()

    """

    # Check whether name is supported.
    if proj not in ["moll", "carr", "utm"]:
        raise ValueError(
            f"Either 'moll' for mollweide, "
            f"'carr' for PlateCarree or 'utm' for UTM.\n'{proj}'"
            f"is not supported."
        )

    if proj == "moll":
        projection = Mollweide(central_longitude=central_longitude)
    elif proj == "carr":
        projection = PlateCarree(central_longitude=central_longitude)
    elif proj == "utm":
        projection = UTM(zone, southern_hemisphere=southern_hemisphere)

    ax = plt.axes(projection=projection)

    return ax


def plot_map(
    fill=True,
    zorder=None,
    labelstopright: bool = True,
    labelsbottomleft: bool = True,
    borders: bool = False,
    rivers: bool = False,
    lakes: bool = False,
    outline: bool = False,
    oceanbg=None,
    ax=None,
    lw=0.5,
):
    """Plots map into existing axes.

    Parameters
    ----------
    fill : bool, optional
        fills the continents in light gray, by default True
    zorder : int, optional
        zorder of the map, by default -10
    projection : cartopy.crs.projection, optional
        projection to be used for the map.
    labelstopright : bool, optional
        flag to turn on or off the ticks
    labelsbottomleft : bool, optional
        flag to turn on or off the ticks
    borders : bool
        plot borders. Default True
    rivers : bool
        plot rivers. Default False
    lakes : bool
        plot lakes. Default True
    lw : float
        outline width

    Returns
    -------
    matplotlib.pyplot.Axes
        Axes in which the map was plotted

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.09.22 11.45


    """

    if ax is None:
        ax = plt.gca()

    # Change outline width
    ax.spines["geo"].set_linewidth(lw)

    if outline:
        edgecolor = "black"
    else:
        edgecolor = "none"

    # Add land
    if fill:
        ax.add_feature(
            cartopy.feature.LAND,
            zorder=zorder,
            edgecolor=edgecolor,
            linewidth=0.5,
            facecolor=(0.9, 0.9, 0.9),
        )
    else:
        ax.add_feature(
            cartopy.feature.LAND,
            zorder=zorder,
            edgecolor=edgecolor,
            linewidth=0.5,
            facecolor=(0, 0, 0, 0),
        )

    if oceanbg:
        ax.add_feature(
            cartopy.feature.OCEAN,
            zorder=zorder,
            edgecolor="none",
            linewidth=0.5,
            facecolor=oceanbg,
        )

    if borders:
        ax.add_feature(
            cartopy.feature.BORDERS,
            zorder=None if zorder is None else zorder + 1,
            facecolor="none",
            edgecolor=(0.5, 0.5, 0.5),
            linewidth=0.25,
        )

    if rivers:
        ax.add_feature(
            cartopy.feature.RIVERS,
            zorder=zorder,
            edgecolor=(0.3, 0.3, 0.7),
        )
        #    edgecolor=(0.5, 0.5, 0.7) )

    if lakes:
        ax.add_feature(
            cartopy.feature.LAKES,
            zorder=None if zorder is None else zorder + 1,
            edgecolor="black",
            linewidth=0.5,
            facecolor=(1.0, 1.0, 1.0),
        )
    return ax


def haversine(lon1, lat1, lon2, lat2):
    """Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Taken from here:
    https://stackoverflow.com/a/29546836/13239311



    Parameters
    ----------
    lon1 : array
        longitude 1
    lat1 : array
        latitude 1
    lon2 : array
        longitude 2
    lat2 : array
        latitude 2

    Returns
    -------
    array
        distance in km for a spherical earth with r = 6371 km.
    """

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def bearing(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dL = lon2 - lon1

    X = cos(lat2) * sin(dL)
    Y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dL)

    return degrees(arctan2(X, Y))


def reckon(lat, lon, distance, bearing):
    """Computes new latitude and longitude from bearing and distance.

    Parameters
    ----------
    lat: in degrees
    lon: in degrees
    bearing: in degrees
    distance: in degrees

    Returns
    -------
    lat, lon


    lat1 = math.radians(52.20472)  # Current lat point converted to radians
    lon1 = math.radians(0.14056)  # Current long point converted to radians
    bearing = np.pi/2 # 90 degrees
    # lat2  52.20444 - the lat result I'm hoping for
    # lon2  0.36056 - the long result I'm hoping for.

    """

    # Convert degrees to radians for numpy
    lat1 = lat / 180 * np.pi
    lon1 = lon / 180 * np.pi
    brng = bearing / 180 * np.pi
    d = distance / 180 * np.pi

    # Compute latitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brng))

    # Compute longitude
    lon2 = lon1 + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat1), np.cos(d) - np.sin(lat1) * np.sin(lat2)
    )

    # Convert back
    lat2 = lat2 / np.pi * 180
    lon2 = lon2 / np.pi * 180

    # Correct the longitude lattitude values
    lon2 = np.where(lon2 < -180.0, lon2 + 360.0, lon2)
    lon2 = np.where(lon2 > 180.0, lon2 - 360.0, lon2)

    return lat2, lon2


def scatterlegend(
    values,
    cmap: Optional[Colormap] = None,
    norm: Optional[Normalize] = None,
    sizefunc: Union[Callable, float] = 5,
    handletextpad: float = -2.0,
    fmt: str = "{0:5.2f}",
    lkw=dict(marker="o", markeredgecolor="k", lw=0.2),
    orientation: str = "h",
    yoffset: float = -50,
    *args,
    **kwargs,
) -> Legend:
    """Creates legend of scatter values parsed to function, including a color
    defined by cmap and norm.

    Parameters
    ----------
    values : Iterable
        Values to be put in the legend
    cmap : Optional[Colormap], optional
        Colormap, by default None
    norm : Optional[Normalize], optional
        Norm, by default None
    sizefunc : Union[Callable, float], optional
        Function to define the size of the markers, or float to define size,
        by default 5
    handletextpad: float, optional
        Use to adjust the location of the text underneath the labels. Positive
        values shift the text to the right, default
        -2.0
    fmt : str, optional
        Format specifier, by default '{0:5.2f}'
    lkw : marker dictionary, optional
        dictionary describing the looks of a marker should probably be the same
        as the one parsed to ``scatter``,
        by default dict(marker='o', markeredgecolor="k", lw=0.25)
    orientation : str, optional, ['h', 'v']
        `h` for horizonatal, 'v' for vertical, by default 'h'
    yoffset: float
        offset of loegend text, different for png and pdf outputs, default -50


    Returns
    -------
    Legend
        legend
    """

    # Get handles and labels
    handles, labels = [], []

    # For each value
    for v in values:
        # Get markersize from float or functions
        if isinstance(sizefunc, float):
            ms = sizefunc
        else:
            ms = np.sqrt(sizefunc(np.abs(v)))

        # Create handle
        h = Line2D([0], [0], ls="", color=cmap(norm(v)), ms=ms, **lkw)

        # Save handle and label
        handles.append(h)
        labels.append(fmt.format(v))

    # Check how the legend is to be oriented
    if orientation == "h":
        legend = plt.legend(
            handles,
            labels,
            *args,
            ncol=len(values),
            columnspacing=1.0,
            handletextpad=handletextpad,
            **kwargs,
        )

        # Adjust text height
        for txt, line in zip(legend.get_texts(), legend.get_lines()):
            txt.set_ha("center")  # horizontal alignment of text item)
            txt.set_y(yoffset)

    elif orientation == "v":
        legend = plt.legend(handles[::-1], labels[::-1], *args, ncol=1, **kwargs)

    return legend


def remove_topright(ax=None):
    """Removes top and right border and ticks from input axes."""

    # Get current axis if none given.
    if ax is None:
        ax = plt.gca()

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


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
        "font.family": "sans-serif",
        "font.style": "normal",
        "font.variant": "normal",
        "font.weight": "normal",
        "font.stretch": "normal",
        "font.size": 12.0,
        "font.serif": [
            "Times New Roman",
            "DejaVu Serif",
            "Bitstream Vera Serif",
            "Computer Modern Roman",
            "New Century Schoolbook",
            "Century Schoolbook L",
            "Utopia",
            "ITC Bookman",
            "Bookman",
            "Nimbus Roman No9 L",
            "Times",
            "Palatino",
            "Charter",
            "serif",
        ],
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Bitstream Vera Sans",
            "Computer Modern Sans Serif",
            "Lucida Grande",
            "Verdana",
            "Geneva",
            "Lucid",
            "Avant Garde",
            "sans-serif",
        ],
        "font.cursive": [
            "Apple Chancery",
            "Textile",
            "Zapf Chancery",
            "Sand",
            "Script MT",
            "Felipa",
            "Comic Neue",
            "Comic Sans MS",
            "cursive",
        ],
        "font.fantasy": [
            "Chicago",
            "Charcoal",
            "Impact",
            "Western",
            "Humor Sans",
            "xkcd",
            "fantasy",
        ],
        "font.monospace": [
            "Roboto Mono",
            "Monaco",
            "DejaVu Sans Mono",
            "Bitstream Vera Sans Mono",
            "Computer Modern Typewriter",
            "Andale Mono",
            "Nimbus Mono L",
            "Courier New",
            "Courier",
            "Fixed",
            "Terminal",
            "monospace",
        ],
        "font.size": 12,
        # 'pdf.fonttype': 3,
        "figure.dpi": 140,
        "font.weight": "normal",
        # 'pdf.fonttype': 42,
        # 'ps.fonttype': 42,
        # 'ps.useafm': True,
        # 'pdf.use14corefonts': True,
        "axes.unicode_minus": False,
        "axes.labelweight": "normal",
        "axes.labelsize": "small",
        "axes.titlesize": "medium",
        "axes.linewidth": 1,
        "axes.grid": False,
        "grid.color": "k",
        "grid.linestyle": ":",
        "grid.alpha": 0.7,
        "xtick.labelsize": "small",
        "xtick.direction": "out",
        "xtick.top": True,  # draw label on the top
        "xtick.bottom": True,  # draw label on the bottom
        "xtick.minor.visible": True,
        "xtick.major.top": True,  # draw x axis top major ticks
        "xtick.major.bottom": True,  # draw x axis bottom major ticks
        "xtick.major.size": 4,  # draw x axis top major ticks
        "xtick.major.width": 1,  # draw x axis top major ticks
        "xtick.minor.top": True,  # draw x axis top minor ticks
        "xtick.minor.bottom": True,  # draw x axis bottom minor ticks
        "xtick.minor.width": 1,  # draw x axis top major ticks
        "xtick.minor.size": 2,  # draw x axis top major ticks
        "ytick.labelsize": "small",
        "ytick.direction": "out",
        "ytick.left": True,  # draw label on the top
        "ytick.right": True,  # draw label on the bottom
        "ytick.minor.visible": True,
        "ytick.major.left": True,  # draw x axis top major ticks
        "ytick.major.right": True,  # draw x axis bottom major ticks
        "ytick.major.size": 4,  # draw x axis top major ticks
        "ytick.major.width": 1,  # draw x axis top major ticks
        "ytick.minor.left": True,  # draw x axis top minor ticks
        "ytick.minor.right": True,  # draw x axis bottom minor ticks
        "ytick.minor.size": 2,  # draw x axis top major ticks
        "ytick.minor.width": 1,  # draw x axis top major ticks
        "legend.fancybox": False,
        "legend.frameon": True,
        "legend.loc": "best",
        "legend.numpoints": 1,
        "legend.fontsize": "small",
        "legend.framealpha": 1,
        "legend.scatterpoints": 3,
        "legend.edgecolor": "inherit",
        "legend.facecolor": "w",
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "cursive",
        "mathtext.tt": "monospace",
        "mathtext.default": "it",
    }

    matplotlib.rcParams.update(params)


def add_fonts(verbose: bool = False):
    # Remove fontlist:
    for file in glob.glob("~/.matplotlib/font*.json"):
        os.remove(file)

    # Fonts
    fontfiles = glob.glob(os.path.join(FONTS, "*.tt?"))

    # for name, fname in fontdict.items():
    for fname in fontfiles:
        font = ft.FT2Font(fname)

        # Just to verify what kind of fonts are added verifiably
        if verbose:
            print(fname, "Scalable:", font.scalable)
            for style in (
                "Italic",
                "Bold",
                "Scalable",
                "Fixed sizes",
                "Fixed width",
                "SFNT",
                "Horizontal",
                "Vertical",
                "Kerning",
                "Fast glyphs",
                "Multiple masters",
                "Glyph names",
                "External stream",
            ):
                bitpos = getattr(ft, style.replace(" ", "_").upper()) - 1
                print(f"{style+':':17}", bool(font.style_flags & (1 << bitpos)))

        # Actually adding the fonts
        fe = fm.ttfFontProperty(font)
        fm.fontManager.ttflist.insert(0, fe)


def right_align_legend(legend: matplotlib.legend.Legend):
    """Does as the title suggests. Takes in a legend and right aligns the text.
    Puts markers to the right that is. and right aligns the text.

    Parameters
    ----------
    legend : matplotlib.legend.Legend
        A legend to be fixed.
    """

    vpc = legend._legend_box._children[-1]._children[:]
    for vp in vpc:
        for c in vp._children:
            c._children.reverse()
        vp.align = "right"


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
    """Creates a formatter for multiples of a certain number. Convenient for
    functions of radians e.g.

    Originally taken from `SOF`_.

    .. _SOF:: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib

    Parameters
    ----------
    denominator : int, optional
        Denominator, by default 2
    number : float, optional
        some number, by default np.pi
    latex : str, optional
        Latex number, by default '\pi'

    Returns
    -------
    a formatter for ticklables

    Example
    -------

    >>> x = np.linspace(-np.pi, 3*np.pi,500)
    >>> plt.plot(x, np.cos(x))
    >>> plt.title(r'Multiples of $\pi$')
    >>> ax = plt.gca()
    >>> ax.grid(True)
    >>> ax.set_aspect(1.0)
    >>> ax.axhline(0, color='black', lw=2)
    >>> ax.axvline(0, color='black', lw=2)
    >>> ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    >>> ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    >>> ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    >>> plt.show()


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.04.09 12.50

    """

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        """Creates a formatter and locator for multiples of a certain number.
        Convenient for functions of radians e.g. More sophisticated than the
        formatter itself.

        Originally taken from `SOF`_.

        .. _SOF:: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib

        Parameters
        ----------
        denominator : int, optional
            Denominator, by default 2
        number : float, optional
            some number, by default np.pi
        latex : str, optional
            Latex number, by default '\pi'

        Returns
        -------
        a formatter for ticklables

        Example
        -------

        >>> tau = np.pi*2
        >>> den = 60
        >>> major = Multiple(den, tau, r'\tau')
        >>> minor = Multiple(den*4, tau, r'\tau')
        >>> x = np.linspace(-tau/60, tau*8/60,500)
        >>> plt.plot(x, np.exp(-x)*np.cos(60*x))
        >>> plt.title(r'Multiples of $\tau$')
        >>> ax = plt.gca()
        >>> ax.grid(True)
        >>> ax.axhline(0, color='black', lw=2)
        >>> ax.axvline(0, color='black', lw=2)
        >>> ax.xaxis.set_major_locator(major.locator())
        >>> ax.xaxis.set_minor_locator(minor.locator())
        >>> ax.xaxis.set_major_formatter(major.formatter())
        >>> plt.show()


        Notes
        -----

        :Author:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.04.09 12.50

        """

        self.denominator = denominator
        self.number = number
        self.latex = latex

    @property
    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    @property
    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )
