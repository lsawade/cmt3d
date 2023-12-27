from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def eigsort(mat: np.ndarray):
    """Returns sorted eigenvalues of a given matrix

    Args:
        mat (np.array): Square matrix

    Returns:
        sorted eigenvalue [vector], and -vectors [matrix]

    Last modified: Lucas Sawade, 2020.09.14 23.00 (lsawade@princeton.edu)
    """

    # Get Eigenvalues
    vals, vecs = np.linalg.eigh(mat)

    # Sort them
    order = vals.argsort()[::-1]

    # Return sorted values and vectors

    return vals[order], vecs[:, order]


def errorellipse(x, y, nstd: int = 2,
                 ax: Union[None, plt.Axes] = None,
                 **kwargs):
    """Plots error ellipse

    Args:
        x (np.ndarray or list):
            N element list
        y (np.ndarray or list):
            N element list
        kwargs:
            parsed to matplotlibs Ellipse function

    Returns:
        ellipse handle

    Last modified: Lucas Sawade, 2020.09.15 15.30 (lsawade@princeton.edu)

    """

    # Get covariance and eigenvectors
    cov = np.cov(x, y)
    vals, vecs = eigsort(cov)

    # Get angle from the vectors
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Get width and height of the ellipse
    w, h = 2 * nstd * np.sqrt(vals)

    # Create elliptical patch
    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
                  width=w, height=h,
                  angle=theta, **kwargs)

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Add patch
    ax.add_patch(ell)

    # return ellipse's handle
    return ell



def get_color(x, cmap="seismic", vmin=0, vmax=100, norm=None):
    n = vmax - vmin
    val = (np.clip(x, vmin, vmax) - vmin) / n

    if type(cmap) not in [ListedColormap, LinearSegmentedColormap]:
        cmap = getattr(cm, cmap)

    if norm is not None:
        val = norm(val)

    return cmap(val)


def scatter_hist(x, y, nbins, z=None, cmap=None,
                 histc=((0.35, 0.35, 0.35),), label=None,
                 zmin=None, zmax=None, norm=None, r=True,
                 xlog=False, ylog=False, ellipses=True,
                 fraction=0.75, mult: bool = False):
    """Function to creat cross plots

    Parameters
    ----------
    x : Arraylike
        Data type x-axis
    y : Arraylike
        Data type y-axis
    nbins : [type]
        [description]
    z : Arraylike, optional
        Optional Z Values values, by default None
    cmap : str, optional
        cmap to color Z values, by default None
    histc : tuple, optional
        Histogramcolors, by default ((0.35, 0.35, 0.35),)
    zmin : float, optional
        minimum z value, by default None
    zmax : float, optional
        maximum z value, by default None
    norm : norm, optional
        norm for coloring the z values, by default None
    r : bool, optional
        Whether R value should be plotted, by default True
    xlog : bool, optional
        Scaling of the x axis, by default False
    ylog : bool, optional
        Scaling of the x axis, by default False
    ellipses : bool, optional
        plot confidence ellipses, by default True
    fraction : float, optional
        how much of the original axes the scatter axis should take up in
        either dimension. That is, the histograms will have a height of
        (1-fraction) * width or height of the original axes.
    mult : bool, optional
        if mult is True x and y are multiple datasets inform of lists.

    Returns
    -------
    [type]
        [description]
    """

    # definitions for the axes
    ax_scatter = plt.gca()
    ax_pos = ax_scatter.get_position()
    scatter_pos = [ax_pos.x0, ax_pos.y0,
                   ax_pos.width * fraction, ax_pos.height * fraction]
    histx_pos = [ax_pos.x0, ax_pos.y0 + ax_pos.height * fraction,
                 ax_pos.width * fraction, ax_pos.height * (1-fraction)]
    histy_pos = [ax_pos.x0 + ax_pos.width * fraction, ax_pos.y0,
                 ax_pos.width * (1-fraction), ax_pos.height * fraction]

    # Create Axes
    ax_scatter.set_position(scatter_pos)
    # ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(histx_pos, sharex=ax_scatter, zorder=-1)
    # ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(histy_pos, sharey=ax_scatter, zorder=-1)

    # scatterplot with color
    # if cmap is not None and z is not None:

    #     cax = ax_scatter.inset_axes([0.05, 0.96, 0.25, 0.03],
    #                                 zorder=10000)

    #     zpos = np.argsort(z)
    #     # the scatter plot:
    #     if zmin is not None:
    #         vminz = zmin
    #     else:
    #         vminz = np.min(z)

    #     if zmax is not None:
    #         vmaxz = zmax
    #     else:
    #         vmaxz = np.max(z)

    #     if norm is not None and cmap is not None:
    #         ax_scatter.scatter(x[zpos], y[zpos], c=cmap(norm(z[zpos])),
    #                            s=20, marker='o', edgecolor='none',
    #                            linewidths=0.5, alpha=0.25)

    #         # Colorbar
    #         cbar_dict = {"orientation": "horizontal"}
    #         plt.colorbar(matplotlib.cm.ScalarMappable(
    #             cmap=cmap, norm=norm),
    #             cax=cax, **cbar_dict)
    #         cax.tick_params(left=False, right=False, bottom=True, top=True,
    #                         labelleft=False, labelright=False,
    #                         labeltop=False,
    #                         labelbottom=True, which='both',
    #                         labelsize=6)

    #     else:
    #         ax_scatter.scatter(
    #             x[zpos], y[zpos],
    #             c=get_color(z[zpos], vmin=vminz, vmax=vmaxz,
    #                         cmap=cmap, norm=norm),
    #             s=20, marker='o', edgecolor='none', linewidths=0.5,
    #             alpha=0.25)

    #         # Colorbar
    #         cbar_dict = {"orientation": "horizontal"}
    #         create_colorbar(vminz, vmaxz, cmap=cmap, cax=cax, **cbar_dict)
    #         cax.tick_params(left=False, right=False, bottom=True, top=True,
    #                         labelleft=False, labelright=False,
    #                         labeltop=False, labelbottom=True, which='both',
    #                         labelsize=6)

    # scatterplot without color
    # else:
    #   cax=None
    if mult:
        xl = x
        yl = y
        cl = histc
        if label is None:
            ll = len(xl) * [None]
        else:
            ll = label
    else:
        xl = [x]
        yl = [y]
        cl = [histc]
        ll = [label]
    # Init lists of mins and maxs
    minxl, maxxl, minyl, maxyl = [], [], [], []

    for x, y, c, l in zip(xl, yl, cl, ll):
        minxl.append(np.min(x))
        maxxl.append(np.max(x))
        minyl.append(np.min(y))
        maxyl.append(np.max(y))
        ax_scatter.scatter(x, y, c=(c,), s=15, alpha=0.75,
                           edgecolor='none', label=l)

    # now determine nice limits by hand:
    minx = np.min(minxl)
    maxx = np.max(maxxl)
    miny = np.min(minyl)
    maxy = np.max(maxyl)
    dx = maxx - minx
    dy = maxy - miny

    # Histogram settings
    if xlog:
        if minx < 1.0:
            minx = 1.0
        binsx = np.logspace(np.log10(minx), np.log10(maxx), nbins + 1)
    else:
        binsx = np.linspace(minx, maxx, nbins + 1)

    if ylog:
        if miny < 1.0:
            miny = 1
        binsy = np.logspace(np.log10(miny), np.log10(maxy), nbins + 1)
    else:
        binsy = np.linspace(miny, maxy, nbins + 1)

    # Set limits with buffer
    prec = 0.05
    ax_scatter.set_xlim((minx - prec * dx, maxx + prec * dx))
    ax_scatter.set_ylim((miny - prec * dy, maxy + prec * dy))

    for _i, (x, y, c, l) in enumerate(zip(xl, yl, cl, ll)):
        # Plot y histogram
        ax_histx.hist(x, bins=binsx, facecolor=c,
                      edgecolor='none', zorder=2*len(xl) - _i - 0.5,
                      alpha=0.75)
        ax_histx.hist(x, bins=binsx, color='k', histtype='step',
                      facecolor='none', zorder=2*len(xl) - _i)
        ax_histx.set_xlim(ax_scatter.get_xlim())

        # Plot y histogram
        ax_histy.hist(y, bins=binsy, orientation='horizontal',
                      facecolor=c, edgecolor='none', zorder=2*len(xl) - _i - 0.5,
                      alpha=0.75)
        ax_histy.hist(y, bins=binsy, orientation='horizontal',
                      color='k', histtype='step', facecolor='none',
                      zorder=2*len(xl) - _i)
        ax_histy.set_ylim(ax_scatter.get_ylim())

    # Remove boundaries
    ax_histx.axis('off')
    ax_histy.axis('off')

    if r and mult is False:
        if xlog:
            xfix = np.log10(x)
        else:
            xfix = x
        if ylog:
            yfix = np.log10(y)
        else:
            yfix = y
        # Write out correlation coefficient in the top right
        corr_coeff = np.corrcoef(xfix, yfix)
        text_dict = {
            "fontfamily": "monospace", "fontsize": "x-small",
            "verticalalignment": 'top',
            "zorder": 100}
        ax_scatter.text(0.975, 0.975,
                        "$R$ = %5.2f\n"
                        "$\\mu_x$ = %5.2f\n"
                        "$\\mu_y$ = %5.2f" % (
                            corr_coeff[0, 1], np.mean(x), np.mean(y)),
                        horizontalalignment='right', **text_dict,
                        transform=ax_scatter.transAxes)

    if ellipses and mult is False:
        ls = ["-", "--", ":"]
        for _i in np.arange(1, 4):
            errorellipse(
                x, y, ax=ax_scatter, nstd=_i, label=r'$%d\sigma$' % _i,
                edgecolor='k', linestyle=ls[_i-1], facecolor='none')

    if mult is True:
        ax_scatter.legend(
            loc='best', ncol=1, fontsize="x-small",
            frameon=True, fancybox=False,
            numpoints=1, scatterpoints=1,
            borderaxespad=0.01, borderpad=0.5, handletextpad=0.2,
            labelspacing=0.2, handlelength=1.0)

    # make the scatter axes the current axes
    plt.sca(ax_scatter)

    return ax_scatter, ax_histx, ax_histy