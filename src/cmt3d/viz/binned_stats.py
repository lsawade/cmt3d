from typing import Union
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import numpy as np

plotdict = dict(
    blines=dict(lw=1.0, color='k'),
    mean=dict(ls='', marker='o', c='k', markersize=2.0),
    std=dict(ls='', markersize=4, c='k', marker='|')
)


def plot_binnedstats(
        x, values, bins=10, range=None,
        plotdict: dict = plotdict, orientation: str = 'horizontal',
        quantile: Union[float, list] = 0.25,
        quantilemarkers: list = None, log: bool = False):
    """Plots stats for each bin. Available stats are: ``mean``, ``std``,
    ``median``, ``quantile``, [``blines``]. ``blines`` is not technically a
    statistic, it plots bars of the width of the bins in form of bars from
    at the ``median`` of each bar if wanted or then ``mean``.

    Parameters
    ----------
    x : arraylike
        data location
    values : arraylike
        data, same size as daa
    bins : any
        define bins either by providing an integers or bin edges,
        by default 10
    range : arraylike, optional
        range of the bins, by default None
    plotdict : dict, optional
        plotting dictionary, the most important part,
        by default

        .. code:: python

            plotdict = dict(
                blines=dict(lw=1.0, color='k'),
                mean=dict(ls='', marker='o', c='k', markersize=2.0),
                std=dict(ls='', markersize=4, c='k', marker='|')
            )

    orientation : str, optional
        'vertical' or 'horizontal', by default 'horizontal'
    quantile : Union[float, list], optional
        plot a single or multiple quantiles, by default 0.25
    quantilemarkers : list, optional
        if provided must match the number of quantiles, by default None
    log : bool, optional
        difference in computing the bin centers.


    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.10.13 23.30

    """

    if ('mean' in plotdict) or ('std' in plotdict):

        # Create binned statistic for mean
        mn = binned_statistic(
            x, values, bins=bins, statistic='mean', range=range)

        # Get bin centers
        if log:
            logbins = np.log10(mn.bin_edges)
            binc = logbins[:-1] + .5 * np.diff(logbins)
            binc = 10**binc
        else:
            binc = mn.bin_edges[:-1] + .5 * np.diff(mn.bin_edges)

        if orientation == "horizontal":
            plt.plot(binc, mn.statistic, **plotdict['mean'])
        elif orientation == "vertical":
            plt.plot(mn.statistic, binc, **plotdict['mean'])
        else:
            raise ValueError(f'{orientation} not implemented.')

    if 'median' in plotdict:

        # Create binned statistic for mean
        md = binned_statistic(
            x, values, bins=bins, statistic='median', range=range)

        binc = md.bin_edges[:-1] + .5 * np.diff(md.bin_edges)

        if orientation == "horizontal":
            plt.plot(binc, md.statistic, **plotdict['median'])
        elif orientation == "vertical":
            plt.plot(md.statistic, binc, **plotdict['median'])
        else:
            raise ValueError(f'{orientation} not implemented.')

    if 'std' in plotdict:

        # Create binned statistic for mean
        st = binned_statistic(
            x, values, bins=bins, statistic='std', range=range)

        # Get bin centers
        binc = st.bin_edges[:-1] + .5 * np.diff(st.bin_edges)

        if orientation == "horizontal":
            plt.plot(binc, mn.statistic + st.statistic, **plotdict['std'])
            plt.plot(binc, mn.statistic - st.statistic, **plotdict['std'])
        elif orientation == "vertical":
            plt.plot(mn.statistic + st.statistic, binc, **plotdict['std'])
            plt.plot(mn.statistic - st.statistic, binc, **plotdict['std'])
        else:
            raise ValueError(f'{orientation} not implemented.')

    if 'quantile' in plotdict:

        if isinstance(quantile, float):
            quantile = [quantile]

        for _i, _quant in enumerate(quantile):

            # Create binned statistic for mean
            qt = binned_statistic(
                x, values, bins=bins,
                statistic=lambda x: np.quantile(x, _quant), range=range)

            # Get bin centers
            binc = qt.bin_edges[:-1] + .5 * np.diff(qt.bin_edges)

            # Check if custom marker for quantiles
            qdict = plotdict['quantile']
            if quantilemarkers:
                qdict['marker'] = quantilemarkers[_i]

            if orientation == "horizontal":
                plt.plot(binc, qt.statistic, **qdict)
            elif orientation == "vertical":
                plt.plot(qt.statistic, binc, **qdict)
            else:
                raise ValueError(f'{orientation} not implemented.')

    # Plot bin lines
    if 'blines' in plotdict:

        if 'median' in plotdict:
            mn = md
        else:
            assert 'mean' in plotdict

        if orientation == "horizontal":
            plt.hlines(
                mn.statistic,
                mn.bin_edges[:-1], mn.bin_edges[1:],
                **plotdict['blines'])

        elif orientation == "vertical":
            plt.vlines(
                mn.statistic,
                mn.bin_edges[:-1], mn.bin_edges[1:],
                **plotdict['blines'])

        else:
            raise ValueError(f'{orientation} not implemented.')