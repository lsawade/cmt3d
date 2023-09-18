import obspy
import typing as tp
import cmt3d
import obsproclib as opr
import numpy as np
import obsplotlib.plot as opl

def make_plot_windows(data: obspy.Stream,
                      synts: obspy.Stream | tp.List[obspy.Stream],
                      labels: list[str]):

    if len(synts) != len(labels):
        raise ValueError("Number of labels must match number of synthetics")

    new_data = obspy.Stream()

    # Add only traces that have windows
    for tr in data:
        if len(tr.stats.windows) > 0:
            new_data += (tr.copy())

    # Select intersection
    pstreams = opl.select_intersection([new_data, *synts], components='ZRT')

    # Divide data and synthetics
    new_data = pstreams[0]
    new_synts = pstreams[1:]

    for tr in new_data:
        windows = tr.stats.windows

        pwindows = []
        for _win in windows:
            pwin = opl.Window(tr, startidx=_win.left, endidx=_win.right)
            pwindows.append(pwin)

        tr.stats.windows = pwindows

        for _synt, _label in zip(new_synts, labels):

            _syntr = _synt.select(network=tr.stats.network,
                                  station=tr.stats.station,
                                  component=tr.stats.component)[0]

            _syntr.stats.windows = pwindows
            _syntr.stats.label = _label

            for _pwin, _tap in zip(tr.stats.windows, tr.stats.tapers):

                # Get the data and synthetic traces
                delta = tr.stats.delta
                obs = tr.data[_pwin.startidx:_pwin.endidx] * _tap
                syn = _syntr.data[_pwin.startidx:_pwin.endidx] * _tap

                # Compute cross-correlation timeshift
                cc_max, ishift = cmt3d.xcorr(obs, syn)
                tshift = ishift * delta

                # Make second synthetic trace with fixed cross-correlation timeshift
                oleft, oright, sleft, sright = cmt3d.correct_window_index(
                    _pwin.startidx, _pwin.endidx, ishift, _syntr.data.size)

                obss = tr.data[oleft:oright]
                syns = _syntr.data[sleft:sright]

                # Compute the correlation ratio
                cc_ratio = np.sum(obss * syns)/np.sum(syns ** 2)

                # L2
                l2_power = np.sum(((obs - syn)) ** 2) / \
                    np.sum((obs) ** 2)

                if not hasattr(_pwin, 'measurements'):
                    _pwin.measurements = dict()

                _pwin.measurements[_label] = dict(
                    DT=tshift,
                    cc_max=cc_max,
                    cc_ratio=cc_ratio,
                    l2_power=l2_power
                )

    return new_data, new_synts