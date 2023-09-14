import typing as tp
import obspy
import numpy as np


def stream_grad_hess_win(data: obspy.Stream, synt: obspy.Stream,
                         dsyn: tp.List[obspy.Stream],
                         normalize: bool = True, verbose: float = False) \
        -> tp.Tuple[np.ndarray, np.ndarray]:
    """Computes the gradient and the approximate hessian of the cost function
    using the Frechet derivative of the forward modelled data.
    The stats object of the Traces in the stream _*must*_ contain both
    `windows` and the `tapers` attributes!

    Parameters
    ----------
    data : Stream
        data
    synt : Stream
        synthetics
    dsyn : Stream
        frechet derivatives

    Returns
    -------
    Tuple[float, float]
        Gradient, Approximate Hessian

    Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
    """

    g = np.zeros(len(dsyn))
    h = np.zeros((len(dsyn), len(dsyn)))

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data
            # Create trace list for the Frechet derivatives
            dsdm = []
            for ds in dsyn:
                dsdm.append(ds.select(network=network, station=station,
                                      component=component)[0].data)

            gt = np.zeros(len(dsyn))
            ht = np.zeros((len(dsyn), len(dsyn)))
            fnorm = 0

            # Loop over windows
            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                # Get data in windows
                wsyn = s[win.left:win.right]
                wobs = d[win.left:win.right]

                # Normalization factor on window
                fnorm += np.sum(tap * wobs ** 2) * dt

                # Compute Gradient
                for _i, _dsdm_i in enumerate(dsdm):
                    # Get derivate with respect to model parameter i
                    wdsdm_i = _dsdm_i[win.left:win.right]
                    gt[_i] += np.sum(((wsyn - wobs) * tap) * wdsdm_i) * dt

                    for _j, _dsdm_j in enumerate(dsdm):
                        # Get derivate with respect to model parameter j
                        wdsdm_j = _dsdm_j[win.left:win.right]
                        ht[_i, _j] += ((wdsdm_i * tap) @ (wdsdm_j * tap)) * dt

            g += gt/fnorm
            h += ht/fnorm

        except Exception as e:
            if verbose:
                print(f"When accessing {network}.{station}.{component}")
                print(e)

    return g, h
