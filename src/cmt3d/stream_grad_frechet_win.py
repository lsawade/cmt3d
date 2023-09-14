import obspy
import numpy as np


def stream_grad_frechet_win(data: obspy.Stream, synt: obspy.Stream,
                            dsyn: obspy.Stream, normalize: bool = True,
                            verbose: float = False) -> float:
    """Computes the gradient of a the least squares cost function wrt. 1
    parameter given its forward computation and the frechet derivative.
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
    float
        gradient

    Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
    """

    x = 0.0

    for tr in data:
        network, station, component = (
            tr.stats.network, tr.stats.station, tr.stats.component)

        # Get the trace sampling time
        dt = tr.stats.delta
        d = tr.data

        try:
            s = synt.select(network=network, station=station,
                            component=component)[0].data
            dsdm = dsyn.select(network=network, station=station,
                               component=component)[0].data
            costt = 0
            fnorm = 0
            for win, tap in zip(tr.stats.windows, tr.stats.tapers):

                wsyn = s[win.left:win.right]
                wobs = d[win.left:win.right]
                wdsdm = dsdm[win.left:win.right]
                costt += np.sum((wsyn - wobs) * wdsdm * tap) * dt
                fnorm += np.sum(tap * wobs ** 2) * dt

            if normalize:
                x += costt/fnorm
            else:
                x += costt

        except Exception as e:
            if verbose:
                print(
                    f"Error - Gradient - {network}.{station}.{component}: {e}")

    return x/len(data)
