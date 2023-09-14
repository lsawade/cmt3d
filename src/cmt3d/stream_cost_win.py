import numpy as np
import obspy


def stream_cost_win(data: obspy.Stream, synt: obspy.Stream,
                    normalize: bool = True, verbose: bool = False) -> float:
    """Takes in data and synthetics stream and computes windowed least squares
    cost. The stats object of the Traces in the stream _*must*_ contain both
    `windows` and the `tapers` attributes!

    Parameters
    ----------
    data : Stream
        data
    synt : Stream
        synthetics

    Returns
    -------
    float
        cost of entire stream

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

            fnorm = 0
            costt = 0
            for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                ws = s[win.left:win.right]
                wo = d[win.left:win.right]
                costt += 0.5 * (np.sum(tap * (ws - wo) ** 2) * dt)
                fnorm += np.sum(tap * wo ** 2) * dt

            if normalize:
                x += costt/fnorm
            else:
                x += costt

        except Exception as e:
            if verbose:
                print(f"Error at ({network}.{station}.{component}): {e}")

    return x
