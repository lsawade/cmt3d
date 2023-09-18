import numpy as np
from obspy.geodetics.base import gps2dist_azimuth


def azi_weights(lat0, lon0, lat, lon, weights=None, nbins=12, p=0.5):
    """Computes azimuthal weights from point and other locations.

    Parameters
    ----------
    lat0 : float
        point location
    lon0 : float
        location
    lat : arraylike
        other points
    lon : arraylike
        other points
    weights : arraylike, optional
        corresponding weights to other points, by default None
    nbins : int, optional
        number of bins, by default 12
    p : float, optional
        exponent for weight computation, by default 0.5

    Returns
    -------
    arraylike
        weights of for the 'other points'

    Notes
    -----

    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.16 16.30

    """

    # Compute azimuth
    def azfunc(lat1, lon1): return gps2dist_azimuth(lat0, lon0, lat1, lon1)[1]
    vazfunc = np.vectorize(azfunc)  # (not faster, but no loop necessary)

    # Compute
    az = vazfunc(lat, lon)

    # Bins
    bins = np.arange(0, 360.1, 360/nbins)

    # Histogram
    H, _ = np.histogram(az, bins=bins, weights=weights)

    # Find which az is in which bin
    binass = np.digitize(az, bins) - 1

    # Compute weights
    w = (1/H[binass])**p

    # Normalize
    w /= np.mean(w)

    return w
