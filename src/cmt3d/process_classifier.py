"""
This contains a class that controls how the waveforms are processed and how
they are weighted in the CMT inversion

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.orcopyleft/gpl.html)

Last Update: Sep 2023

"""

import numpy as np


def filter_scaling(startcorners, startmag, endcorners, endmag, newmag):
    """ This function is taking a set of corner frequencies or periods that
    should be in a sorted list or array and create taper scaling for the
    Global CMT workflow.

    Args:
        startcorners: starting filters
        startmag: magnitude of starting filter
        endcorners: new end boundaries
        endmag: magnitude of ending filter
        newmag: magnitude to query in between startmag and end mag

    Returns:
        list of corners

    .. rubric:: Explanation

    The Global CMT workflow changes the tapers for mantle waves
    linearly depending on the magnitude from magnitude 7.0 to 8.0. This is
    a Python implementation for exactly that function. It's a simple scaling,
    much scaling of an integral in elementary Calculus. It first finds the new
    bounds that depend on the change in starting and end magnitude and then the
    corresponding corner frequencies that bound the flat part, which is found
    by scaling the bounds.


    Examples:

        >>> startcorners = [125, 150, 300, 350]
        >>> endcorners = [200, 400]
        >>> startmag = 7.0
        >>> endmag = 8.0
        >>> newmag = 7.5
        >>> newcorners = filter_scaling(startcorners, startmag, endcorners,
        >>>                             endmag, newmag)
        >>> print(newcorners)
        [162.5         186.11111111  327.77777778  375.]

    """

    lower = np.min(startcorners) \
        + (np.min(endcorners) - np.min(startcorners)) \
        / (endmag - startmag) * (newmag - startmag)
    upper = np.max(startcorners) \
        + (np.max(endcorners) - np.max(startcorners)) \
        / (endmag - startmag) * (newmag - startmag)

    scaled_filter = (startcorners - np.min(startcorners)) \
        * (upper - lower) \
        / (np.max(startcorners) - np.min(startcorners)) + lower

    return scaled_filter.tolist()


def bodywave_params(mw: float):
    """Get body wave filter parameters and weight

    Parameters
    ----------
    mw : float
        moment magnitude

    Returns
    -------
    tuple of weight and filter
    """
    # First figure out body waves
    if mw > 7.5:
        weight = None
    elif mw < 6.5:
        weight = 1.0
    else:
        weight = (7.5 - mw) / (7.5 - 6.5)

    # Get filter parameters
    if mw < 6.5:
        filter = [150.0, 100.0, 50.0, 40.0]
    elif 6.5 <= mw <= 7.5:
        filter = [150.0, 100.0, 60.0, 50.0]
    else:
        filter = None

    # Record length
    time = 1.0 * 3600.0

    return weight, filter, time


def surfacewave_params(mw: float, depth_in_m: float):
    """Get surface wave weight and filter. No weight for events deeper
    that 300km.

    Parameters
    ----------
    mw : float
        moment magnitude
    depth_in_m : float
        depth in meters

    Returns
    -------
    tuple of weight and filter
    """

    # Get weight depeding on depth and moment magnitude
    if depth_in_m > 300000.0:
        weight = None
    else:
        if mw > 7.5:
            weight = None
        elif mw <= 6.5:
            weight = 1.0
        else:
            weight = (7.5 - mw) / (7.5 - 6.5)

    if mw <= 7.5:
        filter = [150.0, 100.0, 60.0, 50.0]
    else:
        filter = None

    # Record length
    time = 2.0 * 3600.0

    return weight, filter, time


def mantlewave_params(mw: float):
    """Uses the scaled corners from magnitude 7.0 to 8.0"""

    if mw > 6.5:
        weight = 1.0
    elif mw < 5.5:
        weight = None
    else:
        weight = (mw - 5.5)/(6.5 - 5.5)



    if mw < 5.5:
        filter = None
    elif 5.5 <= mw <= 7.0:
        filter = [350.0, 300.0, 150.0, 125.0]

    # For events larger than magnitude 7.0s
    elif 7.0 <= mw and mw <= 8.0:

        # Config values
        startcorners = [125.0, 150.0, 300.0, 350.0]
        endcorners = [200.0, 400.0]
        startmag = 7.0
        endmag = 8.0

        # Compute new corners
        newcorners = filter_scaling(startcorners, startmag,
                                    endcorners, endmag, mw)

        # Since the scaling works with a sorted list of periods.
        # We reverse the order so that largest period (lowest f) is the
        # first element of the corners filter
        filter = newcorners[::-1]

    else:
        # Config values
        startcorners = [125.0, 150.0, 300.0, 350.0]
        endcorners = [200.0, 400.0]
        startmag = 7.0
        endmag = 8.0

        # Compute new corners
        newcorners = filter_scaling(startcorners, startmag,
                                    endcorners, endmag, 8.0)

        filter = newcorners[::-1]

    # Record length
    time = 4 * 3600.0 - 400.0

    return weight, filter, time



def get_process_parameters(mw: float, depth_in_m: float):
    """Given a cmtsource this class determines processing scheme for the
    CMT inversion based on the 2012 Ekström article on the updated CMT
    inversion workflow.

    This class could have easily been a function, but the structure of the
    class just makes it so much simpler to modify and understand.

    Parameters:
    -----------
    mw : float
        Moment Magnitude
    depth_in_m : float
        Depth in meters


    .. rubric:: Example output dictionary

    .. code::

        {'body': {'filter': [150.0, 100.0, 60.0, 50.0],
                    'weight': 1.0},
            'mantle': {"filter": [350.0, 300.0, 150.0, 125.0],
                    'weight': 1.0},
            'surface': {'filter': [150.0, 100.0, 60.0, 50.0],
                        'weight': 1.0}

    """

    # Whether to use velocity as a measurement.
    velocity = bool(mw < 5.5)

    # Bodywave parameters
    bw_weight, bw_filter, bw_time = bodywave_params(mw)

    # Surface wave parameters
    sw_weight, sw_filter, sw_time = surfacewave_params(mw, depth_in_m)

    # Mantle wave parameters
    mw_weight, mw_filter, mw_time = mantlewave_params(mw)

    # Create dictionary that contains only the necessary entries
    # to clarify which ones are actual necessary.
    outdict = dict()
    normalization = 0

    if bw_weight is not None:
        normalization += bw_weight
    if sw_weight is not None:
        normalization += sw_weight
    if mw_weight is not None:
        normalization += mw_weight

    # Add body wave parameters if not None
    if bw_weight is not None and bw_weight != 0.0:
        outdict["body"] = dict(
            weight=float(bw_weight) / normalization,
            filter=bw_filter,
            relative_endtime=bw_time,
            velocity=velocity)

    # Add surface wave parameters if not None
    if sw_weight is not None \
            and sw_weight != 0.0:
        outdict["surface"] = dict(
            weight=float(sw_weight) / normalization,
            filter=sw_filter,
            relative_endtime=sw_time,
            velocity=velocity)

    # Add mantle wave parameters if not None
    if mw_weight is not None and mw_weight != 0.0:
        outdict["mantle"] = dict(
            weight=float(mw_weight) / normalization,
            filter=mw_filter,
            relative_endtime=mw_time,
            velocity=False)

    return outdict
