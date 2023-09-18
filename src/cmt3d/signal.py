import numpy as np


def dlna(d, s, w=1.0):
    return 0.5 * np.log(np.sum(w * d ** 2) / np.sum(w * s ** 2))


def norm1(d, w=1.0, n=1.0):
    return n * np.sum(w * np.abs(d))


def norm2(d, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d)**2)


def dnorm1(d, s, w=1.0, n=1.0):
    return n * np.sum(w * np.abs(d-s))


def dnorm2(d, s, w=1.0, n=1.0):
    return 0.5 * n * np.sum(w * (d-s)**2)


def power_l1(arr1, arr2):
    """
    Power(L1 norm, abs) ratio of arr1 over arr2, unit in dB
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) not the same"
                         % (len(arr1), len(arr2)))
    return 10 * np.log10(np.sum(np.abs(arr1)) / np.sum(np.abs(arr2)))


def power_l2(arr1, arr2):
    """
    Power(L2 norm, square) ratio of arr1 over arr2, unit in dB.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Length of arr1(%d) and arr2(%d) not the same"
                         % (len(arr1), len(arr2)))
    return 10 * np.log10(np.sum(arr1 ** 2) / np.sum(arr2 ** 2))


def xcorr(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    # Normalized cross correlation.
    max_cc_value = cc.max() / np.sqrt((s ** 2).sum() * (d ** 2).sum())
    return max_cc_value, time_shift


def correct_window_index(istart, iend, nshift, npts):
    """Correct the window index based on cross-correlation shift

    Parameters
    ----------
    istart : int
        start index
    iend : int
        end index
    nshift : int
        shift in N samples
    npts : int
        Length of window

    Returns
    -------
    Tuple
        indeces

    Raises
    ------
    ValueError
        If resulting windows arent the same length? I don't get this
    """
    istart_d = max(1, istart + nshift)
    iend_d = min(npts, iend + nshift)
    istart_s = max(1, istart_d - nshift)
    iend_s = min(npts, iend_d - nshift)
    if (iend_d - istart_d) != (iend_s - istart_s):
        raise ValueError("After correction, window length not the same: "
                         "[%d, %d] and [%d, %d]" % (istart_d, iend_d,
                                                    istart_s, iend_s))
    return istart_d, iend_d, istart_s, iend_s
