import time
import socket
import typing as tp
import pickle as pickle
import yaml
import numpy as np


def to_delta_gamma(v, w):
    """Converts from Tape2015 parameters to lune coordinates"""
    return to_delta(w), to_gamma(v)


def to_gamma(v):
    """Converts from Tape2015 parameter v to lune longitude"""
    gamma = (1.0 / 3.0) * np.arcsin(3.0 * v)
    return np.rad2deg(gamma)


def to_delta(w):
    """Converts from Tape2015 parameter w to lune latitude"""
    beta0 = np.linspace(0, np.pi, 100)
    u0 = 0.75 * beta0 - 0.5 * np.sin(2.0 * beta0) + 0.0625 * np.sin(4.0 * beta0)
    beta = np.interp(3.0 * np.pi / 8.0 - w, u0, beta0)
    delta = np.rad2deg(np.pi / 2.0 - beta)
    return delta


def lune2mt(rho, v, w, kappa, sigma, h):
    """Converts from lune parameters to moment tensor parameters
    (up-south-east convention)
    """
    kR3 = np.sqrt(3.0)
    k2R6 = 2.0 * np.sqrt(6.0)
    k2R3 = 2.0 * np.sqrt(3.0)
    k4R6 = 4.0 * np.sqrt(6.0)
    k8R6 = 8.0 * np.sqrt(6.0)

    m0 = rho / np.sqrt(2.0)

    delta, gamma = to_delta_gamma(v, w)
    beta = 90.0 - delta

    gamma = np.deg2rad(gamma)
    beta = np.deg2rad(90.0 - delta)
    kappa = np.deg2rad(kappa)
    sigma = np.deg2rad(sigma)
    theta = np.arccos(h)

    Cb = np.cos(beta)
    Cg = np.cos(gamma)
    Cs = np.cos(sigma)
    Ct = np.cos(theta)
    Ck = np.cos(kappa)
    C2k = np.cos(2.0 * kappa)
    C2s = np.cos(2.0 * sigma)
    C2t = np.cos(2.0 * theta)

    Sb = np.sin(beta)
    Sg = np.sin(gamma)
    Ss = np.sin(sigma)
    St = np.sin(theta)
    Sk = np.sin(kappa)
    S2k = np.sin(2.0 * kappa)
    S2s = np.sin(2.0 * sigma)
    S2t = np.sin(2.0 * theta)

    mt0 = (
        m0
        * (1.0 / 12.0)
        * (
            k4R6 * Cb
            + Sb
            * (
                kR3 * Sg * (-1.0 - 3.0 * C2t + 6.0 * C2s * St * St)
                + 12.0 * Cg * S2t * Ss
            )
        )
    )

    mt1 = (
        m0
        * (1.0 / 24.0)
        * (
            k8R6 * Cb
            + Sb
            * (
                -24.0 * Cg * (Cs * St * S2k + S2t * Sk * Sk * Ss)
                + kR3
                * Sg
                * (
                    (1.0 + 3.0 * C2k) * (1.0 - 3.0 * C2s)
                    + 12.0 * C2t * Cs * Cs * Sk * Sk
                    - 12.0 * Ct * S2k * S2s
                )
            )
        )
    )

    mt2 = (
        m0
        * (1.0 / 6.0)
        * (
            k2R6 * Cb
            + Sb
            * (
                kR3 * Ct * Ct * Ck * Ck * (1.0 + 3.0 * C2s) * Sg
                - k2R3 * Ck * Ck * Sg * St * St
                + kR3 * (1.0 - 3.0 * C2s) * Sg * Sk * Sk
                + 6.0 * Cg * Cs * St * S2k
                + 3.0 * Ct * (-4.0 * Cg * Ck * Ck * St * Ss + kR3 * Sg * S2k * S2s)
            )
        )
    )

    mt3 = (
        m0
        * (-1.0 / 2.0)
        * Sb
        * (
            k2R3 * Cs * Sg * St * (Ct * Cs * Sk - Ck * Ss)
            + 2.0 * Cg * (Ct * Ck * Cs + C2t * Sk * Ss)
        )
    )

    mt4 = (
        -m0
        * (1.0 / 2.0)
        * Sb
        * (
            Ck * (kR3 * Cs * Cs * Sg * S2t + 2.0 * Cg * C2t * Ss)
            + Sk * (-2.0 * Cg * Ct * Cs + kR3 * Sg * St * S2s)
        )
    )

    mt5 = (
        -m0
        * (1.0 / 8.0)
        * Sb
        * (
            4.0 * Cg * (2.0 * C2k * Cs * St + S2t * S2k * Ss)
            + kR3
            * Sg
            * ((1.0 - 2.0 * C2t * Cs * Cs - 3.0 * C2s) * S2k + 4.0 * Ct * C2k * S2s)
        )
    )

    if type(mt0) is np.ndarray:
        return np.column_stack([mt0, mt1, mt2, mt3, mt4, mt5])
    else:
        return np.array([mt0, mt1, mt2, mt3, mt4, mt5])


def get_comm_rank_size():
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    return comm, rank, size


def write_pickle(filename, obj):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    with open(filename, "rb") as f:
        obj = pickle.load(f)

    return obj


def read_yaml(filename: str):
    """Read a yaml file"""
    with open(filename, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(data: dict, filename: str):
    """Write a yaml file"""
    with open(filename, "w") as f:
        yaml.dump(data, f)


def sec2hhmmss(
    seconds: float, roundsecs: bool = True
) -> tp.Tuple[int, int, float | int]:
    """Turns seconds into tuple of (hours, minutes, seconds)

    Parameters
    ----------
    seconds : float
        seconds

    Returns
    -------
    Tuple
        (hours, minutes, seconds)

    Notes
    -----
    :Author:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.03.05 18.44

    """

    # Get hours
    hh = int(seconds // 3600)

    # Get minutes
    mm = int((seconds - hh * 3600) // 60)

    # Get seconds
    ss = seconds - hh * 3600 - mm * 60

    if roundsecs:
        ss = round(ss)

    return (hh, mm, ss)


def sec2timestamp(seconds: float) -> str:
    """Gets time stamp from seconds in format "hh h mm m ss s"

    Parameters
    ----------
    seconds : float
        Seconds to get string from

    Returns
    -------
    str
        output timestamp
    """

    hh, mm, ss = sec2hhmmss(seconds)
    return f"{int(hh):02} h {int(mm):02} m {int(ss):02} s"


class retry:
    """
    Decorator that will keep retrying the operation after a timeout.

    Useful for remote operations that are prone to fail spuriously.
    """

    def __init__(self, retries: int, wait_time: float):
        self.retries = retries
        self.wait_time = wait_time

    def __call__(self, f: tp.Callable):
        def wrapped_f(*args, **kwargs):
            for i in range(self.retries):
                try:
                    retval = f(*args, **kwargs)
                except Exception as e:
                    print(f"Retry caught exception try {i+1}/{self.retries}")
                    print(e)
                    time.sleep(self.wait_time)
                    continue
                else:
                    return retval

            raise ValueError(f"Failed after {self.retries} retries.")

        return wrapped_f


def chunkfunc(sequence: tp.Sequence, n: int):
    """
    Converse to split, this function preserves the order of the events. But,
    the last chunk has potentially only a single element. For the case, of
    I/O this is not so bad, but if you are looking for performance, `split()`
    above is the better option.

    sequence = sequence to be split
    n = number of elements per chunk

    return list of n-element chunks where the number of chunks depends on the
    length of the sequence
    """
    n = max(1, n)
    return [sequence[i : i + n] for i in range(0, len(sequence), n)]
