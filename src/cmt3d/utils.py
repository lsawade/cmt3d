import time
import socket
import typing as tp
import pickle as pickle
import yaml


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
