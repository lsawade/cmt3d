import os
import cmt3d
import pickle
from glob import glob
from typing import Optional
from future.utils import lmap
from obspy import UTCDateTime, Stream
import logging
import numpy as np
import pandas as pd
import obsproclib as opr
from scipy.linalg import svdvals
from . import signal


def get_event_files(database: str, label: str):

    # Get all the event directories in a database
    dirlist = os.listdir(database)

    # Check whether directory contains a solution with a label
    files = filter(
        lambda x: os.path.exists(os.path.join(database, x, f"{x}_{label}")),
        dirlist)

    # Return list with labeled files
    return list(map(lambda x: os.path.join(database, x, f"{x}_{label}"), files))


def read_measurements_label(cmtdir: str, label: str):

    measurement_pickle = os.path.join(
        cmtdir, f"measurements_{label}.pkl")

    try:
        with open(measurement_pickle, "rb") as f:
            measurements = pickle.load(f)

        return measurements

    except Exception:
        return None

def get_toffset(
        tsample: int, dt: float, t0: UTCDateTime, origin: UTCDateTime) -> float:
    """Computes the time of a sample with respect to origin time

    Parameters
    ----------
    tsample : int
        sample on trace
    dt : float
        sample spacing
    t0 : UTCDateTime
        time of the first sample
    origin : UTCDateTime
        origin time

    Returns
    -------
    float
        Time relative to origin time
    """

    # Second on trace
    trsec = (tsample*dt)
    return (t0 + trsec) - origin


def get_measurements_and_windows(
        obs: Stream, syn: Stream, event: cmt3d.CMTSource, logger: logging.Logger):
    """Make measurements on two correpsonding streams.

    Parameters
    ----------
    obs : Stream
        Observed stream
    syn : Stream
        synthetic stream
    event : lseis.CMTSource
        event
    logger : logging.Logger, optional
        Logger. Default None


    Returns
    -------
    dict
        dictionary with measurements for each component.
    """

    if logger is None:
        logger = logging.getLogger('cmt3d')

    windows = dict()

    # Create dict to access traces
    for _component in ["R", "T", "Z"]:
        windows[_component] = dict()
        windows[_component]["id"] = []
        windows[_component]["dt"] = []
        windows[_component]["starttime"] = []
        windows[_component]["endtime"] = []
        windows[_component]["nsamples"] = []
        windows[_component]["latitude"] = []
        windows[_component]["longitude"] = []
        windows[_component]["distance"] = []
        windows[_component]["azimuth"] = []
        windows[_component]["back_azimuth"] = []
        windows[_component]["nshift"] = []
        windows[_component]["time_shift"] = []
        windows[_component]["maxcc"] = []
        windows[_component]["dlna"] = []
        windows[_component]["L1"] = []
        windows[_component]["L2"] = []
        windows[_component]["dL1"] = []
        windows[_component]["dL2"] = []
        windows[_component]["trace_energy"] = []
        windows[_component]["L1_Power"] = []
        windows[_component]["L2_Power"] = []
        windows[_component]["corr_ratio"] = []

        for _tr in obs:
            if _tr.stats.component == _component \
                    and "windows" in _tr.stats:

                d = _tr.data
                try:
                    network, station, component = (
                        _tr.stats.network, _tr.stats.station,
                        _tr.stats.component)
                    s = syn.select(
                        network=network, station=station,
                        component=component)[0].data
                except Exception as e:
                    logger.warning(
                        f"{network}.{station}..{component}")
                    logger.error(e)
                    continue

                trace_energy = 0
                for win in _tr.stats.windows:
                    # Get window data
                    wd = d[win.left:win.right]
                    ws = s[win.left:win.right]

                    # Infos
                    dt = _tr.stats.delta
                    npts = _tr.stats.npts
                    winleft = get_toffset(
                        win.left, dt, win.time_of_first_sample,
                        event.origin_time)
                    winright = get_toffset(
                        win.right, dt, win.time_of_first_sample,
                        event.origin_time)

                    # Measurements
                    max_cc_value, nshift = signal.xcorr(wd, ws)

                    # Get fixed window indeces.
                    try:
                        istart, iend = win.left, win.right
                        istart_d, iend_d, istart_s, iend_s = \
                            signal.correct_window_index(
                                istart, iend, nshift, npts)
                        wd_fix = d[istart_d:iend_d]
                        ws_fix = s[istart_s:iend_s]
                    except ValueError as ve:
                        logger.warning(
                            f"Window [{winleft}, {winright}] on trace {_tr.id} "
                            f"was not taken into account: {ve}"
                        )
                        continue

                    # Populate the dictionary
                    windows[_component]["id"].append(_tr.id)
                    windows[_component]["dt"].append(dt)
                    windows[_component]["starttime"].append(winleft)
                    windows[_component]["endtime"].append(winright)
                    windows[_component]["latitude"].append(
                        _tr.stats.latitude
                    )
                    windows[_component]["longitude"].append(
                        _tr.stats.longitude
                    )
                    windows[_component]["distance"].append(
                        _tr.stats.distance
                    )
                    windows[_component]["azimuth"].append(
                        _tr.stats.azimuth
                    )
                    windows[_component]["back_azimuth"].append(
                        _tr.stats.back_azimuth
                    )

                    powerl1 = signal.power_l1(wd, ws)
                    powerl2 = signal.power_l2(wd, ws)
                    norm1 = signal.norm1(wd)
                    norm2 = signal.norm2(wd)
                    dnorm1 = signal.dnorm1(wd, ws)
                    dnorm2 = signal.dnorm2(wd, ws)
                    dlna = signal.dlna(wd_fix, ws_fix)
                    trace_energy += norm2

                    windows[_component]["L1"].append(norm1)
                    windows[_component]["L2"].append(norm2)
                    windows[_component]["dL1"].append(dnorm1)
                    windows[_component]["dL2"].append(dnorm2)
                    windows[_component]["dlna"].append(dlna)
                    windows[_component]["L1_Power"].append(powerl1)
                    windows[_component]["L2_Power"].append(powerl2)
                    windows[_component]["nshift"].append(nshift)
                    windows[_component]["time_shift"].append(
                        nshift * dt
                    )
                    windows[_component]["maxcc"].append(
                        max_cc_value
                    )
                    windows[_component]["corr_ratio"].append(
                        np.sum(wd_fix * ws_fix)/np.sum(ws_fix ** 2)
                    )
                # Create array with the energy
                windows[_component]["trace_energy"].extend(
                    [trace_energy]*len(_tr.stats.windows))

    return windows


def get_all_measurements(
        datadict: dict, syntdict: dict, event: cmt3d.CMTSource,
        logger: Optional[logging.Logger] = None):

    window_dict = dict()

    for _wtype, _obs_stream in datadict.items():

        # Get corresponding Synthetic data
        _syn_stream = syntdict[_wtype]["synt"]

        window_dict[_wtype] = get_measurements_and_windows(
            _obs_stream, _syn_stream, event, logger=logger)

    return window_dict


def get_measurement_N(
        database0: str, label0: str,
        database1: str, label1: str,
        mlabel0: str = None, mlabel1: str = None,
        v: bool = True,
        outfile: str = None,
        catalog0: str = None,
        catalog1: str = None):
    """Takes in databse locations and labels to create a table that contains
    measurement count vs. parameter change.

    Parameters
    ----------
    database0 : str
        Starting database
    label0 : str
        label of starting solution
    mlabel0 : str, optional,
        if the starting label of the measurement differs from the one of the
        solution
    database1 : str
        Final database
    label1 : str
        label of final solution
    mlabel1 : str, optional
        if the final measurement label differs from the cmtfile one
    v : bool, optional
        flag to turn on verbose output
    outfile : str, optional
        save to feather file
    catalog0 : str, optional
        optional catalog input to not require file search
    catalog1 : str, optional
        optional catalog input to not require file search

    Returns
    -------
    Arraylike table
        CID, ddepth, dM0, dcmt, dx, *[measurement counts for wave types]

    """

    # Set labels if not provided
    if not mlabel0:
        mlabel0 = label0

    if not mlabel1:
        mlabel1 = label1

    # Loading or Creating catalog 0
    if catalog0:

        # Create catalogs
        if v:
            print("Loading Catalog0...")
        cat0 = lseis.CMTCatalog.load(catalog0)

    else:
        # Get all cmtfiles
        if v:
            print("Get events for cat 0...")
        cmtfiles0 = get_event_files(database0, label0)

        # Create catalog
        if v:
            print("Create Catalog 0...")
        cat0 = lseis.CMTCatalog.from_file_list(cmtfiles0)

    # Loading or Creating catalog 0
    if catalog1:

        # load catalog
        if v:
            print("Loading Catalog1...")
        cat1 = lseis.CMTCatalog.load(catalog1)

    else:

        # Get cmtfiles
        if v:
            print("Get events for cat 0...")
        cmtfiles1 = get_event_files(database1, label1)

        # Create catalogs
        if v:
            print("Create Catalog 0...")
        cat1 = lseis.CMTCatalog.from_file_list(cmtfiles1)

    if v:
        print("Check ids...")
    cat0, cat1 = cat0.check_ids(cat1)

    # Waves and components
    mtype = 'dlna'  # placeholder measurement
    waves = ['body', 'surface', 'mantle']
    comps = ['Z', 'R', 'T']
    wcomb = [f"{w}-{c}" for w in waves for c in comps]

    # Create numpy structure dtype
    columns = ['event', 'date', 'dz', 'dM0', 'dt', 'dx', *wcomb]

    Nm = []

    for cmt0, cmt1 in zip(cat0, cat1):
        if v:
            print(f"Adding {cmt0.eventname} ...", end='\r')

        # Compute changes
        dz = (cmt1.depth_in_m - cmt0.depth_in_m)/1000.0
        dt = cmt1.time_shift - cmt0.time_shift
        dM0 = (cmt1.M0 - cmt0.M0)/cmt0.M0
        dx = lmap.haversine(
            cmt0.longitude, cmt0.latitude, cmt1.longitude, cmt1.latitude)

        # Get number of measurements involved
        d = read_measurements_label(
            os.path.join(database1, cmt1.eventname),  mlabel1)

        # Empty list to be used for the table
        mlist = 9 * [np.nan]

        # Fill if possible
        if isinstance(d, dict):

            counter = 0
            for w in waves:
                if w not in d:
                    counter += 3
                    continue
                for c in comps:
                    mlist[counter] = len(d[w][c][mtype])
                    counter += 1

        Nm.append(
            (cmt1.eventname, cmt1.cmt_time.matplotlib_date, dz, dM0, dt, dx, *mlist)
        )

    # Create table from measurements
    df = pd.DataFrame(Nm, columns=columns)

    if outfile:
        if 'feather' in outfile:
            df.to_feather(outfile)
        else:
            df.to_pickle(outfile)

    return df


def get_eigenvalues(
        database: str,
        label: str,
        v: bool = True,
        outfile: str = None,
        catalog: str = None):
    """Takes in databse locations and labels to create a table that contains
    measurement count vs. parameter change.

    Parameters
    ----------
    database : str
        Starting database
    label : str
        label of final solution
    v : bool, optional
        flag to turn on verbose output
    outfile : str, optional
        save to feather file
    catalog : str, optional
        optional catalog input to not require file search

    Returns
    -------
    Arraylike table
        CID, time, *[sorted eigenvalues]

    """

    # Loading or Creating catalog 0
    if catalog:

        # Create catalogs
        if v:
            print("Loading Catalog0...")
        cat = lseis.CMTCatalog.load(catalog)

    else:
        # Get all cmtfiles
        if v:
            print("Get events for cat 0...")
        cmtfiles = get_event_files(database, label)

        # Create catalog
        if v:
            print("Create Catalog 0...")
        cat = lseis.CMTCatalog.from_file_list(cmtfiles)

    # Waves and components
    eigv = [f'{i}' for i in range(10)]

    # Create numpy structure dtype
    columns = ['event', 'date', *eigv]

    Nm = []

    for cmt in cat:
        if v:
            print(f"Adding {cmt.eventname} ...", end='\r')

        # Get number of measurements involved
        HH = np.load(os.path.join(database, cmt.eventname, 'summary.npz'))[
            'hessians'
        ]
        print(HH)

        HH = HH[-1]
        HH = HH.reshape(10, 10)

        # eig = np.sort(np.linalg.eigvals(HH.squeeze()))[::-1].tolist()
        eig = np.sort(svdvals(HH.squeeze()))[::-1].tolist()

        Nm.append(
            (cmt.eventname, cmt.cmt_time.matplotlib_date, *eig)
        )

    # Create table from measurements
    df = pd.DataFrame(Nm, columns=columns)

    if outfile:
        if 'feather' in outfile:
            df.to_feather(outfile)
        else:
            df.to_pickle(outfile)

    return df


def get_damping_params(
        database: str,
        label: str,
        v: bool = True,
        outfile: str = None,
        catalog: str = None):
    """Takes in databse locations and labels to create a table that contains
    the model norm and the damping parameter

    Parameters
    ----------
    database : str
        Starting database
    label : str
        label of final solution
    v : bool, optional
        flag to turn on verbose output
    outfile : str, optional
        save to feather file
    catalog : str, optional
        optional catalog input to not require file search

    Returns
    -------
    Arraylike table
        CID, time, *[sorted eigenvalues]

    """

    # Loading or Creating catalog 0
    if catalog:

        # Create catalogs
        if v:
            print("Loading Catalog0...")
        cat = lseis.CMTCatalog.load(catalog)

    else:
        # Get all cmtfiles
        if v:
            print("Get events for cat 0...")
        cmtfiles = get_event_files(database, label)

        # Create catalog
        if v:
            print("Create Catalog 0...")
        cat = lseis.CMTCatalog.from_file_list(cmtfiles)

    # Create numpy structure dtype
    columns = ['event', 'date', 'modelnorm', 'cost']

    Nm = []

    for cmt in cat:
        if v:
            print(f"Adding {cmt.eventname} ...", end='\r')

        # Get number of measurements involved
        summary = np.load(os.path.join(database, cmt.eventname, 'summary.npz'))
        mnorm = np.sqrt(np.sum((summary['model']-summary['init_model'])**2))
        fcost = summary['cost']

        Nm.append(
            (cmt.eventname, cmt.cmt_time.matplotlib_date, mnorm, fcost)
        )

    # Create table from measurements
    df = pd.DataFrame(Nm, columns=columns)

    if outfile:
        if 'feather' in outfile:
            df.to_feather(outfile)
        else:
            df.to_pickle(outfile)

    return df


def bin_summary():

    import argparse
    import sys

    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='database0', help='starting database', type=str)
    parser.add_argument(dest='label0', help='starting label', type=str)
    parser.add_argument(dest='database1', help='final database', type=str)
    parser.add_argument(dest='label1', help='final label', type=str)
    parser.add_argument(dest='outfile', help='verbose output', type=str)
    parser.add_argument('-m0', '--mlabel0', dest='mlabel0',
                        help='label for cmt in dir',
                        required=False, type=str, default=None)
    parser.add_argument('-m1', '--mlabel1', dest='mlabel1',
                        help='label for cmt in dir',
                        required=False, type=str, default=None)
    parser.add_argument('-v', '--verbose', dest='verbose',
                        help='verbose output', action='store_true',
                        required=False, default=False)
    parser.add_argument('-c0', '--catalog0', dest='catalog0',
                        help='Start Catalog', required=False, type=str, default=None)
    parser.add_argument('-c1', '--catalog1', dest='catalog1',
                        help='Final Catalog', required=False, type=str, default=None)

    args = parser.parse_args()

    get_measurement_N(
        database0=args.database0,
        label0=args.label0,
        database1=args.database1,
        label1=args.label1,
        mlabel0=args.mlabel0,
        mlabel1=args.mlabel1,
        v=args.verbose,
        outfile=args.outfile,
        catalog0=args.catalog0,
        catalog1=args.catalog1,
    )