# %%
import os
from copy import deepcopy
import cmt3d
import obspy
import obsproclib as oprc
import obswinlib as owl


from .constants import Constants
from .model import read_model_names, read_perturbation
from .kernel import write_dsdm, read_dsdm_raw
from .forward import write_synt, read_synt, read_synt_raw
from .data import write_data, read_data, write_data_windowed
from .log import get_iter, get_step
from .utils import reset_cpu_affinity, isipython


def process_data(outdir):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity()

    # Get dir
    metadir = os.path.join(outdir, 'meta')

    # Get CMT
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Eventname
    eventname = cmtsource.eventname

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get data database
    datadatabase = inputparams["datadatabase"]

    # Get datadir in data database
    ddatadir = os.path.join(datadatabase, eventname)

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    data = obspy.read(os.path.join(ddatadir, 'waveforms', '*.mseed'))

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        sdata = deepcopy(data)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[_wtype]["process"])

        tprocessdict.pop("relative_starttime")
        tprocessdict.pop("relative_endtime")
        tprocessdict["starttime"] = starttime
        tprocessdict["endtime"] = endtime
        tprocessdict["inventory"] = stations
        tprocessdict.update(dict(
            remove_response_flag=True,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=True)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():
            pdata = oprc.process_stream(sdata, **tprocessdict)
        else:
            pdata = oprc.queue_multiprocess_stream(
                sdata, tprocessdict, nproc=multiprocesses)

        print(f"writing data {_wtype} for {outdir}")

        write_data(pdata, outdir, _wtype)


def process_synt(outdir, verbose=True):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity(verbose=True)

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Define directory
    metadir = os.path.join(outdir, 'meta')

    # Get CMT
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    synt = read_synt_raw(outdir)

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        if verbose:
            print(f"Processing {_wtype} ...")

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[_wtype]["process"])

        tprocessdict.pop("relative_starttime")
        tprocessdict.pop("relative_endtime")
        tprocessdict["starttime"] = starttime
        tprocessdict["endtime"] = endtime
        tprocessdict["inventory"] = stations
        tprocessdict.update(dict(
            remove_response_flag=False,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=False)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():

            # Verbose output
            if verbose:
                print("    ... in serial.")

            # Processing
            pdata = oprc.process_stream(sdata, **tprocessdict)

        else:

            # Verbose output
            if verbose:
                print(f"    ... in parallel using {multiprocesses} cores.")
            # This is sooo important for parallel processing
            # If you don't set this numpy, mkl, etc. will try to use threads
            # for processing, but you do not want that, because you want to
            # distribute work to the different cores manually. If this is not
            # set, the different cores will fight for threads!!!!
            os.environ["OMP_NUM_THREADS"] = "1"

            # Processing
            pdata = oprc.queue_multiprocess_stream(
                sdata, tprocessdict, nproc=multiprocesses, verbose=verbose)

        # Write synthetics
        write_synt(pdata, outdir, _wtype, it, ls)


def process_dsdm(outdir, nm, verbose=False):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity(verbose=verbose)

    # Get iter,step
    it = get_iter(outdir)
    ls = get_step(outdir)

    # Define directory
    metadir = os.path.join(outdir, 'meta')

    # Get CMT
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Read model and model name
    mname = read_model_names(outdir)[nm]
    perturbation = read_perturbation(outdir)[nm]

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read data
    if mname in Constants.nosimpars:
        synt = read_synt_raw(outdir)
    else:
        synt = read_dsdm_raw(outdir, nm)

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    # Process each wavetype.
    for _wtype in processdict.keys():

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[_wtype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[_wtype]["process"])

        tprocessdict.pop("relative_starttime")
        tprocessdict.pop("relative_endtime")
        tprocessdict["starttime"] = starttime
        tprocessdict["endtime"] = endtime
        tprocessdict["inventory"] = stations
        tprocessdict.update(dict(
            remove_response_flag=False,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=False)
        )

        # Choosing whether to process in
        # Multiprocessing does not work in ipython hence we check first
        # we are in an ipython environment
        if multiprocesses <= 1 or isipython():

            # Verbose output
            if verbose:
                print("    ... in serial.")

            # Processing
            pdata = oprc.process_stream(sdata, **tprocessdict)
        else:
            # Verbose output
            if verbose:
                print(f"    ... in parallel using {multiprocesses} cores.")

            # This is sooo important for parallel processing
            # If you don't set this numpy, mkl, etc. will try to use threads
            # for processing, but you do not want that, because you want to
            # distribute work to the different cores manually. If this is not
            # set, the different cores will fight for threads!!!!
            os.environ["OMP_NUM_THREADS"] = "1"

            # Processing
            pdata = oprc.queue_multiprocess_stream(
                sdata, tprocessdict, nproc=multiprocesses)

        # If Frechet derivative with respect to depth in m -> divide by 1000
        # since specfem outputs the derivative with respect to depth in km
        if mname == "depth_in_m":
            oprc.stream_multiply(pdata, 1.0/1000.0)

        # Write synthetics
        write_dsdm(pdata, outdir, _wtype, nm, it, ls)


def wprocess_dsdm(args):
    process_dsdm(*args)


def merge_windows(data_stream: obspy.Stream, synt_stream: obspy.Stream):
    """
    After windowing, the windows are often directly adjacent. In such
    cases, we can simply unite the windows. The `merge_windows` method
    calls the appropriate functions to handle that.
    """

    for obs_tr in data_stream:
        try:
            synt_tr = synt_stream.select(
                station=obs_tr.stats.station,
                network=obs_tr.stats.network,
                component=obs_tr.stats.component)[0]
        except Exception as e:
            print(
                "Couldn't find corresponding synt for "
                f"obsd trace({obs_tr.id}): {e}")
            continue

        if len(obs_tr.stats.windows) > 1:
            obs_tr.stats.windows = owl.merge_trace_windows(
                obs_tr, synt_tr)


def window(outdir, verbose=True):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity()

    # Get dirs
    metadir = os.path.join(outdir, 'meta')

    # Get process parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Get input parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Read number of processes from input params
    multiprocesses = inputparams['multiprocesses']

    # Read stations
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    # Read obspy event
    xml_event = obspy.read_events(os.path.join(metadir, 'init_model.cmt'))

    # Window debug flag
    window_debug_flag = True
    taper_debug_flag = True

    # Loop over
    for _wtype in processdict.keys():

        if verbose:
            print(f"Windowing {_wtype} ...")

        # Read synthetics and data
        synt = read_synt(outdir, _wtype, 0, 0)
        data = read_data(outdir, _wtype)

        for window_dict in processdict[_wtype]["window"]:

            if isinstance(window_dict, dict):
                window_dict = [window_dict]

            for _window_dict in window_dict:

                # Wrap window dictionary
                wrapwindowdict = dict(
                    station=stations,
                    event=xml_event,
                    config_dict=_window_dict,
                    _verbose=window_debug_flag
                )

                # Serial or Multiprocessing
                if multiprocesses <= 1:
                    # Verbose output
                    if verbose:
                        print("    ... in serial")

                    # Windowing
                    owl.window_on_stream(data, synt, **wrapwindowdict)

                else:

                    # Verbose output
                    if verbose:
                        print(f"    ...in parallel using {multiprocesses} cores.")

                    # Windowing
                    data = owl.queue_multiwindow_stream(
                        data, synt,
                        wrapwindowdict, nproc=multiprocesses, verbose=verbose)

        if len(processdict[_wtype]["window"]) > 1:
            merge_windows(data, synt)

        # After each trace has windows attached continue
        owl.add_tapers(data, taper_type="tukey",
                       alpha=0.25, verbose=taper_debug_flag)

        # Some traces aren't even iterated over..
        for _tr in data:
            if "windows" not in _tr.stats:
                _tr.stats.windows = []

        # Write windowed data
        write_data_windowed(data, outdir, _wtype)


def wwindow(args):
    window(*args)


def bin():

    from sys import argv

    outdir, metadir, datadir = argv[1:]

    process_data(outdir)


if __name__ == "__main__":
    bin()
