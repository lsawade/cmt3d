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


def process_data(outdir, multiprocesses=1):

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    for wavetype in processdict.keys():
        process_data_wave(outdir, wavetype, multiprocesses=multiprocesses)


def process_data_wave(outdir, wavetype, multiprocesses=1):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity()

    # Get dir
    metadir = os.path.join(outdir, 'meta')

    # Get CMT
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'))

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

    # Read data
    data = obspy.read(os.path.join(ddatadir, 'waveforms', '*.mseed'))

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    sdata = deepcopy(data)

    # Call processing function and processing dictionary
    starttime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_starttime"]
    endtime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_endtime"]

    # Process dict
    tprocessdict = deepcopy(processdict[wavetype]["process"])

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

    print(f"writing data {wavetype} for {outdir}")

    write_data(pdata, outdir, wavetype)


def process_data_wave_mpi(outdir, wavetype, verbose=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if verbose:
            print(f"Processing {wavetype}")
            print("-> Loading parameters")

        # Get dir
        metadir = os.path.join(outdir, 'meta')

        # Get CMT
        cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
            metadir, 'init_model.cmt'))

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

        # Read data
        data = obspy.read(os.path.join(ddatadir, 'waveforms', '*.mseed'))

        # Read metadata
        stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[wavetype]["process"])

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

    else:

        data = None
        tprocessdict = None


    if verbose and rank == 0:
        print("-> Setting OMP threads to 1")

    os.environ["OMP_NUM_THREADS"] = "1"

    if verbose and rank == 0:
        print("-> Starting the processing")

    pdata = oprc.mpi_process_stream(data, tprocessdict, verbose=verbose)

    if verbose and rank == 0:
        print("-> Finished processing")

    if rank == 0:
        # Write synthetics
        print(f"-> writing data {wavetype} for {outdir}")

        write_data(pdata, outdir, wavetype)


def process_synt(outdir, multiprocesses=1, verbose=True):

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    for wavetype in processdict.keys():
        process_synt_wave(outdir, wavetype, multiprocesses=multiprocesses,
                          verbose=verbose)


def process_synt_wave(outdir, wavetype, multiprocesses=1, verbose=True):

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

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

    # Read data
    synt = read_synt_raw(outdir)

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    if verbose:
        print(f"Processing {wavetype} ...")

    sdata = deepcopy(synt)

    # Call processing function and processing dictionary
    starttime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_starttime"]
    endtime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_endtime"]

    # Process dict
    tprocessdict = deepcopy(processdict[wavetype]["process"])

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
            print(f"    ... in parallel using {multiprocesses} cores."
                    "and multiprocessing.")
        # This is sooo important for parallel processing
        # If you don't set this numpy, mkl, etc. will try to use threads
        # for processing, but you do not want that, because you want to
        # distribute work to the different cores manually. If this is not
        # set, the different cores will fight for threads!!!!
        os.environ["OMP_NUM_THREADS"] = "1"

        # Processing
        pdata = oprc.queue_multiprocess_stream(
            sdata, tprocessdict, nproc=multiprocesses, verbose=verbose)

        # Verbose output
        if verbose:
            print(f"    ... in parallel using {multiprocesses} cores."
                    "and mpi.")


    # Write synthetics
    write_synt(pdata, outdir, wavetype, it, ls)


def process_synt_wave_mpi(outdir, wavetype, verbose=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if verbose:
            print(f"Processing {wavetype}")
            print("-> Loading parameters")

        # Get processing parameters
        processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

        # Get iter,step
        it = get_iter(outdir)
        ls = get_step(outdir)

        # Define directory
        metadir = os.path.join(outdir, 'meta')

        # Get CMT
        cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
            metadir, 'init_model.cmt'
        ))

        # Read data
        synt = read_synt_raw(outdir)

        # Read metadata
        stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

        if verbose:
            print(f"Processing {wavetype} ...")

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[wavetype]["process"])

        tprocessdict.pop("relative_starttime")
        tprocessdict.pop("relative_endtime")
        tprocessdict["starttime"] = starttime
        tprocessdict["endtime"] = endtime
        tprocessdict["inventory"] = stations
        tprocessdict.update(dict(
            remove_response_flag=False,
            event_latitude=cmtsource.latitude,
            event_longitude=cmtsource.longitude,
            geodata=False))

        # Verbose output
        # This is sooo important for parallel processing
        # If you don't set this numpy, mkl, etc. will try to use threads
        # for processing, but you do not want that, because you want to
        # distribute work to the different cores manually. If this is not
        # set, the different cores will fight for threads!!!!
    else:
        sdata = None
        tprocessdict = None

    if verbose and rank == 0:
        print("-> Setting OMP threads to 1")

    os.environ["OMP_NUM_THREADS"] = "1"

    if verbose and rank == 0:
        print("-> Starting the processing")

    # Processing
    pdata = oprc.mpi_process_stream(sdata, tprocessdict, verbose=verbose)

    if verbose and rank == 0:
        print("-> Finished processing")

    if rank == 0:
        # Write synthetics
        write_synt(pdata, outdir, wavetype, it, ls)


def process_dsdm(outdir, nm, multiprocesses=1, verbose=False):

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    for wavetype in processdict.keys():
        process_dsdm_wave(outdir, nm, wavetype, multiprocesses=multiprocesses,
                          verbose=verbose)


def process_dsdm_wave(outdir, nm, wavetype, multiprocesses=1, verbose=True):


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

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Read data
    if mname in Constants.nosimpars:
        synt = read_synt_raw(outdir)
    else:
        synt = read_dsdm_raw(outdir, nm)

    # Read metadata
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    sdata = deepcopy(synt)

    # Call processing function and processing dictionary
    starttime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_starttime"]
    endtime = cmtsource.cmt_time \
        + processdict[wavetype]["process"]["relative_endtime"]

    # Process dict
    tprocessdict = deepcopy(processdict[wavetype]["process"])

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
    # if mname == "depth_in_m":
    #     oprc.stream_multiply(pdata, 1.0/1000.0)

    # Write synthetics
    write_dsdm(pdata, outdir, wavetype, nm, it, ls)


def process_dsdm_wave_mpi(outdir, nm, wavetype, verbose=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if verbose:
            print(f"Processing {wavetype}")
            print("-> Loading parameters")

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

        # Get processing parameters
        processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

        # Read data
        if mname in Constants.nosimpars:
            synt = read_synt_raw(outdir)
        else:
            synt = read_dsdm_raw(outdir, nm)

        # Read metadata
        stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

        sdata = deepcopy(synt)

        # Call processing function and processing dictionary
        starttime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_starttime"]
        endtime = cmtsource.cmt_time \
            + processdict[wavetype]["process"]["relative_endtime"]

        # Process dict
        tprocessdict = deepcopy(processdict[wavetype]["process"])

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
    else:
        sdata = None
        tprocessdict = None

    if verbose and rank == 0:
        print("-> Setting OMP threads to 1")

    os.environ["OMP_NUM_THREADS"] = "1"

    if verbose and rank == 0:
        print("-> Starting the processing")

    # Processing
    pdata = oprc.mpi_process_stream(
        sdata, tprocessdict, verbose=verbose)

    if verbose and rank == 0:
        print("-> Finished processing")

    if rank == 0:
        # If Frechet derivative with respect to depth in m -> divide by 1000
        # since specfem outputs the derivative with respect to depth in km
        if mname == "depth_in_m":
            oprc.stream_multiply(pdata, 1.0/1000.0)

        # Write synthetics
        write_dsdm(pdata, outdir, wavetype, nm, it, ls)



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


def window(outdir, multiprocesses=1, verbose=True):

    # Get processing parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    for wavetype in processdict.keys():
        window_wave(outdir, wavetype, multiprocesses=multiprocesses,
                    verbose=verbose)


def window_wave(outdir, wavetype, multiprocesses=1, verbose=True):

    # Reset CPU affinity important for SUMMIT
    reset_cpu_affinity()

    # Get dirs
    metadir = os.path.join(outdir, 'meta')

    # Get process parameters
    processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

    # Read stations
    stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

    # Read obspy event
    xml_event = obspy.read_events(os.path.join(metadir, 'init_model.cmt'))

    # Window debug flag
    window_debug_flag = True
    taper_debug_flag = True

    # Loop over
    if verbose:
        print(f"Windowing {wavetype} ...")

    # Read synthetics and data
    synt = read_synt(outdir, wavetype, 0, 0)
    data = read_data(outdir, wavetype)

    for window_dict in processdict[wavetype]["window"]:

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

    if len(processdict[wavetype]["window"]) > 1:
        merge_windows(data, synt)

    # After each trace has windows attached continue
    owl.add_tapers(data, taper_type="tukey",
                   alpha=0.25, verbose=taper_debug_flag)

    # Some traces aren't even iterated over..
    for _tr in data:
        if "windows" not in _tr.stats:
            _tr.stats.windows = []

    # Write windowed data
    write_data_windowed(data, outdir, wavetype)


def window_wave_mpi(outdir, wavetype, verbose=True):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if verbose:
            print(f"Processing {wavetype}")
            print("-> Loading parameters")


        # Get dirs
        metadir = os.path.join(outdir, 'meta')

        # Get process parameters
        processdict = cmt3d.read_yaml(os.path.join(outdir, 'process.yml'))

        # Read stations
        stations = cmt3d.read_inventory(os.path.join(metadir, 'stations.xml'))

        # Read obspy event
        xml_event = obspy.read_events(os.path.join(metadir, 'init_model.cmt'))

        # Window debug flag
        window_debug_flag = True
        taper_debug_flag = True

        # Loop over
        if verbose:
            print(f"Windowing {wavetype} ...")

        # Read synthetics and data
        synt = read_synt(outdir, wavetype, 0, 0)
        data = read_data(outdir, wavetype)

        wrap_windowdicts = []
        for window_dict in processdict[wavetype]["window"]:

            if isinstance(window_dict, dict):
                window_dict = [window_dict]

            for _window_dict in window_dict:

                wrap_windowdicts.append(dict(
                    station=stations,
                    event=xml_event,
                    config_dict=_window_dict,
                    _verbose=window_debug_flag))
        NWD = len(wrap_windowdicts)
    else:
        synt = None
        data = None
        wrap_windowdicts = None
        NWD = None

    # Get the proper lengths
    NWD = comm.bcast(NWD, root=0)

    # Broadcast the window dictionaries
    wrap_windowdicts = comm.bcast(wrap_windowdicts, root=0)

    # Windowing
    for i in range(NWD):

        data = owl.mpi_window(data, synt, wrap_windowdicts[i], verbose=verbose)
        comm.Barrier()

    if rank == 0:

        if len(processdict[wavetype]["window"]) > 1:
            merge_windows(data, synt)

        # After each trace has windows attached continue
        owl.add_tapers(data, taper_type="tukey",
                       alpha=0.25, verbose=taper_debug_flag)

        # Some traces aren't even iterated over..
        for _tr in data:
            if "windows" not in _tr.stats:
                _tr.stats.windows = []

        # Write windowed data
        write_data_windowed(data, outdir, wavetype)


def wwindow(args):
    window(*args)


def bin():

    from sys import argv

    outdir, metadir, datadir = argv[1:]

    process_data(outdir)


if __name__ == "__main__":
    bin()
