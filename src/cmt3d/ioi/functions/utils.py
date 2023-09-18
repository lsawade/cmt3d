import psutil
import os
import shutil
import numpy as np
import obspy
from pprint import pprint
import cmt3d
from gf3d.source import CMTSOLUTION
from .constants import Constants
from .model import read_model_names, read_scaling, write_model, \
    write_model_names, write_scaling, write_perturbation
from .log import reset_iter, reset_step, write_status



def createdir(cdir):
    """"Creates directory tree of specified path if it doesn't exist yet

    Parameters
    ----------
    cdir : str
        Path for building directory tree
    """
    if not os.path.exists(cdir):
        os.makedirs(cdir)


def rmdir(cdir):
    """Removes directory tree if it doesnt exist yet

    Parameters
    ----------
    cdir : str
        Removes directory recursively
    """
    shutil.rmtree(cdir)


def check_mt_inv(mnames):
    """Given the model parameter names. This function checks whether
    the inversion is for a moment tensor."""

    # If one moment tensor parameter is given all must be given.
    if any([_par in mnames for _par in Constants.mt_params]):
        checklist = [_par for _par in Constants.mt_params if _par in mnames]
        if not all([_par in checklist for _par in Constants.mt_params]):
            raise ValueError(
                "If one moment tensor parameter is to be "
                "inverted. All must be inverted.\n"
                "Update your parameters.")
        else:
            moment_tensor_inv = True
    else:
        moment_tensor_inv = False

    return moment_tensor_inv


def check_invertible(mnames):
    """Checks whether all parameter can be inverted for."""

    for _par in mnames:
        if _par not in Constants.parameter_check_list:
            raise ValueError(
                f"{_par} not supported at this point. \n"
                f"Available parameters are {Constants.parameter_check_list}")


def downloaddir(inputfile, cmtfilename, get_dirs_only=False):

    # MPI escape!
    if isinstance(inputfile, dict):
        input_params = inputfile
    else:
        # Read inputfile
        input_params = cmt3d.read_yaml(inputfile)

    # Get database location
    databasedir = input_params["datadatabase"]

    # Read CMT file
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get full filename
    outdir = os.path.join(databasedir, cmtsource.eventname)

    # Define the directories
    waveforms = os.path.join(outdir, "waveforms")
    stations = os.path.join(outdir, "stations")

    # Only output outdir if wanted
    if get_dirs_only is False:

        # Create maindirectory
        createdir(outdir)

        # WRITESTATUS
        write_status(outdir, "CREATED")

        # Write cmtsolution
        cmtsource.write_CMTSOLUTION_file(
            os.path.join(outdir, 'init_model.cmt'))

        # Write input file
        cmt3d.write_yaml(input_params, os.path.join(outdir, 'input.yml'))

        # Create directories
        createdir(waveforms)
        createdir(stations)

    return outdir, waveforms, stations


# Setup directories
def optimdir(inputfile, cmtfilename, get_dirs_only=False):
    """Sets up source inversion optimization directory

    Parameters
    ----------
    inputfile : str
        location of the input file
    cmtfilename : cmtfilename
        location of original CMTSOLUTION
    get_dirs_only : bool, optional
        Whether to only output the relevant directories, by default False

    Returns
    -------
    _type_
        _description_
    """

    # Read inputfile
    input_params = cmt3d.read_yaml(inputfile)

    # Get database location
    databasedir = input_params["database"]

    # Read CMT file
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get full filename
    outdir = os.path.join(databasedir, cmtsource.eventname)

    # Define the directories
    modldir = os.path.join(outdir, "modl")
    metadir = os.path.join(outdir, "meta")
    measdir = os.path.join(outdir, "meas")
    datadir = os.path.join(outdir, "data")
    simudir = os.path.join(outdir, "simu")
    ssyndir = os.path.join(simudir, "synt")
    sfredir = os.path.join(simudir, "dsdm")
    syntdir = os.path.join(outdir, "synt")
    dsdmdir = os.path.join(outdir, "dsdm")
    costdir = os.path.join(outdir, "cost")
    graddir = os.path.join(outdir, "grad")
    hessdir = os.path.join(outdir, "hess")
    descdir = os.path.join(outdir, "desc")
    optdir = os.path.join(outdir, 'opt')

    # Only output outdir if wanted
    if get_dirs_only is False:

        # Create directories
        createdir(outdir)
        createdir(modldir)
        createdir(metadir)
        createdir(measdir)
        createdir(datadir)
        createdir(ssyndir)
        createdir(sfredir)
        createdir(syntdir)
        createdir(dsdmdir)
        createdir(costdir)
        createdir(graddir)
        createdir(hessdir)
        createdir(descdir)
        createdir(optdir)

    return outdir, modldir, metadir, datadir, simudir, ssyndir, sfredir, \
        syntdir, dsdmdir, costdir, graddir, hessdir, descdir, optdir


def basiccheck(outdir: str):
    """Should be performed after created the inversion dictionary."""

    # Get input params
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get Zerotrace flag
    zero_trace = inputparams["zero_trace"]

    # Mnames
    mnames = read_model_names(outdir)

    # Check Parameter dict for wrong parameters, raises error if not invertible
    check_invertible(mnames)

    # Check whether inversion is a moment tensor inversion
    moment_tensor_inv = check_mt_inv(mnames)

    # Check zero trace condition
    if zero_trace:
        if moment_tensor_inv is False:
            raise ValueError("Can only use Zero Trace condition "
                             "if inverting for Moment Tensor.\n"
                             "Update your parameters.")


def adapt_processdict(cmtsource, processdict, duration):
    """This is a fairly important method because it implements the
        magnitude dependent processing scheme of the Global CMT project.
        Depending on the magnitude, and depth, the methods chooses which
        wavetypes and passbands are going to be used in the inversion.

    Parameters
    ----------
    cmtsource : lwsspy.seismo.cmtsource.CMTSource
        Earthquake solution
    processdict : dict
        process parameter dictionary
    duration : float
        max duration of the seismograms after processing

    Returns
    -------
    dict
        updated processing parameters
    """

    # Get Process parameters
    proc_params = cmt3d.get_process_parameters(cmtsource.moment_magnitude,
                                               cmtsource.depth_in_m)

    # Print the parameters
    pprint(proc_params)

    # Adjust the process dictionary
    for _wave, _process_dict in proc_params.items():

        if _wave in processdict:

            # Adjust weight or drop wave altogether
            if _process_dict['weight'] == 0.0 \
                    or _process_dict['weight'] is None:
                processdict.popitem(_wave)
                continue

            else:
                processdict[_wave]['weight'] = _process_dict["weight"]

            # Adjust pre_filt
            processdict[_wave]['process']['pre_filt'] = \
                [1.0/x for x in _process_dict["filter"]]

            # Adjust trace length depending on the duration
            # given to the class
            processdict[_wave]['process']['relative_endtime'] = \
                _process_dict["relative_endtime"]

            if processdict[_wave]['process']['relative_endtime'] \
                    > duration:
                processdict[_wave]['process']['relative_endtime'] \
                    = duration

            # Adjust windowing config
            for _windict in processdict[_wave]["window"]:
                _windict["config"]["min_period"] = \
                    _process_dict["filter"][3]

                _windict["config"]["max_period"] = \
                    _process_dict["filter"][0]

    # Remove unnecessary wavetypes
    popkeys = []
    for _wave in processdict.keys():
        if _wave not in proc_params:
            popkeys.append(_wave)

    pprint(popkeys)

    for _key in popkeys:
        processdict.pop(_key, None)

    pprint(processdict)
    return processdict


def prepare_inversion_dir(cmtfile, outdir, inputparamfile):

    # Load CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfile)

    # Read parameterfile
    inputparams = cmt3d.read_yaml(inputparamfile)

    # Write the input parameters to the inversion directory
    # (for easy inversion)
    cmt3d.write_yaml(inputparams, os.path.join(outdir, 'input.yml'))

    # start label
    start_label = '_' + inputparams['start_label'] \
        if inputparams['start_label'] is not None else ''

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Get initial processing directory
    if inputparams["processparams"] is None:
        processdict = Constants.processdict
    else:
        processdict = cmt3d.read_yaml(inputparams['processparams'])

    # Adapting the processing dictionary
    processdict = adapt_processdict(cmtsource, processdict, duration)

    # Writing the new processing file to the directory
    cmt3d.write_yaml(processdict, os.path.join(outdir, 'process.yml'))

    # Writing Original CMTSOLUTION
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', cmtsource.eventname + start_label))

    # Write model with generic name for easy access
    cmtsource.write_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', 'init_model.cmt'))


def prepare_simulation_dirs(outdir):

    # Get relevant dirs
    simudir = os.path.join(outdir, 'simu')
    sfredir = os.path.join(simudir, 'dsdm')

    # Get modelparameter names
    model_names = read_model_names(outdir)

    # Create one simulation directory for each inversion parameter
    for _i, _mname in enumerate(model_names):

        if _mname in Constants.nosimpars:
            continue
        else:
            # Create
            pardir = os.path.join(sfredir, f"dsdm{_i:05d}")
            createdir(pardir)


def prepare_model(outdir):

    # Get the initial model
    init_cmt = cmt3d.CMTSource.from_CMTSOLUTION_file(
        os.path.join(outdir, 'meta', 'init_model.cmt'))

    # Read parameterfile
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get the parameters to invert for
    parameters = inputparams['parameters']

    # Get model names
    model_names = list(parameters.keys())

    # Write model names
    write_model_names(model_names, outdir)

    # Get model vector
    model_vector = np.array([getattr(init_cmt, key)
                            for key in parameters.keys()])

    # Write model vector
    write_model(model_vector, outdir, 0, 0)

    # Get scaling
    scaling_vector = np.array([val['scale'] for _, val in parameters.items()])

    # If moment tensor inversion, scale moment tensor elements with the scalar
    # moment
    if check_mt_inv(model_names):

        # Get initial scalar moment
        M0 = init_cmt.M0

        # Update the parameters
        for _i, _name in enumerate(model_names):
            if _name in Constants.mt_params:
                scaling_vector[_i] = M0

    # Write scaling vector
    write_scaling(scaling_vector.astype(float), outdir)

    # Read scaling throws an error if the scaling vector is for some reason of
    # type object
    read_scaling(outdir)

    # Get perturbation
    perturb_vector = np.array([np.nan if val['pert'] is None
                               else float(val['pert'])
                               for _, val in parameters.items()])

    # Write scaling vector
    write_perturbation(perturb_vector, outdir)


def prepare_stations(outdir):

    # Get dir
    metadir = os.path.join(outdir, 'meta')

    # Get CMT
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(os.path.join(
        metadir, 'init_model.cmt'
    ))

    # Eventname
    eventname = cmtsource.eventname

    # Get parameters
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Get data database
    datadatabase = inputparams["datadatabase"]

    # Get datadir in data database
    stationsdir = os.path.join(datadatabase, eventname, 'stations')

    # Read inventory from the station directory and put into
    # a single stations.xml
    inv = cmt3d.read_inventory(os.path.join(stationsdir, '*.xml'))

    # Write inventory to a single station directory
    inv.write(os.path.join(outdir, 'meta', 'stations.xml'),
              format='STATIONXML')


# def prepare_simulation_dirs(outdir):

#     # Get relevant dirs
#     simudir = os.path.join(outdir, 'simu')
#     ssyndir = os.path.join(simudir, 'synt')
#     sfredir = os.path.join(simudir, 'dsdm')

#     # Get input params
#     inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

#     # SPECFEM directory
#     specfemdir = inputparams["specfem"]

#     # Simulation duration
#     simulation_duration = np.round(inputparams["duration"]/60 * 1.02)

#     # Get modelparameter names
#     model_names = read_model_names(outdir)

#     # Stations file
#     stations_src = os.path.join(outdir, 'meta', 'STATIONS.txt')

#     # Create synthetic directories
#     createsimdir(specfemdir, ssyndir,
#                  specfem_dict=Constants.specfem_dict)

#     # Create one simulation directory for each inversion parameter
#     for _i, _mname in enumerate(model_names):

#         if _mname in Constants.nosimpars:
#             continue
#         else:
#             # Create
#             pardir = os.path.join(sfredir, f"dsdm{_i:05d}")
#             createsimdir(specfemdir, pardir,
#                          specfem_dict=Constants.specfem_dict)

#     # Write stations file for the synthetic directory
#     shutil.copyfile(stations_src, os.path.join(ssyndir, "DATA", "STATIONS"))

#     # Update Par_file depending on the parameter.
#     syn_parfile = os.path.join(ssyndir, "DATA", "Par_file")
#     syn_pars = read_parfile(syn_parfile)
#     syn_pars["USE_SOURCE_DERIVATIVE"] = False

#     # Adapt duration
#     syn_pars["RECORD_LENGTH_IN_MINUTES"] = simulation_duration

#     # Write Stuff to Par_file
#     write_parfile(syn_pars, syn_parfile)

#     # Create one simulation directory for each inversion
#     for _i, _mname in enumerate(model_names):

#         # Half duration an time-shift don't need extra simulations
#         if _mname not in Constants.nosimpars:

#             pardir = os.path.join(sfredir, f"dsdm{_i:05d}")

#             # Write stations file
#             # Write stations file for the synthetic directory
#             shutil.copyfile(stations_src, os.path.join(
#                 pardir, "DATA", "STATIONS"))

#             # Update Par_file depending on the parameter.
#             dsdm_parfile = os.path.join(pardir, "DATA", "Par_file")
#             dsdm_pars = read_parfile(dsdm_parfile)

#             # Adapt duration
#             dsdm_pars["RECORD_LENGTH_IN_MINUTES"] = simulation_duration

#             # Check whether parameter is a source location derivative
#             if _mname in Constants.locations:
#                 dsdm_pars["USE_SOURCE_DERIVATIVE"] = True
#                 dsdm_pars["USE_SOURCE_DERIVATIVE_DIRECTION"] = \
#                     Constants.source_derivative[_mname]
#             else:
#                 dsdm_pars["USE_SOURCE_DERIVATIVE"] = False

#             # Write Stuff to Par_file
#             write_parfile(dsdm_pars, dsdm_parfile)


def create_event_dir(cmtfile, inputfile):

    # Get main dir
    out = optimdir(inputfile, cmtfile)
    outdir = out[0]

    # Prep inversion directories
    prepare_inversion_dir(cmtfile, outdir, inputfile)

    # Prepare model
    prepare_model(outdir)


def create_forward_dirs(cmtfile, inputfile):

    # Get main dir
    out = optimdir(inputfile, cmtfile)
    outdir = out[0]

    # Prep inversion directories
    prepare_inversion_dir(cmtfile, outdir, inputfile)

    # Prepare model
    prepare_model(outdir)

    # # Get data
    # stage_data(outdir)
    basiccheck(outdir)

    # Prep Stations
    prepare_stations(outdir)

    # Preparing the simulation directory
    # prepare_simulation_dirs(outdir)

    # Reset iteration counter and linesearch counter
    reset_iter(outdir)
    reset_step(outdir)

    return outdir


def wcreate_forward_dirs(args):
    return create_forward_dirs(*args)


def read_events(eventdir):
    events = []
    for eventfile in os.listdir(eventdir):
        events.append(os.path.join(eventdir, eventfile))
    print(events)
    return events


def parameters2rundir(modelnames, simudir: str):

    # Rundir counter: specfem runs have to be consecutive and start at 1
    runcounter = 1

    # Synth path
    forwardpath = os.path.join(simudir, f"run{runcounter:0>4}")

    # Start dict with synthetics at run 1
    rundict = dict(synt=forwardpath)

    # Loop over the model parameters to find paths and add to dictionary
    for _i, _mname in enumerate(modelnames):

        # Check whether model parameter has to simulated
        if _mname in Constants.nosimpars:
            rundict[_i] = dict(parameter=_mname, rundir=None)

        # If yes give it a rundir
        else:
            runcounter += 1
            frechetpath = os.path.join(simudir, f"run{runcounter:0>4}")
            rundict[_i] = dict(parameter=_mname, rundir=frechetpath)

    return rundict


def reset_cpu_affinity(verbose: bool = False):

    from sys import platform
    if platform == "darwin":
        return

    # Get main process
    p = psutil.Process()

    # Get current affinity
    if verbose:
        print("Current Affinity", p.cpu_affinity())

    # Get all CPUs
    all_cpus = list(range(0, psutil.cpu_count(logical=True), 1))

    # Set new affinity
    p.cpu_affinity(all_cpus)

    # Print new affinity
    if verbose:
        print("New Affinity", p.cpu_affinity())


def isipython():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def cmt3d2gf3d(cmt3d_source: cmt3d.CMTSource) -> CMTSOLUTION:

    # Convert CMT3D source to GF3D source
    cmt = CMTSOLUTION(
        origin_time=cmt3d_source.origin_time,
        pde_lat=cmt3d_source.pde_latitude,
        pde_lon=cmt3d_source.pde_longitude,
        pde_depth=cmt3d_source.pde_depth_in_m/1000.0,
        mb=cmt3d_source.mb,
        ms=cmt3d_source.ms,
        region_tag=cmt3d_source.region_tag,
        eventname=cmt3d_source.eventname,
        time_shift=cmt3d_source.time_shift,
        hdur=cmt3d_source.half_duration,
        latitude=cmt3d_source.latitude,
        longitude=cmt3d_source.longitude,
        depth=cmt3d_source.depth_in_m/1000.0,
        Mrr=cmt3d_source.m_rr,
        Mtt=cmt3d_source.m_tt,
        Mpp=cmt3d_source.m_pp,
        Mrt=cmt3d_source.m_rt,
        Mrp=cmt3d_source.m_rp,
        Mtp=cmt3d_source.m_tp,
    )

    return cmt