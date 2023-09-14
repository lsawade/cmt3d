
import os
import typing as tp
from lwsspy.seismo.cmt_catalog import CMTCatalog
from lwsspy.utils.io import read_yaml_file, write_yaml_file
from .utils import createdir
from .log import read_status

ResetOpt = tp.Literal["both", "inv", "down"]


def write_event_status(dir, eventname, message):
    with open(os.path.join(dir, eventname), 'w') as f:
        f.truncate(0)
        f.write(message)


def read_event_status(dir, eventname):
    with open(os.path.join(dir, eventname), 'r') as f:
        message = f.read()
    return message


def create_event_status_dir(eventdir, inputfile):

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)
    downdir = os.path.join(neweventdir, "DOWNLOADED")
    initdir = os.path.join(neweventdir, "EVENTS_INIT")
    out_dir = os.path.join(neweventdir, "EVENTS_FINAL")
    statdir = os.path.join(neweventdir, "STATUS")

    # Create dirs if they don't exist
    createdir(downdir)
    createdir(initdir)
    createdir(out_dir)
    createdir(statdir)

    # Write input file to eventdir
    write_yaml_file(inputparams, os.path.join(neweventdir, 'input.yml'))
    
    # Eventfiles
    eventfilelist = [os.path.join(eventdir, i) for i in os.listdir(eventdir)]

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Write CMT's to directory
    cat.cmts2dir(initdir)

    # Check events right after adding them
    check_events(inputfile)


def add_events(eventdir, inputfile):


    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Directory with cmt solutions on file
    initdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Eventfiles
    if os.path.isdir(eventdir):
        eventfilelist = [os.path.join(eventdir, i) for i in os.listdir(eventdir)]
    elif os.path.isfile(eventdir):
        eventfilelist = [eventdir,]

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    for _cmt in cat:

        # Eventname
        cmtname = _cmt.eventname

        # Location in eventstatusdir
        dst = os.path.join(initdir, cmtname)

        # Skip if file already in the events directory
        if os.path.exists(dst):
            continue
        else:
            # Write CMT's to directory
            _cmt.write_CMTSOLUTION_file(dst)

    # Check events right after adding them
    check_events(inputfile)


def check_events(inputfile, resetopt: tp.Optional[ResetOpt] = None):
    """
    Can only be run after the create_event_status_dir. The inputfile
    should be the one in the Event status directory.

    This function is integral to the workflow since it marks which events have 
    to be run and which dont.

    Reset Flag only resets failed jobs. If you want to reset a job that has
    passed. you need to do so at download/inversion directory level by
    changing the content of ``STATUS.txt`` to ``FAIL`` and use the reset flag 
    of this function, or set the content of ``STATUS.txt`` to ``RESET``.
    """

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get database directory
    database = inputparams["database"]

    # Get location of the observed data
    datadatabase = inputparams["datadatabase"]

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Get reset flags
    if resetopt is not None:
        if resetopt == 'both':
            resetdown = True
            resetinv = True
        elif resetopt == 'inv':
            resetdown = False
            resetinv = True
        elif resetopt == 'down':
            resetdown = True
            resetinv = False
    else:
        resetdown = False
        resetinv = False


    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Path to event directory
    eventdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Status dir
    statdir = os.path.join(neweventdir, "STATUS")

    # down dir
    downdir = os.path.join(neweventdir, "DOWNLOADED")

    # Get list of eventfile
    eventfilelist = [os.path.join(eventdir, i) for i in os.listdir(eventdir)]

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Copy event to the init events if it doesn't exist yet
    # Check whether downloaded and if yes, put 1 in to file in
    # DOWNLOADED
    for _cmt in cat:

        # Eventname
        cmtname = _cmt.eventname

        # CMT database directory
        db_cmt = os.path.join(database, cmtname)

        # CMT data directory
        data_cmt = os.path.join(datadatabase, cmtname)

        try:
            # Check download status
            downstat = read_status(data_cmt)

            # Download fails sometimes (or not enough data)
            # Then remove set download flag to fail and tell status that
            # we can't invert
            if 'FAILED' in downstat or 'FAIL' in downstat:
                if resetdown:
                    write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
                    write_event_status(statdir, cmtname, 'CANT')
                else:
                    write_event_status(downdir, cmtname, 'FAIL')
                    write_event_status(statdir, cmtname, 'CANT')
                continue
                
            if 'RESET' in downstat:
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
                write_event_status(statdir, cmtname, 'CANT')
                continue

            # If the download is unfinished set flag to unfinished
            # and inversion flag to cant
            elif (downstat == 'DOWNLOADING'):
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
                write_event_status(statdir, cmtname, 'CANT')
                continue
                
            # If directory created but not yet downloaded, set
            # DL flag to needs downloading, and inversion flag to can't
            elif 'CREATED' in downstat:
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
                write_event_status(statdir, cmtname, 'CANT')
                continue
                
            # If data is donwloaded set downloaded stats to true and move on
            # to check whether we can, or already have inverted
            elif 'DOWNLOADED' in downstat:
                write_event_status(downdir, cmtname, 'TRUE')
            
            # If the event status is undefined set to needs download
            # and inversion flag to cant.
            else:
                write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')    
                write_event_status(statdir, cmtname, 'CANT')
                continue

        # Not created yet
        except Exception:
            write_event_status(downdir, cmtname, 'NEEDS_DOWNLOADING')
            write_event_status(statdir, cmtname, 'CANT')
            continue

        # Inversion status
        try:
            inv_stat = read_status(db_cmt)

            # If finshed status is done
            if "FINISHED" in inv_stat:
                write_event_status(statdir, cmtname, 'DONE')

            # If reset flag is set manually for a specific event
            elif "RESET" in inv_stat:
                write_event_status(statdir, cmtname, 'TODO')

            # If linesearch failed don't add to todo list, unless we reset all
            # events
            elif "FAIL" in inv_stat:
                if resetinv:
                    write_event_status(statdir, cmtname, 'TODO')
                else:
                    write_event_status(statdir, cmtname, 'FAIL')

            # We have certain flags that indicate that an inversion is still
            # running, in that case make the status running
            elif ("SUCCESS" in inv_stat) or ("ADDSTEP" in inv_stat):
                if resetinv:
                    write_event_status(statdir, cmtname, 'TODO')
                else:
                    write_event_status(statdir, cmtname, 'RUNNING')

            # If we flag isn't covered add to todo list
            else:
                write_event_status(statdir, cmtname, 'TODO')

        # Except the case that the status message doesn't exist and add to 
        # todo list
        except Exception:
            write_event_status(statdir, cmtname, 'TODO')


def check_events_todo(inputfile):
    """Can only be run after the ``create_event_status_dir``. The inputfile
    should be the one in the Event status directory."""

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Label for the event status dir
    label = inputparams["solution_label"]
    if label is None:
        label = "new"
    else:
        label = f"{label}"

    # Create event dir
    neweventdir = os.path.join(event_status_dir, label)

    # Path to event directory
    eventdir = os.path.join(neweventdir, "EVENTS_INIT")

    # Status dir
    statdir = os.path.join(neweventdir, "STATUS")

    # Get list of eventfile
    eventfilelist = [os.path.join(eventdir, i) for i in os.listdir(eventdir)]

    # Make catalog
    cat = CMTCatalog.from_file_list(eventfilelist)

    # Copy event to the init events if it doesn't exist yet
    # Check whether downloaded and if yes, put 1 in to file in
    # DOWNLOADED
    TODO = []
    for _i, _cmt in enumerate(cat):

        # Eventname
        cmtname = _cmt.eventname

        # Inversion status
        status = read_event_status(statdir, cmtname)

        # If todo append cmtname
        if status == "TODO":
            TODO.append(os.path.abspath(eventfilelist[_i]))

    return TODO


def check_events_todownload(inputfile):
    """Can only be run after the ``create_event_status_dir``. The inputfile
    should be the one in the Event status directory."""

    # Read input params
    inputparams = read_yaml_file(inputfile)

    # Get eventstatusdir
    event_status_dir = inputparams["event_status"]

    # Get all directories that have events for downloading
    eventstatusdirs = os.listdir(event_status_dir)

    # Check if download dir is in the list and put it last to prioritize 
    # inversion directories for download to do
    if 'download' in eventstatusdirs:
        eventstatusdirs.append(
            eventstatusdirs.pop(
                eventstatusdirs.index('download')
            ))

    # Get full path
    eventstatusdirs = [
        os.path.join(event_status_dir, i) for i in eventstatusdirs
        ]

    # Where to look for downloadable events
    print(eventstatusdirs)

    # Make list of events to be downloaded
    TODOWNLOAD = []

    for _eventstatusdir in eventstatusdirs:

        # Path to event directory
        eventdir = os.path.join(_eventstatusdir, "EVENTS_INIT")

        # Downloaded status
        downdir = os.path.join(_eventstatusdir, "DOWNLOADED")

        # Get list of eventfile
        eventfilelist = [
            os.path.join(eventdir, i) for i in os.listdir(eventdir)]

        # Make catalog
        cat = CMTCatalog.from_file_list(eventfilelist)

        # Check each events downloadstatus
        for _i, _cmt in enumerate(cat):

            # Eventname
            cmtname = _cmt.eventname

            # Inversion status
            status = read_event_status(downdir, cmtname)

            if "TRUE" in status:
                continue
            elif "NEEDS_DOWNLOADING" in status:
                TODOWNLOAD.append(eventfilelist[_i])

    # Filter out repeated events:
    uniquelist = []
    unique_idx = []
    TODOWNLOAD_FILTERED = []
    for _i, _td in enumerate(TODOWNLOAD):
        if os.path.basename(_td) in uniquelist:
            continue
        else:
            uniquelist.append(os.path.basename(_td))
            unique_idx.append(_i)
            TODOWNLOAD_FILTERED.append(_td)

    return TODOWNLOAD_FILTERED
