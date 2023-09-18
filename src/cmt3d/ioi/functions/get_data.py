import os
import numpy as np
from distutils.dir_util import copy_tree
import cmt3d
from .constants import Constants
from .log import write_status
from .utils import downloaddir


def cpdir(src, dst):
    """Copies entire directory from src to dst

    Parameters
    ----------
    src : str
        Source directory
    dst : str
        Destination directory
    """
    copy_tree(src, dst)


def get_data(outdir: str):

    # Get cmt file
    cmtfilename = os.path.join(outdir, 'meta', 'init_model.cmt')

    # Load CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfilename)

    # Get input file
    inputfile = os.path.join(outdir, 'input.yml')

    # Read input param file
    inputparams = cmt3d.read_yaml(inputfile)

    # Get duration from the parameter file
    downdir, waveformdir, stationdir = downloaddir(inputfile, cmtfilename,
                                                   get_dirs_only=True)

    # Check whether data directory exists
    if not os.path.exists(downdir):
        os.makedirs(downdir)

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Download Data Params
    if inputparams["downloadparams"] is None:
        download_dict = Constants.download_dict
    else:
        download_dict = cmt3d.read_yaml(inputparams["downloadparams"])

    # Start and End time of the download
    starttime_offset = inputparams["starttime_offset"]
    endtime_offset = inputparams["endtime_offset"]
    starttime = cmtsource.origin_time + starttime_offset
    endtime = cmtsource.origin_time + duration + endtime_offset

    # WRITESTATUS
    write_status(downdir, "DOWNLOADING")

    # Redirect logger to file
    cmt3d.download_waveforms_to_storage(
        downdir, starttime=starttime, endtime=endtime,
        waveform_storage=waveformdir, station_storage=stationdir,
        logfile=os.path.join(downdir, 'download-log.txt'),
        download_chunk_size_in_mb=100, threads_per_client=1,
        **download_dict)

    # Check whether download can be called successful
    if (len(os.listdir(waveformdir)) <= 30) \
            or (len(os.listdir(stationdir)) <= 10):
        write_status(outdir, "FAILED")
    else:
        write_status(outdir, "DOWNLOADED")

    return None


def stage_data(outdir: str):

    # Final location
    metadir = os.path.join(outdir, 'meta')
    datadir = os.path.join(outdir, 'data')

    # Load CMT solution
    cmtsource = cmt3d.CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Read input param file
    inputparams = cmt3d.read_yaml(os.path.join(outdir, 'input.yml'))

    # Databases
    src_database = inputparams["datadatabase"]
    src_cmtdir = os.path.join(src_database, cmtsource.eventname)

    # Waveformdirs
    src_waveforms = os.path.join(src_cmtdir, 'waveforms')
    dst_waveforms = os.path.join(datadir, 'waveforms')

    # Metadata
    src_stations = os.path.join(src_cmtdir, 'stations')
    dst_stations = os.path.join(metadir, 'stations')

    # Copy Waveforms
    cpdir(src_waveforms, dst_waveforms)
    cpdir(src_stations, dst_stations)



def bin():

    import sys
    outdir = sys.argv[1]

    get_data(outdir)
