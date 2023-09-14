import os
import numpy as np
from distutils.dir_util import copy_tree
from lwsspy.utils.io import read_yaml_file
from lwsspy.seismo.source import CMTSource
from lwsspy.seismo.download_waveforms_to_storage import \
    download_waveforms_to_storage

from .constants import Constants
from .log import write_status


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


# % Get Model CMT
def get_data(outdir: str):

    # Load CMT solution
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(outdir, 'init_model.cmt'))

    # Read input param file
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

    # Get duration from the parameter file
    duration = inputparams['duration']

    # Define the waveform and station directory and the
    waveformdir = os.path.join(outdir, "waveforms")
    stationdir = os.path.join(outdir, "stations")

    # Download Data Params
    if inputparams["downloadparams"] is None:
        download_dict = Constants.download_dict
    else:
        download_dict = read_yaml_file(inputparams["downloadparams"])

    # Start and End time of the download
    starttime_offset = inputparams["starttime_offset"]
    endtime_offset = inputparams["endtime_offset"]
    starttime = cmtsource.cmt_time + starttime_offset
    endtime = cmtsource.cmt_time + duration + endtime_offset

    # WRITESTATUS
    write_status(outdir, "DOWNLOADING")

    # Redirect logger to file
    download_waveforms_to_storage(
        outdir, starttime=starttime, endtime=endtime,
        waveform_storage=waveformdir, station_storage=stationdir,
        logfile=os.path.join(outdir, 'download-log.txt'),
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
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Read input param file
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))

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
