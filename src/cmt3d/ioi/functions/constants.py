import os
from lwsspy.utils.io import read_yaml_file

scriptdir = os.path.dirname(os.path.abspath(__file__))
paramsdir = os.path.dirname(scriptdir)

class Constants:

    # Locations
    # ---------
    processdict = read_yaml_file(os.path.join(paramsdir, "process.yml"))
    inputfilename = os.path.join(os.path.join(paramsdir, "input.yml"))

    # Download dictionary
    download_dict = dict(
        network=",".join(['CU', 'G', 'GE', 'IC', 'II', 'IU', 'MN']),
        channel_priorities=["LH*", "BH*"],
    )

    # Specfem directory
    specfem_dict = {
        "bin": "link",
        "DATA": {
            "Par_file": "file",
            # "STATIONS": "file",
        },
        "DATABASES_MPI": "link",
        "OUTPUT_FILES": {
            'values_from_mesher.h': "file",
            'addressing.txt': "file",
            'gpu_device_info.txt': "file",
            'gpu_device_mem_usage.txt': "file",
            'RECORDHEADERS': "file",
        }
    }

    # Parameter lists:
    # ----------------
    # This parameters we know the frechet derivative computation of
    parameter_check_list = ['depth_in_m', "time_shift", 'latitude', 'longitude',
                            "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    # Parameters that do not need simulation
    nosimpars = ["time_shift", "half_duration"]

    # Parameters related to the hypocenter
    hypo_pars = ['depth_in_m', "time_shift", 'latitude', 'longitude']

    # Parameters that are related to the moment tensor
    mt_params = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    # Locations
    locations = ["latitude", "longitude", "depth_in_m"]

    # Source derivative for SPECFEM (hard-coded)
    source_derivative = dict(latitude=2, longitude=3, depth_in_m=1)
