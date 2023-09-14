import os
from copy import deepcopy
import _pickle as pickle
import numpy as np
from lwsspy.seismo.source import CMTSource
from lwsspy.utils.io import read_yaml_file
from lwsspy.geo.azi_weights import azi_weights
from lwsspy.geo.geo_weights import GeoWeights
from .data import read_data_windowed, write_data_windowed


def compute_weights(outdir):
    """Computing the geographical and azimuthal weights."""

    # Get directories
    metadir = os.path.join(outdir, 'meta')

    # Get source
    cmtsource = CMTSource.from_CMTSOLUTION_file(
        os.path.join(metadir, 'init_model.cmt'))

    # Get component weighting
    inputparams = read_yaml_file(os.path.join(outdir, 'input.yml'))
    weights_rtz = inputparams['component_weights']
    max_weight_ratio = inputparams['max_weight_ratio']

    # Get process parameters
    processparams = read_yaml_file(os.path.join(outdir, 'process.yml'))
    wavetypes = processparams.keys()

    # Read the data into a dictionary
    data_dict = dict()
    for _wtype in wavetypes:
        data_dict[_wtype] = read_data_windowed(outdir, wavetype=_wtype)

    # Weight dictionary
    weights = dict()
    weights["event"] = [cmtsource.latitude, cmtsource.longitude]

    waveweightdict = dict()

    for _i, (_wtype, _stream) in enumerate(data_dict.items()):

        # Dictionary to keep track of the sum in each wave type.
        waveweightdict[_wtype] = 0

        # Get wave type weight from process.yml
        weights[_wtype] = dict()
        waveweight = processparams[_wtype]["weight"]
        weights[_wtype]["weight"] = deepcopy(waveweight)

        # Create dict to access traces
        RTZ_traces = dict()
        for _component, _cweight in weights_rtz.items():

            # Copy compnent weight to dictionary
            weights[_wtype][_component] = dict()
            weights[_wtype][_component]["weight"] = deepcopy(_cweight)

            # Create reference
            RTZ_traces[_component] = []

            # Only add ttraces that have windows.
            for _tr in _stream:
                if _tr.stats.component == _component \
                        and len(_tr.stats.windows) > 0:
                    RTZ_traces[_component].append(_tr)

            # Get locations
            latitudes = []
            longitudes = []
            for _tr in RTZ_traces[_component]:
                latitudes.append(_tr.stats.latitude)
                longitudes.append(_tr.stats.longitude)
            latitudes = np.array(latitudes)
            longitudes = np.array(longitudes)

            # Save locations into dict
            weights[_wtype][_component]["lat"] = deepcopy(latitudes)
            weights[_wtype][_component]["lon"] = deepcopy(longitudes)

            # Get azimuthal weights for the traces of each component
            if len(latitudes) > 1 and len(longitudes) > 2:
                aw = azi_weights(
                    cmtsource.latitude,
                    cmtsource.longitude,
                    latitudes, longitudes, nbins=12, p=0.5)

                # Normalize the azimuthal weights
                aw /= np.sum(aw)/len(aw)

                # Save azi weights into dict
                weights[_wtype][_component]["azimuthal"] \
                    = deepcopy(aw)

                # Get Geographical weights
                gw = GeoWeights(latitudes, longitudes)
                _, _, ref, _ = gw.get_condition(ctype='fracmax', param=0.33)
                geo_weights = gw.get_weights(ref)

                # Normalize the azimuthal weights
                geo_weights /= np.sum(geo_weights)/len(geo_weights)

                # Save geo weights into dict
                weights[_wtype][_component]["geographical"] \
                    = deepcopy(geo_weights)

                # Compute Combination weights.
                aweights = (aw * geo_weights)

                # Now fix the maximum ratio between the weights to the
                # max_ratio from the parameter file. For corner cases, this
                # is important because it prevents the contribution of
                # a single station to become too large. [In corner cases
                # this has happened when geo weights and aziweights do the same
                # thing.]
                aweights = (
                    (max_weight_ratio-1) * (aweights - min(aweights))) / \
                    (max(aweights) - min(aweights)) + 1

                # Normalize by the mean
                aweights /= np.sum(aweights)/len(aweights)

                # Add weights
                weights[_wtype][_component]["combination"] = deepcopy(
                    aweights)

            # Figuring out weighting for 2 events does not make sense
            # There is no relative clustering.
            elif len(latitudes) == 2 and len(longitudes) == 2:
                weights[_wtype][_component]["azimuthal"] = [0.5, 0.5]
                weights[_wtype][_component]["geographical"] = [
                    0.5, 0.5]
                weights[_wtype][_component]["combination"] = [
                    0.5, 0.5]
                aweights = [0.5, 0.5]

            elif len(latitudes) == 1 and len(longitudes) == 1:
                weights[_wtype][_component]["azimuthal"] = [1.0]
                weights[_wtype][_component]["geographical"] = [1.0]
                weights[_wtype][_component]["combination"] = [1.0]
                aweights = [1.0]
            else:
                weights[_wtype][_component]["azimuthal"] = []
                weights[_wtype][_component]["geographical"] = []
                weights[_wtype][_component]["combination"] = []
                aweights = []

            # Add weights to traces
            for _tr, _weight in zip(RTZ_traces[_component], aweights):
                _tr.stats.weights = _cweight * _weight
                waveweightdict[_wtype] += np.sum(_cweight * _weight)

    # Normalize by component and aximuthal weights
    for _i, (_wtype, _stream) in enumerate(data_dict.items()):
        # Create dict to access traces
        RTZ_traces = dict()

        for _component, _cweight in weights_rtz.items():
            RTZ_traces[_component] = []
            for _tr in _stream:
                if _tr.stats.component == _component \
                        and "weights" in _tr.stats:
                    RTZ_traces[_component].append(_tr)

            weights[_wtype][_component]["final"] = []
            for _tr in RTZ_traces[_component]:
                _tr.stats.weights /= waveweightdict[_wtype]

                weights[_wtype][_component]["final"].append(
                    deepcopy(_tr.stats.weights))

    # Write weights to meta directory
    with open(os.path.join(metadir, "weights.pkl"), "wb") as f:
        pickle.dump(deepcopy(weights), f)

    # Write windowed and weighted stream
    for _wtype in wavetypes:
        write_data_windowed(data_dict[_wtype], outdir, wavetype=_wtype)
