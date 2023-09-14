import pickle as pickle
import yaml


def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj


def read_yaml(filename: str):
    """Read a yaml file"""
    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def write_yaml(data: dict, filename: str):
    """Write a yaml file"""
    with open(filename, 'w') as f:
        yaml.dump(data, f)
