import sys
import importlib.util


def import_problem(path):

    spec = importlib.util.spec_from_file_location("problem", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module
