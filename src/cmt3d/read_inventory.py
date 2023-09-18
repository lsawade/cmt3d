from typing import Union
from obspy import read_inventory
from obspy import Inventory


def flex_read_inventory(filenames: Union[str, list], **kwargs) -> Inventory:
    """ Takes in a list of strings and tries to read them as inventories
    Creates a single inventory, not an aggregate of inventories

    Parameters
    ----------
    filenames : str or list
        station file(s). wildcards permitted.

    Returns
    -------
    `obspy.Inventory`
        Inventory without duplicates
    """

    if type(filenames) is str:
        filenames = [filenames]

    inv = Inventory()
    for _file in filenames:
        try:
            add_inv = read_inventory(_file, **kwargs)
            for network in add_inv:
                if len(inv.select(network=network.code)) == 0:
                    inv.networks.append(network)
                else:
                    new_network = inv.select(network=network.code)[0]
                    # print(new_network)
                    for station in network:
                        if len(new_network.select(station=station.code)) == 0:
                            new_network.stations.append(station)

                    inv = inv.remove(network=network.code)
                    inv.networks.append(new_network)

        except Exception as e:
            print(f"{_file} could not be read. Error: {e}")

    return inv
