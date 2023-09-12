"""

Test suite for the workflow process classifier.

:copyright:
    Lucas Sawade (lsawade@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)

Last Update: April 2020

"""

import os
import sys
import pprint
from lwsspy.gcmt3d.process_classifier import filter_scaling
from lwsspy.gcmt3d.process_classifier import ProcessParams
from numpy import testing as npt


def _upper_level(path, nlevel=4):
    """
    Go the nlevel dir up
    """
    for i in range(nlevel):
        path = os.path.dirname(path)
    return path


# Recreates bash touch behaviour
def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def test_filter_scaling():
    """Tests the filter scaling test. """

    # Magnitudes
    m = 7.5

    # Config values
    startcorners = [125, 150, 300, 350]
    endcorners = [200, 400]
    startmag = 7.0
    endmag = 8.0

    # Compute new corners
    newcorners = filter_scaling(startcorners, startmag, endcorners, endmag, m)

    # Check if correct
    npt.assert_array_almost_equal(newcorners, [162.5, 186.11111111111,
                                               327.77777778, 375.])


def test_classifier_class():
    """This one tests the classifier that takes on an earthquake magnitude
    and depth, and then outputs a dictionary with the necessary processing
    information according to the global CMT processing scheme."""

    # Sample magnitude and depth
    mw = 6.5
    depth = 200000  # in meters

    # Create ProcessParam class
    p = ProcessParams(mw, depth)
    pdict = p.determine_all()

    # Print if errored
    print(6.5)
    pprint.pprint(pdict)

    assert {'body': {'filter': [150.0, 100.0, 60.0, 50.0],
                     'relative_endtime': 3600.0,
                     'weight': 0.3333333333333333,
                     'velocity': False},
            'mantle': {'filter': [350.0, 300.0, 150.0, 125.0],
                       'relative_endtime': 10800.0,
                       'weight': 0.3333333333333333,
                       'velocity': False},
            'surface': {'filter': [150.0, 100.0, 60.0, 50.0],
                        'relative_endtime': 7200.0,
                        'weight': 0.3333333333333333,
                        'velocity': False}
            } == pdict

    # Sample magnitude and depth
    mw = 7.5
    depth = 200000.0  # in meters

    # Create ProcessParam class
    p = ProcessParams(mw, depth)
    pdict = p.determine_all()

    # Print if errored
    print(7.5)
    pprint.pprint(pdict)

    assert {'mantle': {'filter': [375.0, 327.77777777777777,
                                  186.11111111111111, 162.5],
                       'relative_endtime': 10800.0,
                       'weight': 1.0,
                       'velocity': False}
            } == pdict

    # Sample magnitude and depth
    mw = 8.0
    depth = 200000.0  # in meters

    # Create ProcessParam class
    p = ProcessParams(mw, depth)
    pdict = p.determine_all()

    # Print if errored
    print(8.0)
    pprint.pprint(pdict)

    assert {'mantle': {'filter': [400.0, 355.55555555555554,
                                  222.22222222222223, 200.0],
                       'relative_endtime': 10800.0,
                       'weight': 1.0,
                       'velocity': False}
            } == pdict

    # Sample magnitude and depth
    mw = 5.25
    depth = 200000  # in meters

    # Create ProcessParam class
    p = ProcessParams(mw, depth)
    pdict = p.determine_all()

    # Print if errored
    print(5.25)
    pprint.pprint(pdict)

    assert {'body': {'filter': [150.0, 100.0, 50.0, 40.0],
                     'relative_endtime': 3600.0,
                     'weight': 0.5,
                     'velocity': True},
            'surface': {'filter': [150.0, 100.0, 60.0, 50.0],
                        'relative_endtime': 7200.0,
                        'weight': 0.5,
                        'velocity': True}
            } == pdict
