#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pycmt3d tests suite.
Run with pytest.
:copyright:
    Wenjie Lei (lei@princeton.edu)
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from __future__ import print_function, division
import inspect
import os
import obspy
import cmt3d
import pytest
import tempfile
import numpy as np
from obspy import read_events

# Most generic way to get the data folder path.
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CMTFILE = os.path.join(DATA_DIR, "CMTSOLUTION")


@pytest.fixture
def cmt():
    return cmt3d.CMTSource.from_CMTSOLUTION_file(CMTFILE)


@pytest.fixture
def event():
    return read_events(CMTFILE)[0]


def test_from_CMTSOLUTION_file(cmt):
    origin_time = obspy.UTCDateTime(2001, 9, 9, 23, 59, 17.78)
    cmt_time = origin_time + 2.0
    cmt_true = \
        cmt3d.CMTSource(origin_time=origin_time,
                  pde_latitude=34.0745, pde_longitude=-118.3792,
                  pde_depth_in_m=6400, mb=4.2, ms=4.2, region_tag="Hollywood",
                  eventname="9703873", cmt_time=cmt_time, half_duration=1.0,
                  latitude=34.1745, longitude=-118.4792, depth_in_m=5400.0,
                  m_rr=1.0e22, m_tt=-1.0e22)

    assert cmt == cmt_true


def test_from_event_file(event):
    origin_time = obspy.UTCDateTime(2001, 9, 9, 23, 59, 17.78)
    cmt_time = origin_time + 2.0
    cmt_true = \
        cmt3d.CMTSource(origin_time=origin_time,
                  pde_latitude=34.0745, pde_longitude=-118.3792,
                  pde_depth_in_m=6400, mb=4.2, ms=4.2, region_tag="SOUTHERN CALIFORNIA",
                  eventname="9703873", cmt_time=cmt_time, half_duration=1.0,
                  latitude=34.1745, longitude=-118.4792, depth_in_m=5400.0,
                  m_rr=1.0e22, m_tt=-1.0e22)
    cmt = cmt3d.CMTSource.from_event(event)
    print(cmt_true)
    print(cmt)
    assert cmt == cmt_true


def test_write_CMTSOLUTION_File(tmpdir, cmt):
    fn = os.path.join(str(tmpdir), "CMTSOLUTION.temp")
    cmt.write_CMTSOLUTION_file(fn)
    new_cmt = cmt3d.CMTSource.from_CMTSOLUTION_file(fn)
    assert new_cmt == cmt


def test_load_quakeML():
    """Test the create directory method"""
    # Check one cmt file
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Cmtfile path
        cmtfile = os.path.join(DATA_DIR, "testCMT")

        # create new directory
        new_xml_path = os.path.join(tmp_dir, "tests.xml")
        xml = read_events(cmtfile)
        xml.write(new_xml_path, format="QUAKEML")

        assert(os.path.exists(new_xml_path)
               and os.path.isfile(new_xml_path))

        print("QuakeML\n", cmt3d.CMTSource.from_quakeml_file(new_xml_path))
        print("CMT\n", cmt3d.CMTSource.from_CMTSOLUTION_file(cmtfile))
        # assertDictAlmostEqual(CMTSource.from_quakeml_file(new_xml_path),
        #                       CMTSource.from_CMTSOLUTION_file(cmtfile))


def test_M0_getter_setter(cmt):

    paramlist = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]

    # Old M0
    M0 = cmt.M0
    check_params = [getattr(cmt, _par) for _par in paramlist]
    print(check_params)

    # M0
    nM0 = 2e+25

    # Factor
    factor = nM0/M0

    # Set M0
    cmt.M0 = nM0

    # Check whether new factor on tensor component matches the M0 factor
    for _i, _par in enumerate(paramlist):
        print(getattr(cmt, _par), check_params[_i])
        if check_params[_i] != 0:
            assert factor == np.abs(getattr(cmt, _par)/check_params[_i])


def test_M0_update_hdur(cmt):

    # Half duration factor
    M0_0 = 2.19370668048397e+27
    hdur_0 = 13.6
    M0_1 = 1.710498757672744e+25
    hdur_1 = 2.7

    # compute actual half duration
    cmt.M0 = M0_0
    np.testing.assert_almost_equal(hdur_0, cmt.half_duration)

    cmt.M0 = M0_1
    np.testing.assert_almost_equal(hdur_1, cmt.half_duration)


def test_decomposition(cmt):

    # Get eigenvalues and vectors for testing
    (M1, M2, M3), ev = cmt.tnp
    print(M1, M2, M3)

    # Get decomposition
    Miso, Mclvd, Mdc = cmt.decomp("iso_clvd_dc")
    print(Miso, Mdc, Mclvd)

    # As defined in Vavryčuk 2015
    eiso = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    edc = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

    if M1+M3-2 * M2 < 0:
        eclvd = 0.5*np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        eclvd = 0.5*np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])

    # Reconstruct moment tensor in eigen basis
    M = Miso * eiso + Mdc * edc + Mclvd * eclvd

    # Compute original tensor using the eigenvectors
    tensor = np.linalg.inv(ev) @ M @ ev

    # Test whether they match.
    np.testing.assert_array_almost_equal(tensor, cmt.fulltensor)
