#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Source and Receiver classes of Instaseis.

:copyright:
    Lion Krischer (krischer@geophysik.uni-muenchen.de), 2014
    Martin van Driel (Martin@vanDriel.de), 2014
    Wenjie Lei (lei@princeton.edu), 2016
    Lucas Sawade (lsawade@princeton.edu), 2019

:license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lgpl.html)

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import io
import numpy as np
import typing as tp
from obspy import UTCDateTime, read_events
from obspy.core.event import Event
import warnings
from . import sourcedecomposition
from inspect import getmembers, isfunction
from .eqpar import eqpar
import matplotlib.pyplot as plt


class CMTSource(object):
    """
    Class to handle a seismic moment tensor source including a source time
    function.
    """

    def __init__(
        self,
        origin_time: UTCDateTime = UTCDateTime(0),
        pde_latitude=0.0,
        pde_longitude=0.0,
        mb=0.0,
        ms=0.0,
        pde_depth_in_m=0.0,
        region_tag="",
        eventname="",
        cmt_time: tp.Union[UTCDateTime, float] = UTCDateTime(0),
        half_duration=0.0,
        latitude=0.0,
        longitude=0.0,
        depth_in_m=0.0,
        m_rr=0.0,
        m_tt=0.0,
        m_pp=0.0,
        m_rt=0.0,
        m_rp=0.0,
        m_tp=0.0,
    ):
        """
        :param latitude: latitude of the source in degree
        :param longitude: longitude of the source in degree
        :param depth_in_m: source depth in m
        :param m_rr: moment tensor components in r, theta, phi in Nm
        :param m_tt: moment tensor components in r, theta, phi in Nm
        :param m_pp: moment tensor components in r, theta, phi in Nm
        :param m_rt: moment tensor components in r, theta, phi in Nm
        :param m_rp: moment tensor components in r, theta, phi in Nm
        :param m_tp: moment tensor components in r, theta, phi in Nm
        :param time_shift: correction of the origin time in seconds. only
            useful in the context of finite sources
        :param sliprate: normalized source time function (sliprate)
        :param dt: sampling of the source time function
        :param origin_time: The origin time of the source. This will be the
            time of the first sample in the final seismogram. Be careful to
            adjust it for any time shift or STF (de)convolution effects.
        """
        self.origin_time = origin_time
        self.pde_latitude = pde_latitude
        self.pde_longitude = pde_longitude
        self.pde_depth_in_m = pde_depth_in_m
        self.mb = mb
        self.ms = ms
        self.region_tag = region_tag
        self.eventname = eventname
        self.cmt_time = cmt_time
        self.half_duration = half_duration
        self.latitude = latitude
        self.longitude = longitude
        self.depth_in_m = depth_in_m
        self.m_rr = m_rr
        self.m_tt = m_tt
        self.m_pp = m_pp
        self.m_rt = m_rt
        self.m_rp = m_rp
        self.m_tp = m_tp

    @classmethod
    def from_CMTSOLUTION_file(cls, filename):
        """
        Initialize a source object from a CMTSOLUTION file.
        :param filename: path to the CMTSOLUTION file
        """

        with open(filename, "rt") as f:
            line = f.readline()
            origin_time = line[5:].strip().split()[:6]
            values = list(map(int, origin_time[:-1])) + [float(origin_time[-1])]
            try:
                origin_time = UTCDateTime(*values)
            except (TypeError, ValueError):
                warnings.warn("Could not determine origin time from line: %s" % line)
                origin_time = UTCDateTime(0)
            otherinfo = line[4:].strip().split()[6:]
            pde_lat = float(otherinfo[0])
            pde_lon = float(otherinfo[1])
            pde_depth_in_m = float(otherinfo[2]) * 1e3
            mb = float(otherinfo[3])
            ms = float(otherinfo[4])
            region_tag = " ".join(otherinfo[5:])

            eventname = f.readline().strip().split()[-1]
            time_shift = float(f.readline().strip().split()[-1])
            cmt_time = origin_time + time_shift
            half_duration = float(f.readline().strip().split()[-1])
            latitude = float(f.readline().strip().split()[-1])
            longitude = float(f.readline().strip().split()[-1])
            depth_in_m = float(f.readline().strip().split()[-1]) * 1e3

            #                                                 unit: N/m
            m_rr = float(f.readline().strip().split()[-1])  # / 1e7
            m_tt = float(f.readline().strip().split()[-1])  # / 1e7
            m_pp = float(f.readline().strip().split()[-1])  # / 1e7
            m_rt = float(f.readline().strip().split()[-1])  # / 1e7
            m_rp = float(f.readline().strip().split()[-1])  # / 1e7
            m_tp = float(f.readline().strip().split()[-1])  # / 1e7

        return cls(
            origin_time=origin_time,
            pde_latitude=pde_lat,
            pde_longitude=pde_lon,
            mb=mb,
            ms=ms,
            pde_depth_in_m=pde_depth_in_m,
            region_tag=region_tag,
            eventname=eventname,
            cmt_time=cmt_time,
            half_duration=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth_in_m=depth_in_m,
            m_rr=m_rr,
            m_tt=m_tt,
            m_pp=m_pp,
            m_rt=m_rt,
            m_rp=m_rp,
            m_tp=m_tp,
        )

    @classmethod
    def from_quakeml_file(cls, filename: str):
        """
        Initialiaze a source object from a quakeml file
        :param filename: path to a quakeml file
        """
        cat = read_events(filename)
        event = cat[0]

        return cls.from_event(event)

    @classmethod
    def from_sdr(cls, s, d, r, M0=1.0, **kwargs):
        """definition from Stein and Wysession
        s is the strike (phi_f), d is the dip (delta), and r is the slip angle
        (lambda). IM ASSIGNING THE COMPONENTS WRONG. SEE EMAIL TO YURI!!!!!

        Note How in Stein and Wysession the z-axis points UP (Figure 4.2-2) and
        in Aki and Richards the z-axis points down (Figure 4.20).
        This results in a sign changes for the fault normal and slip vector.

        Convention for the code here is
        X-North, Y-East, Z-Down
        R-Down, Theta-North, Phi-East


        """

        # To radians
        s = np.radians(s)
        d = np.radians(d)
        r = np.radians(r)

        # Fault normal
        n = np.array([-np.sin(d) * np.sin(s), +np.sin(d) * np.cos(s), -np.cos(d)])

        # Slip vector
        d = np.array(
            [
                np.cos(r) * np.cos(s) + np.cos(d) * np.sin(r) * np.sin(s),
                np.cos(r) * np.sin(s) - np.cos(d) * np.sin(r) * np.cos(s),
                -np.sin(r) * np.sin(d),
            ]
        )

        # Compute moment tensor Stein and Wysession Style
        Mx = M0 * (np.outer(n, d) + np.outer(d, n))

        # # Full terms from AKI & Richards (Box 4.4)
        # Mxx = -M0 * (np.sin(d) * np.cos(r)
        #   * np.sin(2*s) + np.sin(2*d) * np.sin(r) * np.sin(s)**2)
        # Mxy = +M0 * (np.sin(d) * np.cos(r)
        #   * np.cos(2*s) + 0.5 * np.sin(2*d) * np.sin(r) * np.sin(2*s))
        # Mxz = -M0 * (np.cos(d) * np.cos(r)
        #   * np.cos(s) + np.cos(2*d) * np.sin(r) * np.sin(s))
        # Myy = +M0 * (np.sin(d) * np.cos(r)
        #   * np.sin(2*s) - np.sin(2*d) * np.sin(r) * np.cos(s)**2)
        # Myz = -M0 * (np.cos(d) * np.cos(r)
        #   * np.sin(s) - np.cos(2*d) * np.sin(r) * np.cos(s))
        # Mzz = +M0 * np.sin(2*d) * np.sin(r)

        # # Moment tensor creation
        # Mx = np.array( [
        #     [Mxx, Mxy, Mxz],
        #     [Mxy, Myy, Myz],
        #     [Mxz, Myz, Mzz]
        # ])

        Mr = np.array(
            [
                [Mx[2, 2], Mx[2, 0], -Mx[2, 1]],
                [Mx[0, 2], Mx[0, 0], -Mx[0, 1]],
                [-Mx[1, 2], -Mx[1, 0], Mx[1, 1]],
            ]
        )

        return cls(
            m_rr=Mr[0, 0],
            m_tt=Mr[1, 1],
            m_pp=Mr[2, 2],
            m_rt=Mr[0, 1],
            m_rp=Mr[0, 2],
            m_tp=Mr[1, 2],
        )

    @classmethod
    def from_event(cls, event: Event):

        for origin in event.origins:
            if origin.origin_type == "centroid":
                cmtsolution = origin
            else:
                pdesolution = origin

        origin_time = pdesolution.time
        pde_lat = pdesolution.latitude
        pde_lon = pdesolution.longitude
        pde_depth_in_m = pdesolution.depth
        mb = 0.0
        ms = 0.0
        for mag in event.magnitudes:
            if mag.magnitude_type == "Mb":
                mb = mag.mag
            elif mag.magnitude_type == "MS":
                ms = mag.mag
        # Get region tag
        try:
            region_tag = cmtsolution.region
        except Exception:
            try:
                region_tag = cmtsolution.region
            except Exception:
                warnings.warn("Region tag not found.")

        for descrip in event.event_descriptions:
            if descrip.type == "earthquake name":
                eventname = descrip.text
            else:
                eventname = ""
        cmt_time = cmtsolution.time
        focal_mechanism = event.focal_mechanisms[0]
        half_duration = (
            focal_mechanism.moment_tensor.source_time_function.duration / 2.0
        )
        latitude = cmtsolution.latitude
        longitude = cmtsolution.longitude
        depth_in_m = cmtsolution.depth
        tensor = focal_mechanism.moment_tensor.tensor
        m_rr = tensor.m_rr * 1e7
        m_tt = tensor.m_tt * 1e7
        m_pp = tensor.m_pp * 1e7
        m_rt = tensor.m_rt * 1e7
        m_rp = tensor.m_rp * 1e7
        m_tp = tensor.m_tp * 1e7

        return cls(
            origin_time=origin_time,
            pde_latitude=pde_lat,
            pde_longitude=pde_lon,
            mb=mb,
            ms=ms,
            pde_depth_in_m=pde_depth_in_m,
            region_tag=region_tag,
            eventname=eventname,
            cmt_time=cmt_time,
            half_duration=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth_in_m=depth_in_m,
            m_rr=m_rr,
            m_tt=m_tt,
            m_pp=m_pp,
            m_rt=m_rt,
            m_rp=m_rp,
            m_tp=m_tp,
        )

    def to_event(self) -> Event:
        """returns obspy event"""
        with io.BytesIO() as f:
            f.write(self.__str__().encode("utf-8"))
            f.seek(0)
            ev = read_events(f)[0]
        return ev

    @classmethod
    def from_dictionary(cls, d):
        """
        Initialize a source object from a CMTSOLUTION file.

        :param dictionary: dictionary
        """

        origin_time = UTCDateTime(d["origin_time"][3:])
        pde_lat = d["pde_latitude"]
        pde_lon = d["pde_longitude"]
        pde_depth_in_m = d["pde_depth_in_m"]
        mb = d["mb"]
        ms = d["ms"]
        region_tag = d["region_tag"]

        eventname = d["eventname"]
        cmt_time = UTCDateTime(d["cmt_time"][3:])
        half_duration = d["half_duration"]
        latitude = d["latitude"]
        longitude = d["longitude"]
        depth_in_m = d["depth_in_m"]

        # unit: N/m
        m_rr = d["m_rr"]
        m_tt = d["m_tt"]
        m_pp = d["m_pp"]
        m_rt = d["m_rt"]
        m_rp = d["m_rp"]
        m_tp = d["m_tp"]

        return cls(
            origin_time=origin_time,
            pde_latitude=pde_lat,
            pde_longitude=pde_lon,
            mb=mb,
            ms=ms,
            pde_depth_in_m=pde_depth_in_m,
            region_tag=region_tag,
            eventname=eventname,
            cmt_time=cmt_time,
            half_duration=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth_in_m=depth_in_m,
            m_rr=m_rr,
            m_tt=m_tt,
            m_pp=m_pp,
            m_rt=m_rt,
            m_rp=m_rp,
            m_tp=m_tp,
        )

    def to_row(self):
        """Returns a tuple of all parameters to append it to a
        dataframe."""
        return (
            self.origin_time.datetime,
            self.pde_latitude,
            self.pde_longitude,
            self.pde_depth_in_m,
            self.mb,
            self.ms,
            self.region_tag,
            self.eventname,
            self.time_shift,
            self.half_duration,
            self.latitude,
            self.longitude,
            self.depth_in_m,
            self.m_rr,
            self.m_tt,
            self.m_pp,
            self.m_rt,
            self.m_rp,
            self.m_tp,
        )

    def write_CMTSOLUTION_file(self, filename, mode="w"):
        """
        Initialize a source object from a CMTSOLUTION file.
        :param filename: path to the CMTSOLUTION file
        """
        with open(filename, mode) as f:
            # Reconstruct the first line as well as possible. All
            # hypocentral information is missing.
            f.write(
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f"
                " %s\n"
                % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second + self.origin_time.microsecond / 1e6,
                    self.pde_latitude,
                    self.pde_longitude,
                    self.pde_depth_in_m / 1e3,
                    self.mb,
                    self.ms,
                    str(self.region_tag),
                )
            )
            f.write("event name:     %s\n" % (str(self.eventname)))
            f.write("time shift:%12.4f\n" % (self.time_shift,))
            f.write("half duration:%9.4f\n" % (self.half_duration,))
            f.write("latitude:%14.4f\n" % (self.latitude,))
            f.write("longitude:%13.4f\n" % (self.longitude,))
            f.write("depth:%17.4f\n" % (self.depth_in_m / 1e3,))
            f.write("Mrr:%19.6e\n" % self.m_rr)  # * 1e7,))
            f.write("Mtt:%19.6e\n" % self.m_tt)  # * 1e7,))
            f.write("Mpp:%19.6e\n" % self.m_pp)  # * 1e7,))
            f.write("Mrt:%19.6e\n" % self.m_rt)  # * 1e7,))
            f.write("Mrp:%19.6e\n" % self.m_rp)  # * 1e7,))
            f.write("Mtp:%19.6e\n" % self.m_tp)  # * 1e7,))

    @property
    def M0(self):
        """
        Scalar Moment M0 in Nm
        """
        return (
            self.m_rr**2
            + self.m_tt**2
            + self.m_pp**2
            + 2 * self.m_rt**2
            + 2 * self.m_rp**2
            + 2 * self.m_tp**2
        ) ** 0.5 * 0.5**0.5

    @M0.setter
    def M0(self, M0):
        """
        Scalar Moment M0 in Nm
        """
        iM0 = self.M0
        fM0 = M0
        factor = fM0 / iM0
        self.m_rr *= factor
        self.m_tt *= factor
        self.m_pp *= factor
        self.m_rt *= factor
        self.m_rp *= factor
        self.m_tp *= factor

        self.update_hdur()

    @property
    def moment_magnitude(self):
        """
        Moment magnitude M_w
        """
        # =  (log Mo - 9.1) / 1.5 = (2/3) * (log Mo - 9.1)
        return 2 / 3 * np.log10(7 + self.M0) - 10.73

    @property
    def time_shift(self):
        """
        Time shift between cmtsolution and pdesolution
        """
        return self.cmt_time - self.origin_time

    @time_shift.setter
    def time_shift(self, time_shift):
        self.cmt_time = self.origin_time + time_shift

    @property
    def tensor(self):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        return np.array(
            [self.m_rr, self.m_tt, self.m_pp, self.m_rt, self.m_rp, self.m_tp]
        )

    @tensor.setter
    def tensor(self, tensor):
        """
        List of moment tensor components in r, theta, phi coordinates:
        [m_rr, m_tt, m_pp, m_rt, m_rp, m_tp]
        """
        self.m_rr = tensor[0]
        self.m_tt = tensor[1]
        self.m_pp = tensor[2]
        self.m_rt = tensor[3]
        self.m_rp = tensor[4]
        self.m_tp = tensor[5]

        # self.update_hdur()

    @property
    def fulltensor(self):
        """
        ndarray of full moment tensor components in r, theta, phi coordinates:
        """
        return np.array(
            [
                [self.m_rr, self.m_rt, self.m_rp],
                [self.m_rt, self.m_tt, self.m_tp],
                [self.m_rp, self.m_tp, self.m_pp],
            ]
        )

    @fulltensor.setter
    def fulltensor(self, tensor):
        """
        ndarray of full moment tensor components in r, theta, phi coordinates:
        """
        self.m_rr = tensor[0, 0]
        self.m_tt = tensor[1, 1]
        self.m_pp = tensor[2, 2]
        self.m_rt = tensor[0, 1]
        self.m_rp = tensor[0, 2]
        self.m_tp = tensor[1, 2]

        self.update_hdur()

    def update_hdur(self):
        # Updates the half duration
        Nm_conv = 1 / 1e7
        self.half_duration = np.round(
            2.26 * 10 ** (-6) * (self.M0 * Nm_conv) ** (1 / 3), decimals=1
        )

    @property
    def pbt(self):
        """Returns tension (t), null (b), and pressure (p) axis and
        corresponding eigenvalues.

        Returns
        -------
        tuple
            matrix with corresponding eigenvalues, tbp column vectors
        """
        # Get eigenvalues and eigenvectors
        (lb, ev) = np.linalg.eigh(self.fulltensor)

        order = lb.argsort()  # in decreasing order -> pbt

        return lb[order], ev[order]

    @property
    def tnp(self):
        ep = self.eqpar
        return ep[4], ep[5]

    @property
    def eqpar(self):
        """For documentation see lwsspy.seismo.eqpar."""
        return list(eqpar(self.tensor))

    @property
    def sdr(self):
        """
        Returns
        -------
        (strikes, dips, rakes)
        """
        return self.eqpar[1:4]

    @property
    def tbp(self):
        """
        Returns
        -------
        (t[pl, az], b[pl, az], p[pl, az])
        """
        _, _, _, _, _, _, plungs, azims = self.eqpar
        return [[plungs[i], azims[i]] for i in range(3)]

    @staticmethod
    def normal2sd(normal):
        """Compute strike and dip from normal

        Parameters
        ----------
        normal : [type]
            [description]
        """

        # strike
        strike = np.arctan2(normal[1], -normal[2])
        # strike = np.mod(strike, 2*np.pi)

        # dip
        dip = np.arctan2(
            (normal[1] ** 2 + normal[0] ** 2),
            np.sqrt((normal[0] * normal[2]) ** 2 + (normal[1] * normal[2]) ** 2),
        )
        # dip = np.arccos(normal[2]/np.sqrt(np.sum(normal**2)))

        return strike, dip

    def decomp(self, dtype="eps_nu"):
        """Returns decomposition based on eignevalues of the moment tensor

        Parameters
        ----------
        dtype : str, optional
            type of decomposition. implement new decomposition by adding
            function to sourcedecomposition module, by default "eps_nu"

        Returns
        -------
        arraylike
            decomposed source, output depends on function output.

        Raises
        ------
        ValueError
            If dtype is not implemented an error will be raised.
        """

        # Get possible decompositions
        dtypes = [func for func, _ in getmembers(sourcedecomposition, isfunction)]

        if dtype not in dtypes:
            raise ValueError(f"{dtype} not implemented. Possible dtypes are: {dtypes}")

        # Get function from the module
        decompfunc = getattr(sourcedecomposition, dtype)
        (M1, M2, M3), _ = self.tnp  # self.pbt_old

        return decompfunc(M1, M2, M3)

    def __str__(self):

        # Reconstruct the first line as well as possible. All
        # hypocentral information is missing.
        if isinstance(self.origin_time, UTCDateTime):
            return_str = (
                " PDE %4i %2i %2i %2i %2i %5.2f %8.4f %9.4f %5.1f %.1f %.1f"
                " %s\n"
                % (
                    self.origin_time.year,
                    self.origin_time.month,
                    self.origin_time.day,
                    self.origin_time.hour,
                    self.origin_time.minute,
                    self.origin_time.second + self.origin_time.microsecond / 1e6,
                    self.pde_latitude,
                    self.pde_longitude,
                    self.pde_depth_in_m / 1e3,
                    self.mb,
                    self.ms,
                    self.region_tag,
                )
            )
        else:
            return_str = "----- CMT Delta: ------- \n"

        return_str += "event name:  %10s\n" % (str(self.eventname),)
        return_str += "time shift:%12.4f\n" % (self.time_shift,)
        return_str += "half duration:%9.4f\n" % (self.half_duration,)
        return_str += "latitude:%14.4f\n" % (self.latitude,)
        return_str += "longitude:%13.4f\n" % (self.longitude,)
        return_str += "depth:%17.4f\n" % (self.depth_in_m / 1e3,)
        return_str += "Mrr:%19.6e\n" % self.m_rr  # * 1e7,))
        return_str += "Mtt:%19.6e\n" % self.m_tt  # * 1e7,))
        return_str += "Mpp:%19.6e\n" % self.m_pp  # * 1e7,))
        return_str += "Mrt:%19.6e\n" % self.m_rt  # * 1e7,))
        return_str += "Mrp:%19.6e\n" % self.m_rp  # * 1e7,))
        return_str += "Mtp:%19.6e\n" % self.m_tp  # * 1e7,))

        return return_str

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        """Making the class iterable through key,value pairs."""
        # first start by grabbing the Class items
        iters = self.__dict__.items()

        # now 'yield' through the items
        for x, y in iters:
            yield x

    def __getitem__(self, item):
        """Making the CMT Source subscriptable with indeces."""
        return self.__dict__[item]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return self.__dict__ != other.__dict__

    @staticmethod
    def same_eventids(id1, id2):

        id1 = id1 if not id1[0].isalpha() else id1[1:]
        id2 = id2 if not id2[0].isalpha() else id2[1:]

        return id1 == id2

    def __sub__(self, other):
        """USE WITH CAUTION!!
        -> Origin time becomes float of delta t
        -> centroid time becomes float of delta t
        -> half duration is weird to compare like this as well.
        -> the other class will be subtracted from this one and the resulting
           instance will keep the eventname and the region tag from this class
        """

        if not self.same_eventids(self.eventname, other.eventname):
            raise ValueError("CMTSource.eventname must be equal to compare the events")

        # The origin time is the most problematic part
        origin_time = self.origin_time - other.origin_time
        pde_latitude = self.pde_latitude - other.pde_latitude
        pde_longitude = self.pde_longitude - other.pde_longitude
        pde_depth_in_m = self.pde_depth_in_m - other.pde_depth_in_m
        region_tag = self.region_tag
        eventame = self.eventname
        mb = self.mb - other.mb
        ms = self.ms - other.ms
        cmt_time = self.cmt_time - other.cmt_time
        half_duration = self.half_duration - other.half_duration
        latitude = self.latitude - other.latitude
        longitude = self.longitude - other.longitude
        depth_in_m = self.depth_in_m - other.depth_in_m
        m_rr = self.m_rr - other.m_rr
        m_tt = self.m_tt - other.m_tt
        m_pp = self.m_pp - other.m_pp
        m_rt = self.m_rt - other.m_rt
        m_rp = self.m_rp - other.m_rp
        m_tp = self.m_tp - other.m_tp

        return CMTSource(
            origin_time=origin_time,
            pde_latitude=pde_latitude,
            pde_longitude=pde_longitude,
            mb=mb,
            ms=ms,
            pde_depth_in_m=pde_depth_in_m,
            region_tag=region_tag,
            eventname=eventame,
            cmt_time=cmt_time,
            half_duration=half_duration,
            latitude=latitude,
            longitude=longitude,
            depth_in_m=depth_in_m,
            m_rr=m_rr,
            m_tt=m_tt,
            m_pp=m_pp,
            m_rt=m_rt,
            m_rp=m_rp,
            m_tp=m_tp,
        )
