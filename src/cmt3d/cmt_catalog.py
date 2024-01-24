from __future__ import annotations

import os
import typing as tp
import sys
import time
from copy import deepcopy
import inspect
from typing import List, Optional, Union, Iterable
from glob import glob
import _pickle as cPickle
import numpy as np
import obspy
from obspy.core.event import Event

try:
    from pandas.core.frame import DataFrame
except ModuleNotFoundError:
    pass

from .source import CMTSource
from . import sourcedecomposition
from cmt3d.utils import sec2hhmmss


class CMTCatalog:
    cmts: List[CMTSource]

    # Attributes and properties of a CMTSource
    attributes: list = [a for a in vars(CMTSource()).keys()]
    attributes += [
        a
        for a, _ in inspect.getmembers(CMTSource, lambda x: not inspect.isroutine(x))
        if "__" not in a
    ]
    specialpropertylist: list = []

    # Just for printing
    uniqueatt = set(attributes)

    # Get possible decomposition types
    dtypes = [
        func for func, _ in inspect.getmembers(sourcedecomposition, inspect.isfunction)
    ]

    def __init__(self, cmts: Union[List[CMTSource], None] = None):
        """Creates instance of CMTCatalog

        Parameters
        ----------
        cmts : Union[List[CMTSource], None], optional
            List of CMTSource's, by default None
        """

        if cmts is not None:
            self.cmts = cmts
        else:
            self.cmts = []

    def save(self, filename: str):
        """Saves the catalog as Pickle in bytes"""
        with open(filename, "wb") as output_file:
            cPickle.dump(self, output_file)

    @classmethod
    def load(cls, filename: str):
        """Loads the ``catalog.pkl`` in bytes"""
        with open(filename, "rb") as input_file:
            cls = cPickle.load(input_file)
        return cls

    @classmethod
    def from_obspy_catalog(cls, cat: obspy.Catalog):
        """Converts obspy catalog to CMTCatalog"""

        cmts = [CMTSource.from_event(ev) for ev in cat]

        return cls(cmts)

    @classmethod
    def from_file_list(self, file: Union[str, list]):
        """Takes in filestring, globstring list of files, or list of glob
        strings. Then converts each file to a cmtsolution, from which a
        catalog class will be generated."""

        if isinstance(file, str):
            filelist = [file]
        elif isinstance(file, list):
            if isinstance(file[0], str):
                filelist = file
            else:
                raise ValueError("List type not supported. Must be list of strings.")

        cmtfilelist = []
        for _file in filelist:
            cmtfilelist.extend(glob(_file))

        # Raise error if no files matching can be found!
        if len(cmtfilelist) == 0:
            print("\nFiles can't be found.\n")
            print(file)
            raise ValueError()

        cmtlist = []
        for _cmtfile in cmtfilelist:
            cmtlist.append(CMTSource.from_CMTSOLUTION_file(_cmtfile))

        return self(cmtlist)

    def getvals(self, vtype="tensor", dtype: Union[str, None] = None):
        """This function is a bit more elaborate, but it gives access to each
        of the CMT parameters in form of ndarrays.

        Parameters
        ----------
        vtype : str
            String of attribute of CMT solutions, default ``tensor``
            if ``decomp`` (decomposition) is chosen one needs to specify the
            type of decomposition ``dtype``
        dtype : str
            Decomposition type. decompositions are defined in module
            ``sourcedecomposition``, default None
            Needs to be specified if ``vtype`` is chosen to be ``decomp``.

        Return
        ------
        arraylike
            parameter array depending on the chosen parameter
        """

        # If very normal attribute type
        if vtype in self.attributes and vtype not in self.specialpropertylist:
            vals = []
            for _cmt in self.cmts:
                vals.append(getattr(_cmt, vtype))
            return np.array(vals)

        # If very special attribute typem but still attribute
        elif vtype in self.attributes and vtype in self.specialpropertylist:
            lb = []
            ev = []
            for _cmt in self.cmts:
                lb, ev = getattr(_cmt, vtype)
            return np.array(lb), np.array(ev)

        elif vtype == "decomp":
            if dtype is not None and dtype in self.dtypes:
                vals = []
                for _cmt in self.cmts:
                    vals.append(getattr(_cmt, vtype)(dtype))
                return np.array(vals)
            else:
                raise ValueError(
                    f"Method for decomposition must be given and" f"in {self.dtypes}"
                )
        else:
            raise ValueError(
                f"Value {vtype} not implemented, choose from" f"{self.uniqueatt}"
            )

    @property
    def dataframe(self) -> DataFrame:
        if "DataFrame" not in sys.modules.keys():
            raise ValueError(
                "Pandas is probably not installed. It"
                "s not loaded if it is not installed."
            )

        columns = [
            "origin_time",
            "pde_latitude",
            "pde_longitude",
            "pde_depth_in_m",
            "mb",
            "ms",
            "region_tag",
            "eventname",
            "time_shift",
            "half_duration",
            "latitude",
            "longitude",
            "depth_in_m",
            "m_rr",
            "m_tt",
            "m_pp",
            "m_rt",
            "m_rp",
            "m_tp",
        ]
        rows = []
        for cmt in self:
            rows.append(cmt.to_row())
        return DataFrame(rows, columns=columns)

    def add(
        self,
        cmt: Union[
            List[Union[CMTSource, Event]], CMTSource, Event, obspy.Catalog, CMTCatalog
        ],
    ):
        """Adds an event a

        Parameters
        ----------
        cmt : Union[List[Union[CMTSource, Event]], CMTSource, Event, Catalog]
            [description]
        """

        if isinstance(cmt, CMTSource):
            self.cmts.append(cmt)
        elif isinstance(cmt, Event):
            self.cmts.append(CMTSource.from_event(cmt))
        elif (
            isinstance(cmt, list)
            or isinstance(cmt, obspy.Catalog)
            or isinstance(cmt, CMTCatalog)
        ):
            for _cmt in cmt:
                if isinstance(_cmt, CMTSource):
                    self.cmts.append(_cmt)
                elif isinstance(_cmt, Event):
                    self.cmts.append(CMTSource.from_event(_cmt))
        else:
            ValueError(f"Type {type(cmt)} is not supported to be added to the catalog.")

    def get_event(self, eventname: str):
        # If eventid is a string
        for _i, _cmt in enumerate(self.cmts):
            if _cmt.eventname == eventname:
                return _cmt

        raise ValueError(f"No event in catalog for {eventname}")

    def pop(self, eventname: Union[List[str], str, List[int], int]):
        # If eventid is a string
        if isinstance(eventname, str):
            popindices = [
                _i for _i, _cmt in enumerate(self.cmts) if _cmt.eventname == eventname
            ]

        # If eventid is a list
        elif isinstance(eventname, list):
            popindices = []

            for _ev in eventname:
                if isinstance(_ev, str):
                    popindices.append(
                        [
                            _i
                            for _i, _cmt in enumerate(self.cmts)
                            if _cmt.eventname == _ev
                        ][0]
                    )
                elif isinstance(_ev, int):
                    popindices.append(_ev)
                else:
                    raise ValueError(
                        f"Type {eventname[0]} for event " f"popping is not supported."
                    )
        else:
            raise ValueError(f"Type {eventname} for popping is not supported.")

        # Pop indeces in reverse order to not mess up the list.
        for _popindex in reversed(sorted(popindices)):
            self.cmts.pop(_popindex)

    def __len__(self):
        """Returns the lengths of the catalog"""
        return len(self.cmts)

    def __iter__(self):
        """Returns the Iterator object."""
        return iter(self.cmts)

    def __getitem__(self, index: Union[int, Iterable[int], slice]):
        """Returns index from cmt list"""
        if isinstance(index, int):
            return self.cmts[index]
        elif isinstance(index, slice):
            return CMTCatalog(self.cmts[index])
        elif isinstance(index, Iterable):
            retlist = []
            for _i in index:
                retlist.append(self.cmts[int(_i)])
            return CMTCatalog(retlist)
        else:
            raise ValueError("Index type not supported.")

    @staticmethod
    def same_eventids(id1, id2):
        id1 = id1 if not id1[0].isalpha() else id1[1:]
        id2 = id2 if not id2[0].isalpha() else id2[1:]

        return id1 == id2

    def unique(self, ret: bool = False):
        """Applies uniqueness condition depending on eventname or returns
        catalog with unique entries. Default is application on self."""

        # Get eventnames
        eventnames = self.getvals("eventname")

        # Get unique entries from eventnames
        _, uniq = np.unique(eventnames, return_index=True)

        # Return or apply on self
        if ret:
            return CMTCatalog(self[uniq].cmts)
        else:
            self.cmts = self[uniq].cmts

    def sort(self, key="origin_time"):
        """Sorts the loaded CMT solutions after key that is given.

        Parameters
        ----------
        key : str, optional
            Key could be any attribute of a cm solution,
            by default "origin_time"

        Raises
        ------
        ValueError
            If key is not supported.
        """
        if key in self.attributes:
            vals = self.getvals(key)
            indeces = vals.argsort().astype(int)
            self.cmts = self[indeces].cmts
        else:
            raise ValueError(
                f"{key} is not a valid sorting value.\n" f"Use {self.attributes}."
            )

    def filter(self, maxdict: dict = dict(), mindict: dict = dict()):
        """This uses two dictionaries as inputs. One dictionary for
        maximum values and one dictionary that contains min values of the
        elements to filter. To do that we create a dictionary containing
        the attributes and properties of
        :class:``lwsspy.seismo.source.CMTSource``.

        List of Attributes and Properties
        -------------------------

        .. literal::

            origin_time
            pde_latitude
            pde_longitude
            pde_depth_in_m
            mb
            ms
            region_tag
            eventname
            cmt_time
            half_duration
            latitude
            longitude
            depth_in_m
            m_rr
            m_tt
            m_pp
            m_rt
            m_rp
            m_tp
            M0
            moment_magnitude
            time_shift

        Example
        -------

        Let's filter the catalog to only contain events with a maximum depth
        of 20km.

        >>> maxfilterdict = dict(depth_in_m=20000.0)
        >>> cmtcat = CMTCatalog.from_files("CMTfiles/*")
        >>> filtered_cat = cmtcat.filter(maxdict=maxfilterdict)

        will returns a catalog with events shallower than 20.0 km.
        """

        # Create new list of cmts
        newlist = deepcopy(self.cmts)

        poppedlist = []

        # First maxvalues
        for key, value in maxdict.items():
            # Create empty pop set
            popset = set()

            # Check CMTs that are below threshold for key
            for _i, _cmt in enumerate(newlist):
                if getattr(_cmt, key) > value:
                    popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                poppedlist.append(newlist.pop(_i))

        # First minvalues
        for key, value in mindict.items():
            # Create empty pop set
            popset = set()

            # Check CMTs that are above threshold for key
            for _i, _cmt in enumerate(newlist):
                if getattr(_cmt, key) < value:
                    popset.add(_i)

            # Convert set to list and sort
            poplist = list(popset)
            poplist.sort()

            # Pop found indeces
            for _i in poplist[::-1]:
                poppedlist.append(newlist.pop(_i))

        return CMTCatalog(newlist), CMTCatalog(poppedlist)

    def split_to_mechanism(
        self,
        thrust_tension_plunge_threshold: float = 50,
        thrust_null_value_threshold: float = 0.2,
        # thrust_null_plunge_threshold: float = 10.0,
        normal_pressure_plunge_threshold: float = 50,
        normal_null_value_threshold: float = 0.2,
        strike_slip_null_plunge_threshold: float = 20,
        strike_slip_null_value_threshold: float = 0.2,
    ) -> tp.Dict[str, CMTCatalog]:
        """Split catalog into normal, thrust and strike-slip events. Using the
        following criteria:

        - Normal:
            * P axis plunge > normal_tension_plunge_threshold (e.g. 70 degrees)
            * null axis value < normal_null_value_threshold (e.g. 0.2)
        - Thrust:
            * T axis plunge > thrust_tension_plunge_threshold (e.g. 70 degrees)
            * null axis value < thrust_null_value_threshold
        - Strike-slip:
            * Null axis plunge > strike_slip_tension_plunge_threshold (e.g. 70 degrees)
            * null axis value < strike_slip_null_value_threshold (e.g. 0.2


        Parameters
        ----------
        thrust_tension_plunge_threshold : float, optional
            _description_, by default 50
        thrust_null_value_threshold : float, optional
            _description_, by default 0.2
        normal_pressure_plunge_threshold : float, optional
            _description_, by default 70
        normal_null_value_threshold : float, optional
            _description_, by default 0.2
        strike_slip_null_plunge_threshold : float, optional
            _description_, by default 20
        strike_slip_null_value_threshold : float, optional
            _description_, by default 0.2

        Returns
        -------
        tp.Dict[str, CMTCatalog]
            Dictionary containing the catalogs for each mechanism with keywords:
            ``normal``, ``thrust``, ``strike-slip``.
        """
        normal_events = []
        thrust_events = []
        strike_slip_events = []

        for _i, ev in enumerate(self):
            # eqpar gives T-B-P (tension - null - compression) axes
            eivals = ev.eqpar[4]
            plunge = ev.eqpar[6]
            azimuth = ev.eqpar[7]

            # Assign the values to variables (T, B, P)
            _, LB, _ = eivals
            PT, PB, PP = plunge
            # AT, AB, AP = azimuth

            # To get normal events, we select events that have a
            # P axis plunge > normal_tension_plunge_threshold (e.g. 70 degrees)
            # and
            # null axis value < normal_null_value_threshold (e.g. 0.2)
            if (
                PP
                > normal_pressure_plunge_threshold
                # and LB < normal_null_value_threshold
            ):
                normal_events.append(ev)

            # To get thrust events, we select events that have a
            # T axis plunge > thrust_tension_plunge_threshold (e.g. 70 degrees)
            # and
            # null axis value < thrust_null_value_threshold
            if (
                PT
                > thrust_tension_plunge_threshold
                # and LB < thrust_null_value_threshold
            ):
                thrust_events.append(ev)

            # To get strike-slip events, we select events that have a
            # Null axis plunge > strike_slip_tension_plunge_threshold (e.g. 70 degrees)
            # and
            # null axis value < strike_slip_null_value_threshold (e.g. 0.2
            if (
                PB
                > strike_slip_null_plunge_threshold
                # and LB < strike_slip_null_value_threshold
            ):
                strike_slip_events.append(ev)

        return {
            "normal": CMTCatalog(normal_events),
            "thrust": CMTCatalog(thrust_events),
            "strike-slip": CMTCatalog(strike_slip_events),
        }

    def check_ids(self, other: CMTCatalog, verbose: bool = False):
        """Takes in another catalog and returns a tuple of self and other
        that are contain only common eventnames and that are sorted.

        Parameters
        ----------
        other : CMTCatalog
            Another catalog
        verbose : bool
            Print events that were only found in one catalog.
        """

        cmtself = []
        cmtother = []
        for _cmt in self.cmts:
            try:
                _cmtother = other.get_event(_cmt.eventname)
                cmtself.append(_cmt)
                cmtother.append(_cmtother)
            except ValueError:
                if verbose:
                    print(f"Didn't find corresponding events " f"for {_cmt.eventname}")

        return CMTCatalog(cmtself), CMTCatalog(cmtother)

    def in_catalog(
        self, event: Union[str, List[str]], verbose=False, thorough_check: bool = False
    ):
        """Check whether event id is in the catalog or not. If single stringf.
        is provided it will return the CMTSource if it is in the catalog. If
        a list of IDs is provided a new catalog containing those events will be
        returned. If there is no match a ValueError is raised.


        Parameters
        ----------
        event : Union[str, List[str]]
            event id or list of event ids
        verbose: bool
            whether to print explanations
        thorough_check: bool
            check whether maybe the first letter is wrong
        """

        if isinstance(event, str):
            single = True
            event = [event]
        else:
            single = False

        # Loop over cmts and add them to list of they are in the catalog
        cmts = []
        for _cmtname in event:
            try:
                _cmt = self.get_event(_cmtname)
                cmts.append(_cmt)

            except ValueError as e:
                if thorough_check:
                    for _letter in ["B", "S", "M", "C", "E"]:
                        try:
                            # Check corrected name
                            _cmt = self.get_event(_letter + _cmtname[1:])
                            cmts.append(_cmt)
                            if verbose:
                                print(
                                    f"{_cmtname} had wrong ID: {_letter + _cmtname[1:]}"
                                )
                            break

                        except ValueError as e:
                            if verbose:
                                print(f"{_cmtname} not {_letter + _cmtname[1:]}")

                            if _letter == "C":
                                if verbose:
                                    print(
                                        f"Didn't find corresponding events "
                                        f"for {_cmtname}"
                                    )

                else:
                    if verbose:
                        print(e)
                        print(f"Didn't find corresponding events " f"for {_cmtname}")

        if len(cmts) == 0:
            raise ValueError("No CMTs in catalog that match the input id(s).")

        if single:
            return cmts[0]
        else:
            return CMTCatalog(cmts)

    def cmts2dir(self, outdir: str = "./newcatalog"):
        # Create dir if doesn't exist.
        if os.path.exists(outdir) is False:
            os.mkdir(outdir)

        # Start print
        print(f"---> Writing cmts to {outdir}/")
        t0 = time.time()

        # Writing
        for _cmt in self.cmts:
            outfilename = os.path.join(outdir, _cmt.eventname)
            _cmt.write_CMTSOLUTION_file(outfilename)

        # End print
        t1 = time.time()
        print(f"     Done. Elapsed Time: {sec2hhmmss(t1-t0)[-1]}")

    def cmts2file(self, outfile: str = "./catalog.txt"):
        # Start print
        print(f"---> Writing cmts to {outfile}")
        t0 = time.time()

        # Writing
        for _i, _cmt in enumerate(self.cmts):
            if _i == 0:
                _cmt.write_CMTSOLUTION_file(outfile)
            else:
                _cmt.write_CMTSOLUTION_file(outfile, mode="a")

        # End print
        t1 = time.time()
        print(f"     Done. Elapsed Time: {sec2hhmmss(t1-t0)[-1]}")

    def printcmts(self, outfile: Optional[str] = None):
        """Prints the ids or writes them to a file"""

        outcat = deepcopy(self)
        outcat.sort()

        if isinstance(outfile, str):
            with open(outfile, "w") as f:
                for _cmt in self:
                    f.write(_cmt.eventname + "\n")
        else:
            for _cmt in self:
                print(_cmt.eventname)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self):
        Mw = self.getvals(vtype="moment_magnitude")
        utc = self.getvals(vtype="origin_time")
        lat = self.getvals(vtype="latitude")
        lon = self.getvals(vtype="longitude")

        string = "CMTCatalog :\n"
        string += 60 * "-" + "\n"
        string += "\n"
        string += f"Event #:{len(self.cmts):.>52}\n"
        string += f"Starttime:{min(utc).strftime('%Y/%m/%d, %H:%M:%S'):.>50}\n"
        string += f"Endtime:{max(utc).strftime('%Y/%m/%d, %H:%M:%S'):.>52}\n"
        string += f"Bounding Box:{'Latitude: ':.>25}{'':.<2}[{np.min(lat):8.3f}, {np.max(lat):8.3f}]\n"
        string += (
            f"{'Longitude: ':.>39}{'':.<1}[{np.min(lon):8.3f}, {np.max(lon):8.3f}]\n"
        )
        string += f"Moment Magnitude:{'':.>23}[{np.min(Mw):8.3f}, {np.max(Mw):8.3f}]\n"
        string += "\n"
        string += 60 * "-" + "\n"

        return string
