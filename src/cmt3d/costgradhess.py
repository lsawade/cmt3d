from typing import Union, List, Tuple
from obspy import Stream
import numpy as np


class CostGradHess:

    def __init__(self,
                 data: Stream,
                 synt: Stream,
                 dsyn: Union[List[Stream], None] = None,
                 normalize: bool = True,
                 weight: bool = True,
                 verbose: bool = False):
        """Class to compute the cost, gradient, and Hessian
        for source inversion.

        Parameters
        ----------
        data : Stream
            Data Stream
        synt : Stream
            Synthetic Stream
        dsynt : Union[Stream, None], optional
            List of Frechet derivative stream, by default None
        normalize : bool, optional
            Flag on whether to normalize by trace, by default True
        weight : bool, optional
            flag on whether to weight or not, if True
            the datastream's traces' stats attribute needs to contain a
            weight attribute, by default True
        verbose : bool, optional
            Flag print warnings etc., by default False
        """

        self.data = data
        self.synt = synt
        self.dsyn = dsyn
        self.weight = weight
        self.normalize = normalize
        self.verbose = verbose

    def misfits(self, location=True) -> dict:
        """Takes in data and synthetics stream and computes a list of
        windowed least squares costs.

        Parameters
        ----------
        data : Stream
            data
        synt : Stream
            synthetics

        Returns
        -------
        dict
            dictionary over components of the data trace

        Notes
        -----

        :Author:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.03.29 16.30

        """
        if location:
            residuals = dict(
                R=dict(res=[], dlnA=[],  dt=[], lat=[], lon=[], az=[]),
                T=dict(res=[], dlnA=[],  dt=[], lat=[], lon=[], az=[]),
                Z=dict(res=[], dlnA=[],  dt=[], lat=[], lon=[], az=[])
            )
        else:
            residuals = dict(
                R=dict(res=[], dlnA=[],  dt=[]),
                T=dict(res=[], dlnA=[],  dt=[]),
                Z=dict(res=[], dlnA=[],  dt=[])
            )

        for _component, _compdict in residuals.items():
            compstream = self.data.select(component=_component)
            for tr in compstream:
                network, station, component = (
                    tr.stats.network, tr.stats.station, tr.stats.component)
                # Get the trace sampling time
                dt = tr.stats.delta
                d = tr.data

                try:
                    s = self.synt.select(network=network, station=station,
                                         component=component)[0].data

                    fnorm = 0
                    costt = []
                    for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                        ws = s[win.left:win.right]
                        wo = d[win.left:win.right]
                        costt.append(0.5 * (np.sum(tap * (ws - wo) ** 2) * dt))
                        fnorm += np.sum(tap * wo ** 2) * dt

                    # if self.weight:
                    #     costt *= tr.stats.weights

                    if self.normalize and fnorm != 0:
                        costt = np.array(costt)/fnorm

                    _compdict['res'].extend(costt)
                    _compdict['lat'].extend(
                        np.ones_like(costt) * tr.stats.latitude)
                    _compdict['lon'].extend(
                        np.ones_like(costt) * tr.stats.longitude)
                    _compdict['az'].extend(
                        np.ones_like(costt) * tr.stats.az)

                except Exception as e:
                    if self.verbose:
                        print(
                            f"Error at ({network}.{station}.{component}): {e}")

        return residuals

    def cost(self) -> float:
        """Takes in data and synthetics stream and computes windowed least squares
        cost. The stats object of the Traces in the stream _*must*_ contain both
        `windows` and the `tapers` attributes!

        Parameters
        ----------
        data : Stream
            data
        synt : Stream
            synthetics

        Returns
        -------
        float
            cost of entire stream

        Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)

        """

        x = 0.0

        for tr in self.data:
            network, station, component = (
                tr.stats.network, tr.stats.station, tr.stats.component)
            # Get the trace sampling time
            dt = tr.stats.delta
            d = tr.data

            try:
                s = self.synt.select(network=network, station=station,
                                     component=component)[0].data

                fnorm = 0
                costt = 0
                for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                    ws = s[win.left:win.right]
                    wo = d[win.left:win.right]
                    costt += 0.5 * (np.sum(tap * (ws - wo) ** 2) * dt)
                    fnorm += np.sum(tap * wo ** 2) * dt

                if self.weight:
                    costt *= tr.stats.weights

                if self.normalize and fnorm != 0:
                    costt /= fnorm

                x += costt

            except Exception as e:
                if self.verbose:
                    print(f"Error at ({network}.{station}.{component}): {e}")

        return x

    def grad(self) -> np.ndarray:
        """Computes the gradient of a the least squares cost function wrt. 1
        parameter given its forward computation and the frechet derivative.
        The stats object of the Traces in the stream _*must*_ contain both
        `windows` and the `tapers` attributes!

        Parameters
        ----------
        data : Stream
            data
        synt : Stream
            synthetics
        dsyn : Stream
            frechet derivatives

        Returns
        -------
        float
            gradient

        Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
        """

        if self.dsyn is None:
            raise ValueError(
                "List of Frechet derivatives is needed for the gradient\n"
                "commputation.")

        # Create empty gradient vector
        g = np.zeros(len(self.dsyn))

        for tr in self.data:
            network, station, component = (
                tr.stats.network, tr.stats.station, tr.stats.component)

            # Get the trace sampling time
            dt = tr.stats.delta
            d = tr.data

            try:
                s = self.synt.select(network=network, station=station,
                                     component=component)[0].data

                # Create trace list for the Frechet derivatives
                dsdm = []
                for ds in self.dsyn:
                    dsdm.append(ds.select(network=network, station=station,
                                          component=component)[0].data)

                gt = np.zeros(len(self.dsyn))
                fnorm = 0

                for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                    wsyn = s[win.left:win.right]
                    wobs = d[win.left:win.right]
                    fnorm += np.sum(tap * wobs ** 2) * dt

                    # Compute Gradient
                    for _i, _dsdm in enumerate(dsdm):
                        # Get derivate with respect to model parameter i
                        wdsdm = _dsdm[win.left:win.right]
                        gt[_i] += np.sum(((wsyn - wobs) * tap) * wdsdm) * dt

                if self.weight:
                    gt *= tr.stats.weights

                if self.normalize and fnorm != 0:
                    gt /= fnorm

                g += gt

            except Exception as e:
                if self.verbose:
                    print(
                        f"Error - Gradient - {network}.{station}.{component}: {e}")

        return g

    def hess(self) -> np.ndarray:
        """Computes the gradient and the approximate hessian of the cost function
        using the Frechet derivative of the forward modelled data.
        The stats object of the Traces in the stream _*must*_ contain both
        `windows` and the `tapers` attributes!

        Parameters
        ----------
        data : Stream
            data
        synt : Stream
            synthetics
        dsyn : Stream
            frechet derivatives

        Returns
        -------
        Tuple[float, float]
            Gradient, Approximate Hessian

        Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
        """

        if self.dsyn is None:
            raise ValueError(
                "List of Frechet derivatives is needed for the gradient\n"
                "and hessian computation.")

        h = np.zeros((len(self.dsyn), len(self.dsyn)))

        for tr in self.data:
            network, station, component = (
                tr.stats.network, tr.stats.station, tr.stats.component)

            # Get the trace sampling time
            dt = tr.stats.delta
            d = tr.data

            try:
                s = self.synt.select(network=network, station=station,
                                     component=component)[0].data

                # Create trace list for the Frechet derivatives
                dsdm = []
                for ds in self.dsyn:
                    dsdm.append(ds.select(network=network, station=station,
                                          component=component)[0].data)

                ht = np.zeros((len(self.dsyn), len(self.dsyn)))
                fnorm = 0

                # Loop over windows
                for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                    wsyn = s[win.left:win.right]
                    wobs = d[win.left:win.right]
                    fnorm += np.sum(tap * wobs ** 2) * dt

                    # Compute Gradient
                    for _i, _dsdm_i in enumerate(dsdm):
                        # Get derivate with respect to model parameter i
                        wdsdm_i = _dsdm_i[win.left:win.right]

                        for _j, _dsdm_j in enumerate(dsdm):
                            # Get derivate with respect to model parameter j
                            wdsdm_j = _dsdm_j[win.left:win.right]
                            ht[_i, _j] += ((wdsdm_i * tap) @
                                           (wdsdm_j * tap)) * dt

                if self.weight:
                    ht *= tr.stats.weights

                if self.normalize and fnorm != 0:
                    ht /= fnorm

                h += ht

            except Exception as e:
                if self.verbose:
                    print(f"When accessing {network}.{station}.{component}")
                    print(e)

        return h

    def grad_and_hess(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the gradient and the approximate hessian of the cost function
        using the Frechet derivative of the forward modelled data.
        The stats object of the Traces in the stream _*must*_ contain both
        `windows` and the `tapers` attributes!

        Parameters
        ----------
        data : Stream
            data
        synt : Stream
            synthetics
        dsyn : Stream
            frechet derivatives

        Returns
        -------
        Tuple[float, float]
            Gradient, Approximate Hessian

        Last modified: Lucas Sawade, 2020.09.28 19.00 (lsawade@princeton.edu)
        """

        if self.dsyn is None:
            raise ValueError(
                "List of Frechet derivatives is needed for the gradient\n"
                "and hessian computation.")

        g = np.zeros(len(self.dsyn))
        h = np.zeros((len(self.dsyn), len(self.dsyn)))

        for tr in self.data:
            network, station, component = (
                tr.stats.network, tr.stats.station, tr.stats.component)

            # Get the trace sampling time
            dt = tr.stats.delta
            d = tr.data

            try:
                s = self.synt.select(network=network, station=station,
                                     component=component)[0].data

                # Create trace list for the Frechet derivatives
                dsdm = []
                for ds in self.dsyn:
                    dsdm.append(ds.select(network=network, station=station,
                                          component=component)[0].data)

                gt = np.zeros(len(self.dsyn))
                ht = np.zeros((len(self.dsyn), len(self.dsyn)))
                fnorm = 0

                # Loop over windows
                for win, tap in zip(tr.stats.windows, tr.stats.tapers):
                    wsyn = s[win.left:win.right]
                    wobs = d[win.left:win.right]
                    fnorm += np.sum(tap * wobs ** 2) * dt

                    # Compute Gradient
                    for _i, _dsdm_i in enumerate(dsdm):
                        # Get derivate with respect to model parameter i
                        wdsdm_i = _dsdm_i[win.left:win.right]
                        gt[_i] += np.sum(((wsyn - wobs) * tap) * wdsdm_i) * dt

                        for _j, _dsdm_j in enumerate(dsdm):
                            # Get derivate with respect to model parameter j
                            wdsdm_j = _dsdm_j[win.left:win.right]
                            ht[_i, _j] += ((wdsdm_i * tap) @
                                           (wdsdm_j * tap)) * dt

                if self.weight:
                    gt *= tr.stats.weights
                    ht *= tr.stats.weights

                if self.normalize and fnorm != 0:
                    gt /= fnorm
                    ht /= fnorm

                g += gt
                h += ht

            except Exception as e:
                if self.verbose:
                    print(f"When accessing {network}.{station}.{component}")
                    print(e)

        return g, h
