import numpy as np
from .snn import SNN


class GeoWeights:

    def __init__(self, latitude, longitude):
        """Creates GEoWeights class to get geographical density weights.

        Parameters
        ----------
        latitude : arraylike
            latitudes
        longitude : arraylike
            longitudes

        Notes
        -----

        :Author:
            Lucas Sawade (lsawade@princeton.edu)

        :Last Modified:
            2021.04.16 09.30

        """

        self.lat = latitude
        self.lon = longitude

        self.kdtree = SNN(self.lat, self.lon)
        self.dij = self.kdtree.sparse_distance_matrix()

    def get_weights(self, ref: float = 1.0):
        """Compute the weights based on equation 22 in Ruan et al. 2019

        Parameters
        ----------
        ref : float, optional
            Reference distance, by default 1.0

        Returns
        -------
        arraylike
            weights
        """

        distsexp = np.exp(-(self.dij/ref)**2)
        w = 1.0 / np.sum(distsexp, axis=1)
        w /= np.sum(w) / len(w)

        return w

    def get_condition(self, ctype='fracmax', param=0.33):
        """Evaluate condition of distribution to compute a reference distance
        used to compute the weights

        Parameters
        ----------
        ctype : str, optional
            Way of choosing the condition number. Possible values
            ``['fracmax', 'q', 'max', 'dist']``. ``Fracmax`` chooses the
            reference distance based on a fraction of the maximum that the
            condition number graph is taking, ``param`` defines the fraction.
            ``q`` chooses the first quantile of the condition number graph,
            ``param`` defines which quantile. ``max`` chooses the maximum
            of the condition number graph ``param`` is ununsed. ``dist``
            evaluates the distribution of distances, where ``param`` is the
            value that is taken from the dist, and takes the values of
            ``['max', 'mean', 'median']``, by default 'fracmax'
        param : float, optional
            See ``ctype`` for detailed description, by default 0.33

        Returns
        -------
        tuple
            reference vector, condition number vector, actual reference,
            actual condition number


        Raises
        ------
        ValueError
            if wrong ctype is chosen
        ValueError
            if wrong dist type is chosen
        """

        # Condition number function
        def condfunc(ref):
            w = self.get_weights(ref=ref)
            return np.max(w)/np.min(w)

        # vectorize function
        reffunc = np.vectorize(condfunc)

        # Number of times to compute this
        vref = np.arange(1, 180, 1.0)

        # Get condition numbers
        vcondno = reffunc(vref) - 1.0

        # Get quantile condition number
        cumcondo = np.cumsum(vcondno)/np.sum(vcondno)
        if ctype == "q":
            ind_cond = np.abs(cumcondo-param).argmin()
        elif ctype == "max":
            ind_cond = vcondno.argmax()
        elif ctype == "fracmax":
            if type(param) == int or type(param) == float:
                ind_max = vcondno.argmax()
                ind_cond = np.abs(
                    vcondno[:ind_max]-(param*vcondno.max())).argmin()
            else:
                raise ValueError(f"For {ctype} the parameter must be numeric")
        elif ctype == "dist":
            triu = np.triu(self.dij)
            dists = triu[np.nonzero(triu)].flatten()
            n, bins = np.histogram(dists, bins=int(np.ceil(len(dists)*0.1)))
            xbins = bins[:-1] + 0.5*np.diff(bins)
            if param == "median":
                ind_d = np.argsort(n)[len(n)//2]
            elif param == "mean":
                ind_d = np.abs(xbins-np.mean(n)).argmin()
            elif param == "max":
                ind_d = n.argmax()
            d = xbins[ind_d]
            distcondno = condfunc(d)
            ind_cond = np.abs(distcondno-vcondno).argmin()
        else:
            raise ValueError(f"{ctype.capitalize()} is not implemented.")

        condno = vcondno[ind_cond]
        ref = vref[ind_cond]

        return vref, vcondno, ref, condno
