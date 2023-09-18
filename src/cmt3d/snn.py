"""
Taken from ObsPy originally, but heavily modified.

"""
# External
from __future__ import annotations
import typing as tp
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree


def geo2cart(r: float or np.ndarray or list,
             theta: float or np.ndarray or list,
             phi: float or np.ndarray or list) \
    -> tp.Tuple[float | np.ndarray | list,
                float | np.ndarray | list,
                float | np.ndarray | list]:
    """Computes Cartesian coordinates from geographical coordinates.

    Parameters
    ----------
    r : float or numpy.ndarray or list
        Radius
    theta : float or numpy.ndarray or list
        Latitude (-90, 90)
    phi : float or numpy.ndarray or list
        Longitude (-180, 180)

    Returns
    -------
    float or np.ndarray or list, float or np.ndarray or list, float or np.ndarray or list
        (x, y, z)
    """

    if type(r) is list:
        r = np.array(r)
        theta = np.array(theta)
        phi = np.array(phi)

    # Convert to Radians
    thetarad = theta * np.pi/180.0
    phirad = phi * np.pi/180.0

    # Compute Transformation
    x = r * np.cos(thetarad) * np.cos(phirad)
    y = r * np.cos(thetarad) * np.sin(phirad)
    z = r * np.sin(thetarad)

    if type(r) is list:
        x = x.tolist()
        y = y.tolist()
        z = z.tolist()

    return x, y, z


class SNN(object):
    """Spherical nearest neighbour queries using scipy's fast kd-tree
    implementation.

    Attributes
    ----------
    data : numpy.ndarray
        cartesian point data array [x,y,z]
    kd_tree : scipy.spatial.cKDTree
        a KDTree used to query data


    Methods
    -------
    query(qlat, qlon)
        Query a set of latitudes and longitudes
    query_pairs(maximum_distance)
        Find pairs of points that are within a certain distance of each other
    interp(data, qlat, qlon)
        Use the kdtree to interpolate data corresponding
        to the points of the Kdtree onto a new set of points using nearest
        neighbor interpolation or weighted nearest neighbor
        interpolation (default).

    Notes
    -----

    :Authors:
        Lucas Sawade (lsawade@princeton.edu)

    :Last Modified:
        2021.09.29 19.00

    """

    def __init__(self, lat, lon):
        """Initialize class

        Parameters
        ----------
        lat : numpy.ndarray
            latitudes
        lon : numpy.ndarray
            longitudes
        """
        cart_data = self.spherical2cartesian(lat, lon)
        self.data = cart_data
        self.kd_tree = cKDTree(data=cart_data, leafsize=10)

    def query(self, qlat, qlon):
        """Query latitude and longitude values from kdtree

        Parameters
        ----------
        qlat : np.ndarray or float
            query latitude
        qlon : np.ndarray or float
            query longitude

        Returns
        -------
        tuple
            (distance, indeces) to closest point in kdtree
        """
        points = self.spherical2cartesian(qlat, qlon)
        d, i = self.kd_tree.query(points)

        # Filter NaNs. Happens when not enough points are available.
        m = np.isfinite(d)
        return d[m], i[m]

    def query_pairs(self, maximum_distance=180.0):
        """Query pairs within the kdtree

        Parameters
        ----------
        maximum_distance : float
            Maximum query distance in deg

        Returns
        -------
        set or ndarray
            Set of pairs (i, j) where i < j
        """
        distkm = np.abs(2 * np.sin(maximum_distance/2.0 /
                                   180.0*np.pi)) * 6371.0
        return self.kd_tree.query_pairs(distkm, output_type='ndarray')

    def sparse_distance_matrix(self, other: SNN | None = None,
                               maximum_distance=180.0, sparse: bool = False,
                               km: bool = False):
        """Computes the sparse distance matrix between two kdtree. if no other
        kdtree is provided, this kdtree is used

        Parameters
        ----------
        other : SNN
            other SNN tree
        maximum_distance : float
            Maximum query distance in deg
        sparse: bool
            Flag to output sparse array for memory. Default False
        km: bool
            flag to output distances in kilometers. Default False


        Returns
        -------
        set or ndarray
            Set of pairs (i, j) where i < j
        """
        # Get distance
        distkm = np.abs(2 * np.sin(maximum_distance/2.0 /
                                   180.0*np.pi)) * 6371.0

        # Get tree
        if isinstance(other, SNN):
            other = other.kd_tree
        else:
            other = deepcopy(self).kd_tree

        # Compute dense or sparse distance matrix
        if sparse:
            output_mat = self.kd_tree.sparse_distance_matrix(
                other, distkm)
        else:
            output_mat = self.kd_tree.sparse_distance_matrix(
                other, distkm).toarray()

        # Convert form kilometers to degrees.
        if km is False:
            output_mat *= 180.0 / np.pi / 6371.0

        return output_mat

    def interpolator(self, qlat, qlon, maximum_distance=None,
                     no_weighting=False, k: tp.Optional[int] = 10,
                     p: float = 2.0):
        """Spherical interpolation function using the ``SNN`` object.
        Returns an interpolator that can be used for interpolating the same
        set of locations based on the KDTree. The only input the interpolator
        takes are the data corresponding to the points in the KDTree.

        Parameters
        ----------
        qlat : numpy.ndarray
            query latitudes
        qlon : numpy.ndarray
            query longitude
        maximum_distance : float, optional
            max distace for the interpolation in degree angle. Default None.
            If the mindistance to any points is larger than maximum_distance the
            interpolated value is set to ``np.nan``.
        no_weighting : bool, optional
            Whether or not the function uses a weightied nearest neighbor
            interpolation
        k : int, optional
            Define maximum number of neighbors to be used for the weighted
            interpolation. Not used if ``no_weighting = True``. Default 10
        p : float, optional
            Exponent to compute the inverse distance weights. Note that in
            the limit ``p->inf`` is just a nearest neighbor interpolation.
            Default is 2


        Notes
        -----

        It is always recommended to set k, since k may or may not be dependent
        on the data you want to interpolate.

        In the future, I may add a variable weighting function for the
        weighted interpolation.

        Please refer to https://en.wikipedia.org/wiki/Inverse_distance_weighting
        for the interpolation weighting.
        """

        # Get query points in cartesian
        shp = qlat.shape
        points = self.spherical2cartesian(qlat.flatten(), qlon.flatten())

        # Query points
        if no_weighting:
            # Get distances and indeces
            d, inds = self.kd_tree.query(points)

            def interpolator(data):

                # Assign the interpolation data.
                qdata = data[inds]

                # Filter out distances too far out.
                if maximum_distance is not None:
                    qdata = np.where(
                        d <= np.abs(
                            2 * np.sin(maximum_distance/2.0/180.0*np.pi))
                        * 6371.0,
                        qdata, np.nan)

                return data[inds].reshape(shp)

        else:

            # Set K to the max number of points if not given
            if k is None:
                k = self.kd_tree.n
            else:
                if k > self.kd_tree.n:
                    k = self.kd_tree.n

            # Filter out distances too far out.
            # Modified Shepard's method
            if maximum_distance is not None:

                # Get cartesian distance
                cartd = np.abs(2 * np.sin(maximum_distance/2.0/180.0*np.pi)
                               * 6371.0)

                # Get multiple distances and indeces
                d, inds = self.kd_tree.query(
                    points, k=k)  # Do not add this: distance_upper_bound=cartd*1.01)

                if d.shape == (1,):
                    d = d.reshape((1, 1))
                    inds = inds.reshape((1, 1))

                # Compute the weights using my inverse distance metric to avoid
                # division by 0. Based on 1/(1 + x**2)
                w = (np.fmax(0, (cartd - d)/cartd)/(1 + (d/cartd)**2))**p

                def interpolator(data):
                    """Using the weights and indices, we can return an interpolator
                    """

                    # Making sure that the wighted sum does not include weights
                    # where there is no data.
                    tmp_w = np.where(~np.isnan(data[inds]), w, 0.0)
                    tmp_wsum = np.sum(tmp_w, axis=1)
                    datarows = ~np.isclose(tmp_wsum, 0.0)

                    # Empty array
                    qdata = np.empty_like(tmp_wsum)
                    qdata[:] = np.nan

                    # Check which rows only contain nans
                    tmp_weights = tmp_w[datarows] / \
                        tmp_wsum[datarows, np.newaxis]

                    # Interpolated data
                    idata = tmp_weights * data[inds][datarows]
                    nanrows = np.isnan(idata[:, 0])

                    # Do interpolation and replace only-nan rows with np.nan
                    tmp2 = np.nansum(idata, axis=1)
                    tmp2[nanrows] = np.nan

                    qdata[datarows] = tmp2

                    return qdata.reshape(shp)

            # Shepard's method
            else:

                # Get multiple distances and indeces
                d, inds = self.kd_tree.query(
                    points, k=k)

                if d.shape == (1,):
                    d = d.reshape((1, 1))
                    inds = inds.reshape((1, 1))

                # Compute the weights using modified shepard
                # The distance being normalized by the standard deviation of
                # the closest points
                if np.all(np.nanstd(d[:, 0]) == 0.0):
                    if np.all(np.nanstd(d[:, :]) == 0.0):
                        std = 1.0
                    else:
                        std = np.nanstd(d[:, :])
                else:
                    std = np.nanstd(d[:, 0])

                w = (1 / (1 + d / std)) ** p

                def interpolator(data):
                    """Using the weights and indices, we can return an interpolator
                    """

                    # Making sure that the wighted sum does not include weights
                    # where there is no data.
                    tmp_w = np.where(~np.isnan(data[inds]), w, 0.0)
                    tmp_wsum = np.sum(tmp_w, axis=1)
                    datarows = ~np.isclose(tmp_wsum, 0.0)

                    # Empty array
                    qdata = np.empty_like(tmp_wsum, dtype=np.float32)
                    qdata[:] = np.nan

                    # Check which rows only contain nans
                    tmp_weights = tmp_w[datarows] / \
                        tmp_wsum[datarows, np.newaxis]

                    # Interpolated data
                    idata = tmp_weights * data[inds][datarows]
                    nanrows = np.isnan(idata[:, 0])

                    # Do interpolation and replace only-nan rows with np.nan
                    tmp2 = np.nansum(idata, axis=1)
                    tmp2[nanrows] = np.nan

                    qdata[datarows] = tmp2

                    return qdata.reshape(shp)

        return interpolator

    @ staticmethod
    def spherical2cartesian(lat, lon):
        """
        Converts lats and lons to shape(len(list), 3)
        containing x/y/z in meters.
        """
        # Create three arrays containing lat/lng/radius.
        r = np.ones_like(lat) * 6371.0

        # Convert data from lat/lng to x/y/z.
        x, y, z = geo2cart(r, lat, lon)

        return np.vstack((x, y, z)).T
