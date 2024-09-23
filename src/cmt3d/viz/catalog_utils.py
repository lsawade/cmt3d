import typing as tp
import numpy as np
from collections import OrderedDict

if tp.TYPE_CHECKING:
    from cmt3d.catalog import CMTCatalog


labeldict = {
    "thrust": "Thrust",
    "normal": "Normal",
    "strike-slip": "Strike-Slip",
    "unknown": "Unknown",
    "m_rr": "$M_{rr}$",
    "m_tt": "$M_{\\theta\\theta}$",
    "m_pp": "$M_{\\phi\\phi}$",
    "m_rt": "$M_{r\\theta}$",
    "m_rp": "$M_{r\\phi}$",
    "m_tp": "$M_{\\theta\\phi}$",
    "depth_in_m": "z",
    "longitude": "Lon",
    "latitude": "Lat",
    "time_shift": "$T_s$",
    "gamma": "$\\gamma$",
    "lambda1": "$\\lambda_1$",
    "lambda2": "$\\lambda_2$",
    "lambda3": "$\\lambda_3$",
    "lune_gamma": "$\\gamma$",
    "lune_kappa": "$\\kappa$ St",
    "lune_theta": "$\\theta$ Dp",
    "lune_sigma": "$\\sigma$ Rk",
    "lune_M0": "$M_0$",
    "moment_magnitude": "$M_w$",
}


def cat2array(
    cat, parameters: tp.List[str], normalize: bool = True  #: CMTCatalog,
) -> np.ndarray:

    N = len(cat)
    M = len(parameters)

    A = np.zeros((N, M))

    # Get the right axes instance
    for _i, param in enumerate(parameters):

        A[:, _i] = cat.getvals(vtype=param)

        if not normalize:
            continue

        if param in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            A[:, _i] /= cat.getvals(vtype="M0")

        elif param == "depth_in_m":
            minmw = np.min(A[:, _i])
            maxmw = np.max(A[:, _i])
            A[:, _i] = (A[:, _i] - minmw) / (maxmw - minmw)

        elif param == "longitude":
            A[:, _i] = A[:, _i] / 180

        elif param == "latitude":
            A[:, _i] = A[:, _i] / 90

        elif param == "gamma":
            A[:, _i] = A[:, _i] / (np.pi / 6)

        elif param == "time_shift":

            minmw = np.min(A[:, _i])
            maxmw = np.max(A[:, _i])
            A[:, _i] = (A[:, _i] - minmw) / (maxmw - minmw)

        elif param == "moment_magnitude":
            minmw = np.min(A[:, _i])
            maxmw = np.max(A[:, _i])
            A[:, _i] = (A[:, _i] - minmw) / (maxmw - minmw)

        elif param == "lune_gamma":
            A[:, _i] /= 30

        elif param == "lune_kappa":
            A[:, _i] /= 180

        elif param == "lune_theta":
            A[:, _i] /= 90

        elif param == "lune_sigma":
            A[:, _i] /= 90

        elif param == "lune_M0":
            minm0 = np.min(A[:, _i])
            maxm0 = np.max(A[:, _i])
            A[:, _i] = (A[:, _i] - minm0) / (maxm0 - minm0)

        elif param in ["lambda1", "lambda2", "lambda3"]:
            A[:, _i] /= cat.getvals(vtype="M0")

        else:
            raise ValueError(f"Parameter {param} not recognized.")

    if normalize:
        limits = {
            "m_rr": (-1.1, 1.1),
            "m_tt": (-1.1, 1.1),
            "m_pp": (-1.1, 1.1),
            "m_rt": (-1.1, 1.1),
            "m_rp": (-1.1, 1.1),
            "m_tp": (-1.1, 1.1),
            "lambda1": (-1.1, 1.1),
            "lambda2": (-1.1, 1.1),
            "lambda3": (-1.1, 1.1),
            "depth_in_m": (0, 1),
            "time_shift": (0, 1),
            "latitude": (-1, 1),
            "longitude": (-1, 1),
            "moment_magnitude": (0, 1),
            "gamma": (-1, 1),
            "lune_gamma": (-1, 1),
            "lune_kappa": (0, 1),
            "lune_theta": (0, 1),
            "lune_sigma": (-1, 1),
            "lune_M0": (0, 1),
        }
        sublimits = OrderedDict()
        for param in parameters:
            sublimits[param] = limits[param]
    else:
        sublimits = OrderedDict()
        for _i, param in parameters:
            sublimits[param] = (np.min(A[:, _i]), np.max(A[:, _i]))

    return A, sublimits


def split_cat_mech_depth(
    cat,
    ranges: tp.Dict[str, tp.Tuple[int, int, int]] = dict(
        shallow=(0, 70, 10), intermediate=(70, 200, 20), deep=(200, 900, 100)
    ),
):
    from collections import OrderedDict

    splitcat = OrderedDict()

    for i, (_name, (dlow, dhigh, dd)) in enumerate(ranges.items()):
        # Set range for this depth and name
        if _name not in splitcat:
            splitcat[_name] = dict()
            splitcat[_name]["range"] = (dlow, dhigh, dd)

        # Filter catalog by depth
        _tcat, _ = cat.filter(
            maxdict=dict(depth_in_m=dhigh * 1000),
            mindict=dict(depth_in_m=dlow * 1000),
        )

        # Split the catalog into the different mechanisms with the null value threshold
        # set to 1.0 to include all NDC events
        splitcat[_name]["catalogs"] = _tcat.split_to_mechanism(
            thrust_null_value_threshold=1.0,
            normal_null_value_threshold=1.0,
            strike_slip_null_value_threshold=1.0,
        )

        # Sort events from shallow to deep for each catalog
        for name, _scat in splitcat[_name]["catalogs"].items():
            # Sort the catalog by depth
            _scat.sort(key="depth_in_m")

    return splitcat


def make_catalog_dc(cat):  #: CMTCatalog
    for cmt in cat:

        # Get eignevalues and eigenvectors
        lmd, ev = cmt.tnp

        # Store scalar moment
        M0 = cmt.M0

        # Init new eigenvalues
        lmd_dc = np.zeros(3)
        lmd_dc[0] = 1
        lmd_dc[1] = 0
        lmd_dc[2] = -1

        # Reorient the moment tensor
        mt = ev @ np.diag(lmd_dc) @ np.linalg.inv(ev)

        # Set moment tensor
        cmt.fulltensor = mt

        # Remove the trace
        # tr = np.trace(cmt.fulltensor)
        # cmt.fulltensor -= np.eye(3) * (tr / 3)

        # Update M0
        cmt.M0 = M0


def add_param_noise(cat, noise: dict, meanM0: float | None = None):  #: CMTCatalog,

    for cmt in cat:

        M0 = cmt.M0

        for key, val in noise.items():
            if key in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
                setattr(
                    cmt,
                    key,
                    getattr(cmt, key) + M0 * val + M0 * val * 0.025 * np.random.randn(),
                )

        # Remove the trace
        tr = np.trace(cmt.fulltensor)
        cmt.fulltensor -= np.eye(3) * (tr / 3)

        # Make sure the scalar moment remains the same.
        M0_new = cmt.M0
        cmt.M0 *= M0 / M0_new

        # print(cmt.M0, M0, M0_new)


def add_param_noise_bias(
    cat, param: str, sigma: float = 1.0, mean: float = 0.0  #: CMTCatalog,
):

    for cmt in cat:

        M0 = cmt.M0

        if param in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]:
            setattr(
                cmt,
                param,
                getattr(cmt, param) + M0 * (sigma * np.random.randn() + mean),
            )

        # Remove the trace
        tr = np.trace(cmt.fulltensor)
        cmt.fulltensor -= np.eye(3) * (tr / 3)

        # Make sure the scalar moment remains the same.
        M0_new = cmt.M0
        cmt.M0 *= M0 / M0_new

        # print(cmt.M0, M0, M0_new)


def add_corr_rt_rp(cat, fraction: float = 0.1):  #: CMTCatalog,

    N = len(cat)

    means = [2, 2]
    stds = [1 / 3, 1]
    corr = 0.0  # correlation
    covs = [
        [stds[0] ** 2, stds[0] * stds[1] * corr],
        [stds[0] * stds[1] * corr, stds[1] ** 2],
    ]

    m_rt, m_rp = np.random.multivariate_normal(means, covs, N).T

    mod = np.ones_like(m_rp)
    mod[::2] = -1
    # m_rp = m_rp * mod
    m_rt = m_rt * mod

    for _i, cmt in enumerate(cat):

        M0 = cmt.M0
        setattr(cmt, "m_rt", getattr(cmt, "m_rt") + M0 * fraction * m_rt[_i])
        setattr(cmt, "m_rp", getattr(cmt, "m_rp") + M0 * fraction * m_rp[_i])

        # Remove the trace
        tr = np.trace(cmt.fulltensor)
        cmt.fulltensor -= np.eye(3) * (tr / 3)

        cmt.M0 = M0


def add_clvd(cat, fraction: float = 0.1, positive: bool = True):  #: CMTCatalog,

    for cmt in cat:

        M0 = cmt.M0

        # Get eignevalues and eigenvectors
        lmd, ev = cmt.tnp

        # Store scalar moment
        M0 = cmt.M0

        # positive modifier
        mod = (-1) ** int(positive == False)

        # Init new eigenvalues
        lmd_clvd = np.zeros(3)
        lmd_clvd[0] = 1 * mod
        lmd_clvd[1] = 1 * mod
        lmd_clvd[2] = -2 * mod

        # Reorient the moment tensor
        mt = (
            ev
            @ np.diag(lmd + np.abs(lmd).max() * fraction * lmd_clvd)
            @ np.linalg.inv(ev)
        )

        # Set moment tensor
        cmt.fulltensor = mt

        # Update M0
        cmt.M0 = M0


def cat2meca(cat: "CMTCatalog"):
    focal_mechanisms = []

    for ev in cat:
        # Get exponent to scale moment tensor
        exp = np.ceil(np.log10(ev.M0))

        # Append the focal mechanism parameters to the list
        focal_mechanisms.append(
            [
                ev.longitude,
                ev.latitude,
                ev.depth_in_m / 1000.0,
                ev.m_rr / exp,
                ev.m_tt / exp,
                ev.m_pp / exp,
                ev.m_rt / exp,
                ev.m_rp / exp,
                ev.m_tp / exp,
                exp,
            ]
        )

    # Make numpy array
    focal_mechanisms = np.array(focal_mechanisms)

    return focal_mechanisms
