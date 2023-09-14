import numpy as np


def dev(M1, M2, M3):
    """Returns the deviatoric eigenvalues moment tensor.

    Parameters
    ----------
    M1 : arraylike
        Largest eigenvalue
    M2 : arraylike
        middle eigenvalue
    M3 : arraylike
        smallest eigenvalue

    Returns
    -------
    arraylike
        columns are the deviatoric eigenvalues

    """
    lb = np.vstack((M1, M2, M3)).T

    trace = np.ones_like(lb) * np.sum(lb, axis=1)
    if lb.size > 3:
        trace = trace.T

    dev = lb - 1/3 * trace

    return dev


def iso_clvd_dc(M1, M2, M3):
    """Return the values for decomposed moment tensor
        See Vavryčuk 2015 for a review on Moment tensor decomposition."""

    # Isotropic (Iso) part
    Miso = 1/3 * (M1 + M2 + M3)

    # compensated linear vector dipole (CLVD) part
    Mclvd = 2/3 * (M1 + M3 - 2 * M2)

    # Double Couple (DC)
    Mdc = 1/2 * (M1-M3-np.abs(M1 + M3 - 2*M2))

    return Miso, Mclvd, Mdc


def iso_clvd_dc_norm(M1, M2, M3):
    """Return the values for decomposed moment tensor
        See Vavryčuk 2015 for a review on Moment tensor decomposition."""

    # Isotropic (Iso) part
    Miso = 1/3 * (M1 + M2 + M3)

    # compensated linear vector dipole (CLVD) part
    Mclvd = 2/3 * (M1 + M3 - 2 * M2)

    # Double Couple (DC)
    Mdc = 1/2 * (M1-M3-np.abs(M1 + M3 - 2*M2))

    # Normalize
    M = np.abs(Miso) + Mdc + np.abs(Mclvd)

    return Miso/M, Mclvd/M, Mdc/M


#############################################################
# Tape & Tape 2012 - A geometric setting for moment tensors #
#############################################################

def gamma(M1, M2, M3):
    """Returns the gamma angle to distinguish between DC and CLVD."""

    return np.arctan((np.sqrt(3)*M2)/(M1-M3))

########################
# hexagonal bi‐pyramid #
########################


def u_v(M1, M2, M3):
    """Returns the u-v coordinates entry as tuple.
    (See Aso et al. 2016)"""

    # Compute u and v
    u = -2/3 * (M1 - 2*M2 + M3)/(np.max(np.vstack((M1, -M3)).T, axis=1))
    v = 1/3 * (M1 + M2 + M3)/(3 * np.max(np.vstack((M1, -M3)).T, axis=1))

    # For convenience
    if len(u) == 1:
        u = u[0]
        v = v[0]

    return u, v


def tau_k(M1, M2, M3):
    """Returns the tau-k coordinates entry as tuple.
    (See Aso et al. 2016)"""

    # Compute common denominator
    denom = 3 * (M1-M3) + np.abs(M1 - 2*M2 + M3) + 2*np.abs(M1 + M2 + M3)

    # Compute tau-k
    tau = (-4*(M1 - 2*M2 + M3))/denom
    k = 2 * (M1 + M2 + M3)/denom

    return tau, k


def T_k(M1, M2, M3):
    """Returns the tau-k coordinates entry as tuple.
    (See Aso et al. 2016)"""

    # Compute common denominator
    denom_T = 3 * (M1-M3) + np.abs(M1 - 2*M2 + M3)
    denom_k = 3 * (M1-M3) + np.abs(M1 - 2*M2 + M3) + 2*np.abs(M1 + M2 + M3)

    # Compute tau-k
    T = (-4*(M1 - 2*M2 + M3))/denom_T
    k = 2 * (M1 + M2 + M3)/denom_k

    return T, k


def eps_nu(M1, M2, M3):
    """Returns the epsilon value for a given moment tensor.
    (See Aso et al. 2016)"""

    # Compute epsilon nu
    epsilon = (-2*(M1-2*M2+M3)) / (3*(M1-M3) + np.abs(M1-2*M2+M3))

    # Compute nu
    nu = (M1 + M2 + M3)/(3 * np.max(np.vstack((M1, -M3)).T, axis=1))

    if isinstance(epsilon, float):
        nu = nu[0]

    return epsilon, nu
