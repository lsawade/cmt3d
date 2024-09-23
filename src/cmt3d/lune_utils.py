import numpy as np
from scipy.spatial.transform import Rotation as R


def get_lune_coordinates(tensor: np.ndarray):

    # Move from up-south-east to south-east-up
    m11 = tensor[1]
    m22 = tensor[2]
    m33 = tensor[0]
    m12 = tensor[5]
    m13 = tensor[3]
    m23 = tensor[4]

    # Get eigenvalues and eigenvectors
    fulltensor = np.array([[m11, m12, m13], [m12, m22, m23], [m13, m23, m33]])

    # Get eigenvalues and eigenvectors
    (lb, ev) = np.linalg.eigh(fulltensor)

    lb = lb[::-1]
    ev = ev[:, ::-1]

    # Get Lune values from eigenvalues
    gamma, delta, M0, thetadc, lamdev, lamiso = lam2lune(lb)

    # Get moment tensor orientation
    kappa, theta, sigma, K, N, S = U2sdr(ev)

    return gamma, delta, M0, thetadc, lamdev, lamiso, kappa, theta, sigma, K, N, S


def NS2sdr(N, S, bdisplay=False, bSproj=False):
    """Converts normal and slip vectors to strike, dip, and rake.

    Parameters
    ----------
    N : arraylike
        normal vector
    S : arraylike
        slip vector
    bdisplay : bool, optional
        display results, by default False

    Returns
    -------
    arraylike
        strike, dip, rake, K, N, S
    """

    EPSVAL = 1e-6

    # ~1 -> 1, ~0 -> 0, ~-1 -> -1
    N = setzero(N)
    S = setzero(S)

    # 4 possible compbinations of normal and strike vectors
    S1 = S
    N1 = N
    S2 = -S
    N2 = -N
    S3 = N
    N3 = S
    S4 = -N
    N4 = -S

    theta1, sigma1, kappa1, K1, sigmaproj1 = faultvec2ang(S1, N1, bSproj)
    theta2, sigma2, kappa2, K2, sigmaproj2 = faultvec2ang(S2, N2, bSproj)
    theta3, sigma3, kappa3, K3, sigmaproj3 = faultvec2ang(S3, N3, bSproj)
    theta4, sigma4, kappa4, K4, sigmaproj4 = faultvec2ang(S4, N4, bSproj)

    # There are four combinations of N and S that represent a double couple
    # moment tensor, as shown in Figure 15 of TT2012beach.
    # From these four combinations, there are two possible fault planes.
    # We want to isolate the combination that is within the bounding
    # region shown in Figures 16 and B1.
    thetaall = np.array([theta1, theta2, theta3, theta4])
    sigmaall = np.array([sigma1, sigma2, sigma3, sigma4])
    kappaall = np.array([kappa1, kappa2, kappa3, kappa4])
    btheta = thetaall <= 90 + EPSVAL
    # dip angles
    bsigma = abs(sigmaall) <= 90 + EPSVAL
    # rake angle
    bmatch = btheta & bsigma

    simgaprojall = np.array([sigmaproj1, sigmaproj2, sigmaproj3, sigmaproj4])

    itemp = np.where(bmatch)[0]
    match len(itemp):
        case 0:
            raise ValueError("No match found")
        case 1:
            imatch = itemp[0]
        case 2:
            # choose one of the two
            i1 = itemp[0]
            i2 = itemp[1]
            ipick = pickP1(
                thetaall[i1],
                sigmaall[i1],
                kappaall[i1],
                thetaall[i2],
                sigmaall[i2],
                kappaall[i2],
            )
            imatch = itemp[ipick]

        case 3:
            # this is a more unusual case, like for horizontal faults
            print(
                f"moment tensor on boundary of orientation domain ({len(itemp)} candidates)."
            )
            imatch = itemp[0]

        case 4:
            raise ValueError("All combinations match. This should not happen.")

    # Finally get fault vectors
    K = np.zeros(3)
    K[:] = np.nan
    N = np.zeros(3)
    N[:] = np.nan
    S = np.zeros(3)
    S[:] = np.nan
    kappa = np.nan
    theta = np.nan
    sigma = np.nan
    sigmaproj = np.nan

    kk = imatch
    match kk:
        case 0:
            K[:] = K1
            N[:] = N1
            S[:] = S1
            kappa = kappa1
            theta = theta1
            sigma = sigma1
            sigmaproj = sigmaproj1
        case 1:
            K[:] = K2
            N[:] = N2
            S[:] = S2
            kappa = kappa2
            theta = theta2
            sigma = sigma2
            sigmaproj = sigmaproj2
        case 2:
            K[:] = K3
            N[:] = N3
            S[:] = S3
            kappa = kappa3
            theta = theta3
            sigma = sigma3
            sigmaproj = sigmaproj3
        case 3:
            K[:] = K4
            N[:] = N4
            S[:] = S4
            kappa = kappa4
            theta = theta4
            sigma = sigma4
            sigmaproj = sigmaproj4

    return kappa, theta, sigma, K, N, S, sigmaproj


def pickP1(thetaA, sigmaA, kappaA, thetaB, sigmaB, kappaB):
    # choose between two moment tensor orientations based on Figure B1 of TT2012beach
    # NOTE THAT NOT ALL FEATURES OF FIGURE B1 ARE IMPLEMENTED HERE
    EPSVAL = 1e-6

    # these choices are based on the strike angle
    if abs(thetaA - 90) < EPSVAL:
        ipick = np.where(np.array([kappaA, kappaB]) < 180)[0]

    if abs(sigmaA - 90) < EPSVAL:
        ipick = np.where(np.array([kappaA, kappaB]) < 180)[0]

    if abs(sigmaA + 90) < EPSVAL:
        ipick = np.where(np.array([kappaA, kappaB]) < 180)[0]

    if ipick not in locals.keys() or len(ipick) == 0:
        print(thetaA, sigmaA, kappaA)
        print(thetaB, sigmaB, kappaB)
        raise ValueError("ipick is empty")


def faultvec2ang(S, N, bdisplay=False, bSproj=False):
    # returns fault angles in degrees, assumes input vectors in south-east-up basis

    rad2deg = 180 / np.pi

    # for north-west-up basis (as in TT2012beach)
    # zenith = [0 0 1]'; north  = [1 0 0]';

    # for up-south-east basis (GCMT)
    # zenith = [1 0 0]'; north  = [0 -1 0]';

    # for south-east-up basis (as in TT2012beach)
    zenith = np.array([0, 0, 1])
    north = np.array([-1, 0, 0])

    kappa = np.nan
    theta = np.nan
    sigma = np.nan
    K = np.zeros(3)
    K[:] = np.nan

    sigmaproj = np.nan
    if bSproj:
        Splane = np.zeros(3)
        Splane[:] = np.nan
        Sperp = np.zeros(3)
        Sperp[:] = np.nan

    # strike vector from TT2012beach Eq. 29
    v = np.cross(zenith, N)
    if np.linalg.norm(v) == 0:
        # TT2012beach Appendix B
        print("horizontal fault -- strike vector is same as slip vector")
        K[:] = S[:]
    else:
        K[:] = v / np.linalg.norm(v)

    # TT2012beach Figure 14
    kappa = fangle_signed(north, K, -zenith)

    # TT2012beach Figure 14
    costh = np.dot(N, zenith)
    theta = np.arccos(costh) * rad2deg

    # TT2012beach Figure 14
    sigma = fangle_signed(K, S, N)

    if bSproj:
        # see TT2013 Figure 16
        # projection of slip vector onto fault plane
        Sperp[:] = np.dot(S, N) / np.dot(N, N) * N
        Splane[:] = S - Sperp
        sigmaproj = fangle_signed(K, Splane, N)

    kappa = wrap360(kappa)

    return theta, sigma, kappa, K, sigmaproj


def wrap360(x):
    # WRAP360 wraps angles to 0-360 degrees
    x = np.mod(x, 360)
    return x


def fangle(va, vb):

    ma = np.linalg.norm(va)
    mb = np.linalg.norm(vb)
    vadvb = np.sum(va * vb)
    theta = 180 / np.pi * np.arccos(vadvb / (ma * mb))

    return theta


def fangle_signed(va, vb, vnor):
    # FANGLE_SINGED returns the signed angle (of rotation) between two vectors

    # Get rotation angle
    theta = fangle(va, vb)

    # Initialize to rotation angle
    stheta = theta

    if np.isclose(np.abs(theta - 180), 0):
        stheta = 180
    else:
        Dmat = np.vstack([va, vb, vnor]).T
        if np.linalg.det(Dmat) < 0:
            stheta = -theta

    return stheta


def setzero(x):
    """Set values close to zero to zero. Set values close to 1 to 1. Set values close to -1 to -1."""
    x = np.where(np.isclose(x, 0), 0, x)
    x = np.where(np.isclose(x, 1), 1, x)
    x = np.where(np.isclose(x, -1), -1, x)
    return x


@staticmethod
def U2sdr(U, bdisplay=False):

    # Get rotation matrix around the Y axis
    Yrot = R.from_euler("y", 45, degrees=True).as_matrix()

    # Compute candidate fault vectors
    S = np.zeros(3)
    N = np.zeros(3)

    V = U @ Yrot  # V = U * Yrot (TT2012beach p. 487)
    S = V[:, 0]  # fault slip vector
    N = V[:, 2]  # fault normal vector

    # Compute the strike dip and rake from the candiate fault vectors
    kappa, theta, sigma, K, N, S, sigmaproj = NS2sdr(N, S)

    return kappa, theta, sigma, K, N, S


@staticmethod
def lam2lune(lb: np.ndarray):
    """Expects sorted eigenvalues"""

    # Split values
    lam1, lam2, lam3 = lb

    # Convert radians to degrees
    rad2deg = 180 / np.pi

    # Get rho
    rho = np.sqrt(lam1**2 + lam2**2 + lam3**2)

    # Compute scalar moment from rho
    M0 = rho / np.sqrt(2)

    # Compute beta
    cos_beta = (lam1 + lam2 + lam3) / (np.sqrt(3) * rho)
    cos_beta = np.clip(cos_beta, -1, 1)

    # Compute lune delta
    delta = 0

    if np.isclose(np.sum(lb), 0):
        delta = 0
    else:
        delta = (np.pi / 2 - np.arccos(cos_beta)) * rad2deg

    # Compute lune gamma
    if np.isclose(np.abs(lam1 - lam3), 0) == False:
        tan_gamma = (-lam1 + 2 * lam2 - lam3) / (np.sqrt(3) * (lam1 - lam3))
        gamma = np.arctan2(tan_gamma, 1) * rad2deg
    else:
        gamma = 0

    # thetadc
    thetadc = np.arccos((lam1 - lam3) / (np.sqrt(2) * rho)) * rad2deg

    # lamiso
    lamiso = (lam1 + lam2 + lam3) / 3

    # lamdev
    lamdev = lb - lamiso

    return gamma, delta, M0, thetadc, lamdev, lamiso
