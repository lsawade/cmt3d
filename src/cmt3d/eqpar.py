import numpy as np


def eqpar(fmomin):
    '''
    Parameters
    ----------
    fmomin(6)
        input moment tensor components

    Returns
    -------
    Tuple of
        scmom       output scalar moment
        phs(2)      output strike azimuths
        dl(2)       output dips
        rlam(2)     output 'rake' angles
        eivals(3)   output eigenvalues (t-axis,inter.,p-axis)
        eivecs(3,3) output: columns are normalized principal axes
        plungs(3)   output plunges of principal axes [T,B,P]
        azims(3)    output azimuths of principal axes [T,B,P]

    Notes
    -----

    Legacy code translated form fortran, needs a thorough clean up
    All angles are in degrees!


    '''

    fmom = np.zeros(6)
    fmom[:] = fmomin
    phs = np.zeros(2)
    dl = np.zeros(2)
    rlam = np.zeros(2)
    eivals = np.zeros(3)
    eivecs = np.zeros((3, 3))
    v = np.zeros((3, 3))
    rn = np.zeros(3)
    e = np.zeros(3)
    plungs = np.zeros(3)
    azims = np.zeros(3)

    # Conversions
    rad2deg = 180/np.pi
    deg2rad = 1/rad2deg

    hsq2 = (0.5*np.sqrt(2.0))

    scale = 0.
    scale = np.max(np.abs(fmom))

    if scale == 0:
        raise ValueError('Moment Tensor not valid')

    # Setup full moment tensor
    eivecs[0, 0] = fmom[0]/scale
    eivecs[1, 1] = fmom[1]/scale
    eivecs[2, 2] = fmom[2]/scale
    eivecs[0, 1] = fmom[3]/scale
    eivecs[1, 0] = fmom[3]/scale
    eivecs[0, 2] = fmom[4]/scale
    eivecs[2, 0] = fmom[4]/scale
    eivecs[1, 2] = fmom[5]/scale
    eivecs[2, 1] = fmom[5]/scale

    # Get Eigen values and vectors
    w, v = np.linalg.eigh(eivecs)

    # Manually get the imxa and imin because the eigenvalues are sorted
    imax = 2
    iint = 1
    imin = 0
    eimax = w[imax]
    eiint = w[1]
    eimin = w[imin]

    # Get scaled scalar moment
    scmom = 0.5 * (np.abs(eimax) + np.abs(eimin))

    # Create separate eigenvalue array
    eivals[0] = eimax
    eivals[1] = eiint
    eivals[2] = eimin

    # Reassign eigenvectors to matrix
    for i in range(3):
      eivecs[i, 0] = v[i, imax]
      eivecs[i, 1] = v[i, iint]
      eivecs[i, 2] = v[i, imin]
    
    # Normalize the eigenvectors
    for j in range(3):
        sum = 0.
        for i in range(3):
            sum = sum + eivecs[i, j] * eivecs[i, j]
        sum = 1./np.sqrt(sum)

        # This seems like a weird thing
        # Given the Setup, I'd a assum this makes no sense...
        if (eivecs[0, j] > 0.0):
            sum = -sum

        for i in range(3):
            eivecs[i, j] = eivecs[i, j]*sum

    # Get plunge and azimuth
    for i in range(3):

        # Plunge
        plungs[i] = np.arctan2(
            -eivecs[0, i],
            np.sqrt(eivecs[1, i]**2 + eivecs[2, i]**2)) * rad2deg

        # Azimuth
        azims[i] = np.arctan2(
            -eivecs[2, i],
            eivecs[1, i]) * rad2deg + 180.0

    # Loop to get the D and Rake
    sgn = -1.
    for isgn in range(2):
        sgn = -sgn
        for i in range(3):
            rn[i] = hsq2 * (eivecs[i, 0] + sgn*eivecs[i, 2])
            e[i] = hsq2 * (eivecs[i, 0] - sgn*eivecs[i, 2])

        # Flip sign if necessary
        if (rn[0] < 0.):
            for i in range(3):
                rn[i] = -rn[i]
                e[i] = -e[i]

        sind = np.sqrt(rn[1]**2 + rn[2]**2)

        dl[isgn] = np.arctan2(sind, rn[0]) * rad2deg
        phs[isgn] = np.arctan2(-rn[1], -rn[2]) * rad2deg + 180.0
        rlam[isgn] = np.arctan2(e[0], rn[1]*e[2]-rn[2]*e[1]) * rad2deg

    for i in range(3):
        eivals[i] = eivals[i] * scale

    # Scale the scalar moment agin
    scmom = scmom * scale

    return scmom, phs, dl, rlam, eivals, eivecs, plungs, azims
