import os
# import jax.numpy as np
import numpy as np


def g(m, X):
    """Generates arbitrary gaussian distribution for the corresponding arrays
    x and, at expected value (xo, yo), and standard deviations
    (sigma_x, sigma_y). To ensure a possibilty of angled distributions
    and offset distributions model parameters theta and offset are included.

    Args:
        x (np.ndarray):
            x coordinates
        y (np.ndarray):
            y coordinates
        amplitude (float):
            amplitude of the Gaussian function
        xo (float):
            center in x direction
        yo (float):
            center in y direction
        sigma_x ():
            std in x direction
        sigma_y (float):
            std in y direction
        theta (float):
            angle of the distribution
        offset (float):
            offset of the distribution (could be gaussian + offset)

    Returns:
        G(x, y) as np.ndarray

    Last modified: Lucas Sawade, 2020.09.15 19.44 (lsawade@princeton.edu)
    """
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = m
    x, y = X

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    ga = offset + amplitude * \
        np.exp(-(a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))

    return ga


def dgdm(m, X, pn):
    """Generates arbitrary the gradient of the gaussian distribution for the 
    corresponding arrays x,y at expected value (xo, yo), and standard deviations
    (sigma_x, sigma_y), with respect to the model parameters. 
    To ensure a possibilty of angled distributions
    and offset distributions model parameters theta and offset are included.

    Args:
        x (np.ndarray):
            x coordinates
        y (np.ndarray):
            y coordinates
        amplitude (float):
            amplitude of the Gaussian function
        xo (float):
            center in x direction
        yo (float):
            center in y direction
        sigma_x ():
            std in x direction
        sigma_y (float):
            std in y direction
        theta (float):
            angle of the distribution
        offset (float):
            offset of the distribution (could be gaussian + offset)


    Returns:
        G(x, y) as np.ndarray

    Last modified: Lucas Sawade, 2020.09.15 19.44 (lsawade@princeton.edu)

    For abbreviation in partial deriv. variable naming
    Amplitude = A
    Offset = C
    sigma_x = sigx
    sigma_y = sigy

    """
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = m
    x, y = X

    # Evaluation of the Gaussian without offset and with amplitude 1
    eG = g(np.array([1, xo, yo, sigma_x, sigma_y, theta, 0]), (x, y))

    # Simplification
    dx = x-xo
    dy = y-yo

    # Gradient at each point (x,y) and for each model parameter
    dg_dm = np.zeros((*x.shape, 7))

    # Coefficients
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)

    # Partial derivates of the coefficients
    da_dsigx = -np.cos(theta)**2/sigma_x**3
    da_dsigy = -np.sin(theta)**2/sigma_y**3

    db_dsigx = np.sin(2*theta)/(2*sigma_x**3)
    db_dsigy = -np.sin(2*theta)/(2*sigma_y**3)

    dc_dsigx = -np.sin(theta)**2/sigma_x**3
    dc_dsigy = -np.cos(theta)**2/sigma_y**3

    da_dtheta = \
        - np.sin(theta)*np.cos(theta)/sigma_x**2 \
        + np.sin(theta)*np.cos(theta)/sigma_y**2
    db_dtheta = \
        - np.cos(2*theta)/(2*sigma_x**2) \
        + np.cos(2*theta)/(2*sigma_y**2)
    dc_dtheta = \
        np.sin(theta)*np.cos(theta)/sigma_x**2 \
        - np.sin(theta)*np.cos(theta)/sigma_y**2

    # Full partial derivatives of the gaussian
    # Amplitude
    dg_dm[..., 0] = eG
    # xo
    dg_dm[..., 1] = amplitude * eG * -(-2*a*dx - 2*b*dy)
    # yo
    dg_dm[..., 2] = amplitude * eG * -(-2*b*dx - 2*c*dy)
    # sigma_x
    dg_dm[..., 3] = amplitude * eG * \
        -(dx**2 * da_dsigx + 2*dx*dy*db_dsigx + dy**2 * dc_dsigx)
    # sigma_y
    dg_dm[..., 4] = amplitude * eG * \
        -(dx**2 * da_dsigy + 2*dx*dy*db_dsigy + dy**2 * dc_dsigy)
    # theta
    dg_dm[..., 5] = amplitude * eG * \
        -(dx**2 * da_dtheta + 2*dx*dy*db_dtheta + dy**2 * dc_dtheta)
    # Offset
    dg_dm[..., 6] = np.ones_like(x)

    if pn is not None:
        return dg_dm[..., pn]
    else:
        return dg_dm
