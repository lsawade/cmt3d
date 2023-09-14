import numpy as np


def update_alpha(w1, w2, alpha_l, alpha_r, alpha, factor=10.0):

    # not a sufficient decrease, we've been too far
    if w1 is False:
        alpha_r = alpha
        alpha = (alpha_l+alpha_r)*0.5

    elif (w1 is True) and (w2 is False):
        alpha_l = alpha
        # sufficient decrease but too close already backeted decrease in interval
        if alpha_r > 0:
            alpha = (alpha_l + alpha_r) * 0.5
        # sufficient decrease but too close, then increase a
        else:
            alpha = factor * alpha

    return alpha_l, alpha_r, alpha


def wolfe_conditions(
        q,
        qnew,
        cost,
        cost_new,
        alpha,
        c1: float = 1e-4,
        c2: float = 0.9,
        strong: bool = False):
    """Checks Wolfe conditions

    Parameters
    ----------
    q : float
        q
    qnew : float
        qnew
    cost : float
        cost
    cost_new : float
        cost new
    alpha : float
        alpha
    c1 : float, optional
        c1, by default 1e-4
    c2 : float, optional
        c2, by default 0.9
    strong : bool, optional
        strong wolfe condition, by default False

    Returns
    -------
    tuple of bools
        w1,w2,w3
    """

    # Init wolfe boolean
    w1 = False
    w2 = False
    w3 = True

    # Check descent direction
    if q > 0:
        w3 = False

    # Check first wolfe
    if cost_new <= cost + c1 * alpha * q:
        w1 = True

    # Check second wolfe
    if strong is False:
        if qnew >= c2 * q:
            w2 = True
    else:
        if np.abs(qnew) >= np.abs(c2 * q):
            w2 = True

    return w1, w2, w3
