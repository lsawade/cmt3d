from scipy.optimize import minimize,

def rosen(x):

    x =

    c = np.sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
    return cost


x0 = [1.3, 0.7]
res = minimize(rosen, x0, method='ste', tol=1e-6)
res.x
