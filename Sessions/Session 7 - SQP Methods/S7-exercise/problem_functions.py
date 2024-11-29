import numpy as np

# ------ Problem functions ------
# Objective function
def ffun(xy: np.array):
    x = xy[0]
    y = xy[1]    
    f = 0.5*(y/2)**2 + 0.5*x**2
    return np.array([f])

# Equality constraint function
def hfun(xy: np.array):
    x = xy[0]
    y = xy[1]    
    h = y - (3 + (x-1)**2 - x)
    return np.array([h])

# Inequality constraint function
def gfun(xy):
    # such that g(x) <= 0
    x = xy[0]
    y = xy[1]
    
    g = -2.0*x + 0.4*x**2 + y
    return np.array([g])