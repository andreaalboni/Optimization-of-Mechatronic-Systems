import numpy as np

def merit_function_equality_constrained_sqp(ffun, hfun, c, x):
    f = ffun(x)
    h = hfun(x)    
    ret = f + c*np.linalg.norm(h, 1) #Calculates the 1-norm
    return ret

def merit_function_inequality_constrained_sqp(ffun, hfun, gfun, c, x):

    f = ffun(x)    
    h = hfun(x)
    g = gfun(x)
    
    ret = f + c* (np.linalg.norm(h, 1) + np.linalg.norm(np.fmax(g,0),1))

    return ret