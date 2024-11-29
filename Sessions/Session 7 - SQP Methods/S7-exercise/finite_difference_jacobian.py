import numpy as np

def finite_difference_jacobian(fun, x0: np.array):
    # make sure x0 is a column vector
    Nx,cols = x0.shape
    if cols != 1:
        raise ValueError('x0 needs to be a column vector')

    # make sure fun returns a column vector
    f0 = fun(x0)
    Nf,cols = f0.shape
    if cols != 1:
        raise ValueError('fun needs to return a column vector');

    # initialize empty J
    J = np.zeros((Nf, Nx))

    # perform the finite difference jacobian evaluation
    h = 1e-6
    for k in range(Nx):
        x = x0.copy()
        x[k] = x[k] + h
        f = fun(x)
        grad = (f - f0)/h
        J[:,[k]] = grad

    return f0, J
