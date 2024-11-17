import numpy as np

def line_search(fun, x0, Jdk, dk, sigma, beta):
    # Line search using Armijo conditions and backtracking
    # fun   - scalar function
    # x0    - initial guess
    # Jdk   - J is gradient of fun at x0, Jdk is J'*dk
    # dk    - search direction
    # sigma - Armijo condition scaling function
    # beta  - backtracking parameter

    # make sure sigma and beta are in reasonable range
    if sigma <= 0 or 1 <= sigma:
        raise ValueError('sigma must be in (0,1)')

    if beta <= 0 or 1 <= beta:
        raise ValueError('beta must be in (0,1)')
    
    too_many_steps_counter = 1000

    # initialize alpha, evaluate function
    alpha = 1
    f0 = fun(x0)

    trial_x = x0 + alpha*dk
    while fun(trial_x) > f0 + sigma*alpha*Jdk:
        # trial step in x
        alpha   = beta * alpha
        trial_x = x0 + alpha*dk

        # if the line search takes too many iterations, throw an error
        too_many_steps_counter = too_many_steps_counter - 1
        if too_many_steps_counter == 0:
            raise RuntimeError('line search fail - took too many iterations')

    return trial_x, alpha
