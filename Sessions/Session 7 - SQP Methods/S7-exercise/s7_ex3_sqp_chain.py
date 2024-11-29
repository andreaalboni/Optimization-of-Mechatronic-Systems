"""
Solve the hanging chain problem
""" 
import numpy as np
from minimize_sqp import minimize_sqp
from plot_solution import plot_iterates, plot_convergence
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)


def ex5_sqp_chain():

    N = 20
    linear_constraints = True

    x0 = np.linspace(-1,1,N)

    # Different initial guesses for y
    # y0 = np.ones(N)
    y0 = 1+0.2*np.cos(x0*np.pi/2)

    if linear_constraints:
        gfun = gfun_linear
    else:
        gfun = gfun_quadratic

    xy0 = np.vstack((x0.reshape((N,1)), y0.reshape((N,1))))

    [xy ,_, gradients_lagrangian] = minimize_sqp(ffun, hfun, gfun, xy0, with_line_search=True, with_powells_trick=True, callback=callback)
    x = xy[:N]
    y = xy[N:]
    n_iter = gradients_lagrangian.shape[1]

    # # ------ Plot the solution ------
    # plot solution
    fig1 = plt.figure()
    plt.plot(x, y, 'bo', label='chain elements')
    plt.plot(x, y, 'r', label='chain')

    if linear_constraints:
        plt.plot(x, 0.15*x+0.3, color='k', label="linear constraint")
    else:
        plt.plot(x, -0.6*x**2 +0.15*x +0.5, color='k', label='quadratic constraint')

    title_str = "Optimal solution"
    plt.title(title_str)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # Plot convergence
    plot_convergence(n_iter, gradients_lagrangian)


def callback(x):
    N = round(x.shape[0]/2)
    plt.plot(x[0:N], x[N:],'r')
    plt.plot(x[0:N], x[N:],'bo')
    plt.title("SQP Iterations")

def ffun(xy: np.array):
    N = round(xy.shape[0]/2)
    x = xy[:N]
    y = xy[N:]
    
    raise NotImplementedError('Implement the function f')
    #V = ???

    return np.array([[V]])

def hfun(xy):
    N = round(xy.shape[0]/2)
    x = xy[:N]
    y = xy[N:]
    
    r = 1.4*2/N
    
    raise NotImplementedError('Implement the function h')
    # h = ?????

    return h

def gfun_linear(xy):
    N = round(xy.shape[0]/2)
    x = xy[:N]
    y = xy[N:]
    
    raise NotImplementedError('Implement the function g linear')
    # g = ???

    return g

def gfun_quadratic(xy):
    N = round(xy.shape[0]/2)
    x = xy[:N]
    y = xy[N:]
    
    raise NotImplementedError('Implement the function g quadratic')
    # g = ????

    return g

# ------ Run the script ------
if __name__ == '__main__':
    ex5_sqp_chain()

