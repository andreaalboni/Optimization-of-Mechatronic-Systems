"""
Exercise Session 7: SQP Methods

Solve the problem 

minimize        0.5*(y/2)**2 + 0.5*x**2
  x,y
subject to      y - (3 + (x-1)**2 - x) = 0

The solution to this problem is:
x^* = [0.793885, 2.2486]
lambdA^* = -0.56215

"""
import numpy as np
from matplotlib import pyplot as plt
from problem_functions import ffun, hfun
from plot_solution import plot_iterates, plot_convergence
from minimize_eq_sqp import minimize_eq_sqp

def ex5_eq_sqp(with_line_search=True, with_powells_trick=False):

    # ------ First let's plot the function ------
    # Define the the meshgrids for the 2D function
    xlist = np.linspace(-1, 4, 40)
    ylist = np.linspace(-3, 4, 40)
    [X,Y] = np.meshgrid(xlist, ylist)
    obj = np.vectorize(lambda x, y: ffun([x, y]), signature="(),()->()")(X, Y)

    # Plot contour lines
    fig = plt.figure()
    c = plt.contour(X, Y, obj, 40)
    t = np.linspace(-4,4,400)
    plt.plot(t, 3 + (t-1)**2 - t, color='black', label='constraint')

    plt.title('Contour plot of objective function with constraint')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.xlim((-1, 4))
    plt.ylim((-3, 4))
    plt.legend()
    plt.show()

    # ------ Solve the problem ------
    # Initial guess
    x0 = np.array([[3.0],[1.0]])
    # x0 = np.array([[2.5], [3.]])

    _, x, gradients_lagrangian = minimize_eq_sqp(ffun,
                                                 hfun,
                                                 x0,
                                                 with_line_search,
                                                 with_powells_trick)

    n_iter = gradients_lagrangian.shape[1]

    # ------ Plot the solution ------
    # Plot iterates x
    # Plot contour lines
    fig = plt.figure()
    c = plt.contour(X, Y, obj, 40)
    t = np.linspace(-4,4,400)
    plt.plot(t, 3 + (t-1)**2 - t, color='black', label='constraint')

    cbar = fig.colorbar(c)

    plt.plot(x[0,:], x[1,:],color='r')
    plt.plot(x[0,:], x[1,:],color='r', marker="o")
    plt.title('Contour plot of objective function with constraint')
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.xlim((-1, 4))
    plt.ylim((-3, 4))
    plt.legend()
    plt.show()

    # Plot convergence
    plot_convergence(n_iter, gradients_lagrangian)



# ------ Run the script ------
if __name__ == '__main__':
    # No linesearch, no Powell
    ex5_eq_sqp(with_line_search=False, with_powells_trick=False)

    # Only linesearch
    ex5_eq_sqp(with_line_search=True, with_powells_trick=False)

    # Only Powell's trick
    ex5_eq_sqp(with_line_search=False, with_powells_trick=True)

    # Linesearch and Powell's trick
    ex5_eq_sqp(with_line_search=True, with_powells_trick=True)