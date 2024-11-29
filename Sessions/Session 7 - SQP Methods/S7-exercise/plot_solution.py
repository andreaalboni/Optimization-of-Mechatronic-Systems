import numpy as np
from matplotlib import pyplot as plt

def plot_iterates(x: np.array):
    plt.figure()
    plt.plot(x[0,:], x[1,:],color='r')
    plt.plot(x[0,:], x[1,:],color='r', marker="o")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()

def plot_convergence(n_iter, gradients_lagrangian):
    plt.figure()
    plt.semilogy(np.arange(n_iter), gradients_lagrangian.squeeze(), marker='.')
    plt.grid()
    plt.title('Gradient norm iterations')
    plt.xlabel('iterations')
    plt.ylabel('$\\Vert \\nabla L(x_k, \\lambda_k)\\Vert_\\infty$')
    plt.show()