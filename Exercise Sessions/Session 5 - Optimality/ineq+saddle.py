import numpy as np
import matplotlib.pyplot as plt

def funct(x, y):
    return 8*x + 12*y + x**2 - 2*y**2

# Define the saddle point
saddle_point = (-4, 3)

# Create a grid of points
x = np.linspace(saddle_point[0] - 4, saddle_point[0] + 4, 400)
y = np.linspace(saddle_point[1] - 4, saddle_point[1] + 4, 400)
X, Y = np.meshgrid(x, y)

# Evaluate the function on the grid
Z = funct(X, Y)

# Plot the filled contour lines
plt.figure(figsize=(8, 6))
contourf = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contourf)
plt.title('Filled Contour Plot of the Function')
plt.xlabel('x')
plt.ylabel('y')

# Mark the saddle point
plt.plot(saddle_point[0], saddle_point[1], 'ro')  # Red dot at the saddle point
plt.text(saddle_point[0], saddle_point[1], '  Saddle Point', color='red')

# Set the limits of the plot to center around the saddle point
plt.xlim(saddle_point[0] - 4, saddle_point[0] + 4)
plt.ylim(saddle_point[1] - 4, saddle_point[1] + 4)

plt.show()



# Define the inequalities
def inequality1(x, y):
    return -y

def inequality2(x, y):
    return x**5 + y

# Create a grid of points
x = np.linspace(-4, 4, 400)
y = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x, y)

# Evaluate the inequalities on the grid
Z1 = inequality1(X, Y)
Z2 = inequality2(X, Y)

# Create a mask for the region where both inequalities are satisfied
region = (Z1 <= 0) & (Z2 <= 0)

# Plot the region
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, region, levels=1, colors=['lightblue'], alpha=0.5)
plt.contour(X, Y, Z1, levels=[0], colors='blue', linestyles='dashed')
plt.contour(X, Y, Z2, levels=[0], colors='green', linestyles='dashed')
plt.title('Region Where Both Inequalities Are Satisfied')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()