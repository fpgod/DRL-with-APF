import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def artificial_potential_field(q_goal, q_obs, eta, rho, epsilon):
    def attractive_potential(q, q_goal, eta):
        return 0.5 * eta * np.linalg.norm(q - q_goal)**2

    def repulsive_potential(q, q_obs, rho, epsilon):
        repulsive_pot = 0
        for q_i in q_obs:
            dist = np.linalg.norm(q - q_i)
            if dist < epsilon:
                repulsive_pot += 0.5 * rho * (1 / dist - 1 / epsilon)**2
        return repulsive_pot

    def total_potential(q, q_goal, q_obs, eta, rho, epsilon):
        return attractive_potential(q, q_goal, eta) + repulsive_potential(q, q_obs, rho, epsilon)

    grid_size = 100
    x = np.linspace(-10, 10, grid_size)
    y = np.linspace(-10, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(grid_size):
        for j in range(grid_size):
            q = np.array([X[i, j], Y[i, j]])
            Z[i, j] = total_potential(q, q_goal, q_obs, eta, rho, epsilon)

    return Z

# Example usage:
q_goal = np.array([5, 5])  # Goal position
q_obs = np.array([[2, 2], [3, 6], [7, 8]])  # Obstacle positions
eta = 1.0  # Attractive potential scaling factor
rho = 1.0  # Repulsive potential scaling factor
epsilon = 1.0  # Distance threshold for repulsive potential

# Create a grid for smoother visualization
grid_size = 100  # Adjusted grid size to 64
x = np.linspace(-10, 10, grid_size)
y = np.linspace(-10, 10, grid_size)
X, Y = np.meshgrid(x, y)

# Calculate potential field and use interpolation for smoother visualization
Z = artificial_potential_field(q_goal, q_obs, eta, rho, epsilon)

# Flatten X, Y, and Z arrays to match the number of data points
X_flat = X.ravel()
Y_flat = Y.ravel()
Z_flat = Z.ravel()

# Create a new grid for interpolation
grid_size_new = 100  # Increase grid size for smoother interpolation
x_new = np.linspace(-10, 10, grid_size_new)
y_new = np.linspace(-10, 10, grid_size_new)
X_new, Y_new = np.meshgrid(x_new, y_new)

# Use interpolation to create a smoother Z array
Z_interpolated = griddata((X_flat, Y_flat), Z_flat, (X_new, Y_new), method='cubic')

# Plot the potential field in 3D with smoother visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot surface plot
surf = ax.plot_surface(X_new, Y_new, Z_interpolated, cmap='plasma', rstride=1, cstride=1, linewidth=0, antialiased=False)

# Add colorbar
fig.colorbar(surf, label='Potential')

# Add coordinate axes
arrow_length = 5  # Adjust arrow length based on the grid size
ax.quiver(0, 0, 0, arrow_length, 0, 0, color='red', label='X-axis', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, arrow_length, 0, color='green', label='Y-axis', arrow_length_ratio=0.1)
ax.quiver(0, 0, 0, 0, 0, arrow_length, color='blue', label='Z-axis', arrow_length_ratio=0.1)

# Add XY axes lines
ax.plot([-10, 10], [0, 0], [0, 0], color='black', linestyle='dotted', linewidth=1.0)
ax.plot([0, 0], [-10, 10], [0, 0], color='black', linestyle='dotted', linewidth=1.0)


# Add obstacles and goal
ax.scatter(q_obs[:, 0], q_obs[:, 1], 0, c='red', marker='o', s=100, label='Obstacles')
ax.scatter(q_goal[0], q_goal[1], 0, c='green', marker='x', s=100, label='Goal')

# Set title
ax.set_title('Smoothed 3D Artificial Potential Field')

# Add legend
ax.legend()

# # Set custom viewing angle (elev, azim)
# ax.view_init(elev=30, azim=45)

# Show grid lines with custom color and linestyle
ax.grid(True, color='black', linestyle='dotted', linewidth=0.5)

# Show the 3D plot
plt.show()