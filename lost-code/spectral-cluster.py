import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(0)

# Number of points in each ring
num_points = 100

# Radius of the inner and outer rings
inner_radius = 5
outer_radius = 10

# Generate random angles for both rings
theta_inner = np.random.uniform(0, 2 * np.pi, num_points)
theta_outer = np.random.uniform(0, 2 * np.pi, num_points)

# Generate random distances from the center for both rings
inner_distances = np.random.uniform(0, inner_radius, num_points)
outer_distances = np.random.uniform(inner_radius, outer_radius, num_points)

# Calculate x and y coordinates for the points
x_inner = inner_distances * np.cos(theta_inner)
y_inner = inner_distances * np.sin(theta_inner)
x_outer = outer_distances * np.cos(theta_outer)
y_outer = outer_distances * np.sin(theta_outer)

# Create a scatter plot to visualize the points
plt.scatter(x_inner, y_inner, color='red', label='Inner Ring (Red Points)')
plt.scatter(x_outer, y_outer, color='blue', label='Outer Ring (Blue Points)')

# Set equal aspect ratio to ensure circles appear as circles
plt.gca().set_aspect('equal', adjustable='box')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
#plt.legend()

# Show the plot
plt.show()
