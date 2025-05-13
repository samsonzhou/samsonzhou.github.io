import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(0)

# Number of points in each cluster
num_points = 50

# Generate random points for three clusters
cluster1_x = np.random.normal(0, 1, num_points)
cluster1_y = np.random.normal(0, 1, num_points)

cluster2_x = np.random.normal(5, 1, num_points)
cluster2_y = np.random.normal(5, 1, num_points)

cluster3_x = np.random.normal(10, 1, num_points)
cluster3_y = np.random.normal(0, 1, num_points)

# Create a scatter plot to visualize the clusters with different colors
plt.scatter(cluster1_x, cluster1_y, color='red', label='Cluster 1')
plt.scatter(cluster2_x, cluster2_y, color='blue', label='Cluster 2')
plt.scatter(cluster3_x, cluster3_y, color='green', label='Cluster 3')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
