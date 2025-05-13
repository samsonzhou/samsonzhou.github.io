import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate sample data (you can replace this with your own data)
data = np.random.rand(10, 2)  # 10 data points with 2 features each

# Perform hierarchical clustering
linkage_matrix = linkage(data, method='ward')  # You can choose a different linkage method

# Create a dendrogram
plt.figure(figsize=(8, 6))
dendrogram(linkage_matrix, labels=range(len(data)), leaf_rotation=90, leaf_font_size=12)
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()
