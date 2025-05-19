import numpy as np
import matplotlib.pyplot as plt

#seed was 7
np.random.seed(7)

def lloyds_algorithm(points, k, n_steps):
    """
    Run Lloyd's algorithm for k-means clustering.

    Input:
    points (np.ndarray):
    k (int): Number of clusters.
    n_steps (int): Number of iterations to run the algorithm.

    Output:
    centroids (np.ndarray): Final centroids after n_steps.
    labels (np.ndarray): Array of labels for each point.
    """
    # Randomly initialize the centroids by selecting k random points from the dataset
    #centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    centroids = np.array([[1,0],[0,1],[10,10]])
    
    for _ in range(n_steps):
        # Step 1: Assign points to the nearest centroid
        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Step 2: Update centroids
        new_centroids = np.array([points[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids

    return centroids, labels

def points_regroup(points, labels):
    out_1 = []
    out_2 = []
    out_3 = []
    for i in range(len(points)):
        if labels[i] == 0:
            out_1.append(points[i])
        elif labels[i] == 1:
            out_2.append(points[i])
        else:
            out_3.append(points[i])
    return (out_1,out_2,out_3)

def random_unit_vectors():
    # Generate a random angle in radians
    # theta = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi/10)
    
    # Calculate the unit vector components
    x = np.cos(theta)
    y = np.sin(theta)
    
    return (np.array([x, y]), np.array([-y, x]))

def subpoints(points, labels):
    out = []
    for i in range(len(points)):
        if labels[i] == 0 or labels[i] == 1:
            out.append(points[i])
    return out

#Count for the first n_1 coordinates that are 0,
#the next n_2 coordinates that are 1,
#and the next n_3 coordinates that are 2
def score(arr, n_1, n_2, n_3):
    count = 0
    i = 0
    while(i < n_1):
        if arr[i] == 0:
            count += 1
        i += 1
    while(i < n_1 + n_2):
        if arr[i] == 1:
            count += 1
        i += 1
    while(i < n_1 + n_2 + n_3):
        if arr[i] == 2:
            count += 1
        i += 1
    return count

# Parameters
#n = 10000
allNs = [5000,10000,15000,20000,25000]
p = 1/3

iters = 100

lloyds_avgs = []
our_avgs = []
ratio_avgs = []

for n in allNs:

    lloyds_scores =[]
    our_scores = []

    for numIter in range(iters):

        n_1 = np.sum(np.random.binomial(1,p,n))
        n_2 = np.sum(np.random.binomial(1,p,n-n_1))
        n_3 = n - n_1 - n_2

        mean_1 = [0, 0]  # Mean of the distribution 1
        mean_2 = mean_1  # Mean of the distribution 2
        cov_1 = [[1, 0], [0, 8]]  # Covariance of matrix 1
        cov_2 = [[8, 0], [0, 1]]  # Covariance of matrix 2

        mean_3 = [10,10] # Mean of the distribution 3
        cov_3 = cov_1  # Covariance of matrix 3


        # Generate points
        in_points_1 = np.random.multivariate_normal(mean_1, cov_1, n_1)
        in_points_2 = np.random.multivariate_normal(mean_2, cov_2, n_2)
        in_points_3 = np.random.multivariate_normal(mean_3, cov_3, n_3)

        points = np.concatenate((in_points_1,in_points_2,in_points_3))
        out = lloyds_algorithm(points, 3, 100)

        list_points = subpoints(points, out[1])
        new_labels = []

        (u,v) = random_unit_vectors()

        for point in list_points:
            if abs(np.dot(point,u)) > abs(np.dot(point,v)):
                new_labels.append(1)
            else:
                new_labels.append(0)

        new_labels = np.concatenate((new_labels,[2]*n_3))

        lloyds_scores.append(score(out[1],n_1,n_2,n_3))
        our_scores.append(n_3+score(new_labels,n_1,n_2,0))

    lloyds_avgs.append(np.mean(lloyds_scores))
    our_avgs.append(np.mean(our_scores))
    ratio_avgs.append(np.mean(our_scores)/np.mean(lloyds_scores))

print(lloyds_avgs)
print(our_avgs)
print(ratio_avgs)

out_points = points_regroup(points, new_labels)
#lloyds_points = points_regroup(points, out[1])
#out_points = points_regroup(points, out[1])

points_1 = np.array(out_points[0])
points_2 = np.array(out_points[1])
points_3 = np.array(out_points[2])

# Plot points_1 in red with 'o' markers
plt.scatter(points_1[:, 0], points_1[:, 1], color='red', marker='o', s=0.1)
# Plot points_2 in blue with 'o' markers
plt.scatter(points_2[:, 0], points_2[:, 1], color='blue', marker='o', s=0.1)
# Plot points_3 in blue with 'o' markers
plt.scatter(points_3[:, 0], points_3[:, 1], color='green', marker='o', s=0.1)

# Plot the points
plt.title("Clustering by our Algorithm")
# Set x-axis limits
plt.xlim(-10, 15)  # Set x-axis range from 0 to 10
# Set y-axis limits
plt.ylim(-10, 20)  # Set y-axis range from -1 to 1
#plt.title("Clustering by Lloyd's Algorithm")
plt.xlabel("X")
plt.ylabel("Y")
#plt.axis('equal')
plt.show()
