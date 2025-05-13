import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sample_uniform(S, k):
    """
    Sample k points uniformly at random from S (with replacement).

    Parameters:
        S (list or np.ndarray): Candidate points, shape (n, d)
        k (int): Number of points to sample

    Returns:
        list: k sampled points from S
    """
    S = np.array(S)

    if len(S) == 0 or k == 0:
        return []

    # Assign equal probability to each point
    probs = np.ones(len(S)) / len(S)

    # Sample k points from S with replacement according to uniform distribution
    indices = np.random.choice(len(S), size=k, replace=True, p=probs)
    return S[indices].tolist()

def sample_by_distance(S, C, k):
    """
    Sample k points from S with probability proportional to distance from C.
    
    Parameters:
        S (list or np.ndarray): Candidate points, shape (n, d)
        C (list or np.ndarray): Center points, shape (m, d)
        k (int): Number of points to sample

    Returns:
        list: k sampled points from S
    """
    S = np.array(S)
    C = np.array(C)

    if len(S) == 0 or k == 0:
        return []

    if len(C) == 0:
        # If no centers, sample uniformly
        probs = np.ones(len(S))
    else:
        # Compute distance from each point in S to its nearest center in C
        dists = np.min(np.linalg.norm(S[:, np.newaxis] - C, axis=2), axis=1)
        probs = dists

    # Normalize to get probabilities
    total = np.sum(probs)
    if total == 0:
        probs = np.ones(len(S)) / len(S)  # fallback to uniform
    else:
        probs = probs / total

    # Sample k points from S with replacement according to probs
    indices = np.random.choice(len(S), size=k, replace=True, p=probs)
    return S[indices].tolist()

def count_within_radius(S, C, r):
    """
    Count the number of points in S within distance r from any point in C.

    Parameters:
        S (list or np.ndarray): Points to check, shape (n, d)
        C (list or np.ndarray): Center points, shape (m, d)
        r (float): Radius threshold

    Returns:
        int: Number of points in S within distance r from some center in C
    """
    S = np.array(S)
    C = np.array(C)

    if len(S) == 0 or len(C) == 0:
        return 0 # No points or centers → no coverage

    # Compute distances from each point in S to all points in C
    dists = np.linalg.norm(S[:, np.newaxis] - C, axis=2)

    # Find the minimum distance to a center for each point in S
    min_dists = np.min(dists, axis=1)

    # Count how many distances are <= r
    return np.sum(min_dists <= r)

def remove_within_radius(S, C, r):
    """
    Remove points in S that are within distance r from any point in C.

    Parameters:
        S (list or np.ndarray): Candidate points, shape (n, d)
        C (list or np.ndarray): Center points, shape (m, d)
        r (float): Distance threshold

    Returns:
        list: Filtered list of points in S that are farther than r from all points in C
    """
    S = np.array(S)
    C = np.array(C)

    if len(S) == 0 or len(C) == 0:
        return S.tolist()  # No points or centers → return all of S

    # Compute distances between each point in S and all centers in C
    dists = np.linalg.norm(S[:, np.newaxis] - C, axis=2)
    
    # Minimum distance to any center for each point in S
    min_dists = np.min(dists, axis=1)

    # Filter points whose min distance > r
    filtered_S = S[min_dists > r]

    return filtered_S.tolist()

def assign_labels(S, C):
    """
    Assigns each point in S to the index of the closest center in C.

    Parameters:
    - S (ndarray): An (n, d) array of n points in d-dimensional space.
    - C (ndarray): A (k, d) array of k center points.

    Returns:
    - labels (ndarray): An array of length n where labels[i] is the index j of the closest center C[j] to S[i].
    """
    return np.argmin(np.linalg.norm(S[:, np.newaxis] - C, axis=2), axis=1)

def mettu_plaxton(points, k, c):
    """
    Mettu–Plaxton-style algorithm:
    - Runs O(log n) rounds
    - Samples k points per round using sample_by_distance

    Parameters:
        points (np.ndarray): Array of shape (n, d)
        k (int): Number of samples per round
        constant c

    Returns:
        np.ndarray: Array of shape (k * log n, d) with sampled candidates
    """
    n = len(points)
    log_n = max(1, int(c*np.ceil(np.log2(n))))
    candidates = points.copy()
    centers = []

    for _ in range(log_n):
        new_samples = sample_uniform(candidates, k)
        centers.extend(new_samples)
        r = 0.1
        while(count_within_radius(candidates, centers, r) < len(candidates)/2):
            r = r*1.01
        candidates = remove_within_radius(candidates, centers, r)

    return np.array(centers)

def adaptive_sampling(points, k, c):
    """
    Adaptive sampling algorithm:
    - Runs O(k) rounds
    - Samples 1 points per round using sample_by_distance

    Parameters:
        points (np.ndarray): Array of shape (n, d)
        k (int): Number of samples per round
        constant c

    Returns:
        np.ndarray: Array of shape (k * log n, d) with sampled candidates
    """
    n = len(points)
    candidates = points.copy()
    centers = []

    for _ in range(k*c):
        new_samples = sample_by_distance(candidates, centers, 1)
        centers.extend(new_samples)
    return np.array(centers)

def generate_gaussian_points(n, k, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    points = []
    labels = []

    for i in range(k):
        # Randomly generate mean in range [-10, 10] for each Gaussian
        mean = np.random.uniform(-10, 10, size=2)
        
        # Create a random positive-definite covariance matrix
        A = np.random.rand(2, 2)
        cov = np.dot(A, A.T) + np.eye(2) * 0.5  # ensures it's positive-definite

        # Sample n points from this Gaussian
        samples = np.random.multivariate_normal(mean, cov, n)
        points.append(samples)
        labels.extend([i] * n)

    # Stack all the points together
    all_points = np.vstack(points)
    all_labels = np.array(labels)
    return all_points, all_labels

def round_vector_to_nearest_power(v, q):
    """
    Rounds each coordinate of vector v to the nearest power of q.
    
    - Positive values → nearest positive power of q
    - Negative values → nearest negative power of q
    - Zero stays zero

    Parameters:
    - v: 1D NumPy array of real numbers
    - q: base > 1

    Returns:
    - A NumPy array with each value rounded accordingly
    """
    v = np.asarray(v)
    if q <= 1:
        raise ValueError("q must be > 1")

    result = np.zeros_like(v)

    # Non-zero values
    nonzero_mask = v != 0
    abs_vals = np.abs(v[nonzero_mask])
    exponents = np.round(np.log(abs_vals) / np.log(q))
    powers = np.power(q, exponents)

    result[nonzero_mask] = np.sign(v[nonzero_mask]) * powers
    return result

def plot_clusters(S, C, labels):
    k = C.shape[0]
    base_cmap = plt.colormaps.get_cmap('tab10')
    color_list = [base_cmap(i % 10) for i in range(k)]  # tab10 has only 10 unique colors

    plt.figure(figsize=(8, 6))

    for j in range(k):
        cluster_points = S[labels == j]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    s=30, color=color_list[j], label=f'Cluster {j}')

    plt.scatter(C[:, 0], C[:, 1], 
                s=200, c='black', marker='X', label='Centers')

    plt.title('Clusters and Centers')
    plt.legend()
    plt.grid(True)
    plt.show()

i = 10
n = 100 * 2**i # number of points in each cluster
k = 5 # number of clusters
points, labels = generate_gaussian_points(n, k, random_seed=42)
c = 2 # constant scaling for number of samples
mp_centers = mettu_plaxton(points, k, c)
as_centers = adaptive_sampling(points, k, c)
q = 2 # base for rounding

rounded_as_centers = round_vector_to_nearest_power(as_centers, q)

mp_labels = assign_labels(points, mp_centers)
as_labels = assign_labels(points, as_centers)
rounded_as_labels = assign_labels(points, rounded_as_centers)
S = mp_centers[mp_labels]
S = as_centers[as_labels]
S = rounded_as_centers[rounded_as_labels]
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(S)  # S is your (n, d) array of data points

labels = kmeans.labels_       # labels[i] is the index of the cluster assigned to S[i]
centers = kmeans.cluster_centers_  # the resulting cluster centers

plot_clusters(points, centers, labels)
