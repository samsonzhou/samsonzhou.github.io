import numpy as np
import matplotlib.pyplot as plt

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

num_centers_mp = []
num_centers_as = []

for i in range(15):
    n = 100 * 2**i
    k = 5
    points, labels = generate_gaussian_points(n, k, random_seed=42)
    mp_centers = mettu_plaxton(points, k, 1)
    as_centers = adaptive_sampling(points, k, 1)
    num_centers_mp.append(len(mp_centers))
    num_centers_as.append(len(as_centers))

x_values = [2**i * 100 for i in range(len(num_centers_mp))]

# Plot
plt.figure(figsize=(8, 5))

plt.scatter(x_values, num_centers_mp, color='blue', label='MP Centers', marker='o')
plt.scatter(x_values, num_centers_as, color='red', label='AS Centers', marker='x')

plt.xscale('log', base=2)
plt.xlabel('Number of points (log scale)')
plt.ylabel('Number of centers')
plt.title('MP vs AS Centers vs Input Size')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()

#num_centers_mp: [40, 45, 50, 55, 65, 65, 70, 80, 80, 80, 90, 95, 105, 105, 110]
#num_centers_as: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

### Plotting
##plt.figure(figsize=(8, 6))
##for i in range(k):
##    plt.scatter(points[labels == i][:, 0], points[labels == i][:, 1], label=f'Gaussian {i}')
##plt.scatter(mp_centers[:, 0], mp_centers[:, 1])
##plt.scatter(as_centers[:, 0], as_centers[:, 1])
##plt.title('Generated 2D Gaussian Clusters')
##plt.xlabel('X-axis')
##plt.ylabel('Y-axis')
##plt.legend()
##plt.grid(True)
##plt.show()

S = [(0, 0), (1, 1), (3, 3), (5, 5)]
C = [(0, 0), (2, 2)]
r = 2.0

print(remove_within_radius(S, C, r))
