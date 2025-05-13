import numpy as np

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

    if len(C) == 0:
        return 0  # No centers → no coverage

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

    if len(C) == 0:
        return S.tolist()  # No centers → return all of S

    # Compute distances between each point in S and all centers in C
    dists = np.linalg.norm(S[:, np.newaxis] - C, axis=2)
    
    # Minimum distance to any center for each point in S
    min_dists = np.min(dists, axis=1)

    # Filter points whose min distance > r
    filtered_S = S[min_dists > r]

    return filtered_S.tolist()


S = [(0, 0), (1, 1), (3, 3), (5, 5)]
C = [(0, 0), (2, 2)]
r = 2.0

print(remove_within_radius(S, C, r))
