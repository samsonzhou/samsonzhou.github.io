import csv
import random
import numpy as np

# Load CSV into a list of list of lists (n by s by T)
with open("output.txt", "r") as file:
    reader = csv.reader(file)
    data = [row for row in reader]  # Convert reader object to a list of lists

n = len(data)
s = len(data[0])
T = len(data[0][0])
eta = 1/T**0.5

min_value = min(min(min(loss) for loss in row) for row in data)
max_value = max(max(max(loss) for loss in row) for row in data)

def sample_index_np(e_probs):
    """Sample an index from e_probs with probability proportional to its values using NumPy."""
    return np.random.choice(len(e_probs), p=np.array(e_probs) / sum(e_probs))

weights = [-1]*n

lambda_param = 1  # Rate parameter (λ)
for t in range(T):
    exp_list = [random.expovariate(lambda_param) for _ in range(n)]
    e_probs = [eta*np.exp(weights[i]) for i in range(len(weights))]
    coordinator_list = []
    for i in range(n):
        j=1
    picked_expert = sample_index_np(e_probs)
    
