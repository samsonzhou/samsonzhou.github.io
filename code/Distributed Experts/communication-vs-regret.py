import json
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)  # Set seed for reproducibility

# Load the JSON file
with open("meta-test-dataset.json", "r") as file:
    data = json.load(file)

counts = set()

csv_data = list()

for model in data.keys():
    j = 0
    model_stats = list()
    for sub in data[model].keys():
        if j < 2:
            model_stats.append([data[model][sub]['y'][i][0] for i in range(200)])
            j += 1
    csv_data.append(model_stats)

data = csv_data

n = len(data)
s = len(data[0])
T = len(data[0][0])
p = 1
Rs = [0.25*i for i in range(21)]
log_factor = 10*np.log(n)/np.log(2)
min_value = min(min(min(loss) for loss in row) for row in data)
max_value = max(max(max(loss) for loss in row) for row in data)

def sample_index_np(e_probs):
    """Sample an index from e_probs with probability proportional to its values using NumPy."""
    return np.random.choice(len(e_probs), p=np.array(e_probs) / sum(e_probs))


thresh_factor = 1
lambda_param = 1  # Rate parameter
all_our_bits = []
all_prev_bits = []

for R in Rs:
    prob = T**(0.2*R-1)
    threshold = s**(1/p)/thresh_factor
    our_bits = 0
    prev_bits = 0
    for t in range(T):
        for i in range(n):
            for j in range(s):
                exp_list = [random.expovariate(lambda_param) for _ in range(n)]
                if(data[i][j][t]/(exp_list[i]**(1/p)) > threshold):
                    prev_bits += 1
                    if random.random() < prob:
                        our_bits += 1
    all_our_bits.append(our_bits)
    all_prev_bits.append(prev_bits)
    
# Create the plot
plt.figure(figsize=(8, 5))  # Set figure size

# Plot each loss list using ps as the x-axis
plt.plot(Rs, all_our_bits, marker='o', linestyle='-', label="Ours", color='blue')
plt.plot(Rs, all_prev_bits, marker='s', linestyle='--', label="Prev", color='red')

# Add labels and title
plt.xlabel("R")  # Label for x-axis
plt.ylabel("Communication")  # Label for y-axis
plt.title("Communication vs. Regret")  # Title

# Add legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
