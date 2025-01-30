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

ps = [1+0.5*i for i in range(20)]
n = len(data)
s = len(data[0])
T = len(data[0][0])
eta = 1/T**0.5
log_factor = 10*np.log(n)/np.log(2)
min_value = min(min(min(loss) for loss in row) for row in data)
max_value = max(max(max(loss) for loss in row) for row in data)

def sample_index_np(e_probs):
    """Sample an index from e_probs with probability proportional to its values using NumPy."""
    return np.random.choice(len(e_probs), p=np.array(e_probs) / sum(e_probs))

weights = [-1]*n
thresh_factors = [0.5,1,1.5,2,2.5,3]
lambda_param = 1  # Rate parameter
all_lists = []
for thresh_factor in thresh_factors:
    list_size = []
    for p in ps:
        #threshold = s**(1/p)/(100*log_factor)*min_value
        threshold = s**(1/p)/thresh_factor
        coordinator_list = []
        for t in range(T):
            e_probs = [eta*np.exp(weights[i]) for i in range(len(weights))]
            for i in range(n):
                for j in range(s):
                    exp_list = [random.expovariate(lambda_param) for _ in range(n)]
                    if(data[i][j][t]/(exp_list[i]**(1/p)) > threshold):
                        coordinator_list.append(data[i][j][t])
            picked_expert = sample_index_np(e_probs)
        list_size.append(len(coordinator_list))
    all_lists.append(list_size)

naive = [6400 for i in range(20)]

# Create x-values (assuming same indices for both lists)
x = list(range(1, 21))  # 1 to 20

# Create the plot
plt.figure(figsize=(8, 5))  # Set figure size

# Plot first line (Naive)
plt.plot(x, naive, marker='o', linestyle='-', color='b', label="Naive")

# Plot each list as a separate line
for i, data in enumerate(all_lists):
    plt.plot(x, data, marker='o', linestyle='-', label=f"Threshold {thresh_factors[i]}")
    
# Add labels and title
plt.xlabel("p")
plt.ylabel("Total Communication")
plt.title("Improvement in Communication over Naive")

# Add legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
