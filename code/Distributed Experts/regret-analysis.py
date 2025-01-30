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
eta = 1/T**0.5
log_factor = 10*np.log(n)/np.log(2)
min_value = min(min(min(loss) for loss in row) for row in data)
max_value = max(max(max(loss) for loss in row) for row in data)

def sample_index_np(e_probs):
    """Sample an index from e_probs with probability proportional to its values using NumPy."""
    return np.random.choice(len(e_probs), p=np.array(e_probs) / sum(e_probs))


thresh_factor = 1
ps = [1+0.5*i for i in range(20)]
lambda_param = 1  # Rate parameter
best_losses = []
our_losses = []
mwu_losses = []

for p in ps:
    threshold = s**(1/p)/thresh_factor
    mwu_weights = [0.001]*n
    our_weights = [0.001]*n
    all_losses = [0]*n
    
    mwu_loss = 0
    our_loss = 0
    for t in range(T):
        mwu_probs = [np.exp(eta*mwu_weights[i]) for i in range(n)]
        mwu_expert = sample_index_np(mwu_probs)
        our_probs = [np.exp(eta*our_weights[i]) for i in range(n)]
        our_expert = sample_index_np(our_probs)
        for i in range(n):
            this_loss = 0
            coordinator_list = []
            for j in range(s):
                this_loss += data[i][j][t]**p
                exp_list = [random.expovariate(lambda_param) for _ in range(n)]
                if(data[i][j][t]/(exp_list[i]**(1/p)) > threshold):
                    coordinator_list.append(data[i][j][t]/(exp_list[i]**(1/p)))
            if(len(coordinator_list)>0):
                our_weights[i] += max(coordinator_list)
            this_loss = this_loss**(1/p)
            mwu_weights[i] = this_loss
            all_losses[i] += this_loss
            if i == mwu_expert:
                mwu_loss += this_loss
            if i == our_expert:
                our_loss += this_loss
    best_losses.append(max(all_losses))
    our_losses.append(our_loss)
    mwu_losses.append(mwu_loss)
    
# Create the plot
plt.figure(figsize=(8, 5))  # Set figure size

# Plot each loss list using ps as the x-axis
plt.plot(ps, our_losses, marker='o', linestyle='-', label="Our Reward", color='blue')
plt.plot(ps, mwu_losses, marker='s', linestyle='--', label="MWU Reward", color='red')
plt.plot(ps, best_losses, marker='^', linestyle='-.', label="Best Reward", color='green')

# Add labels and title
plt.xlabel("p")  # Label for x-axis
plt.ylabel("Reward")  # Label for y-axis
plt.title("Rewards across p Values")  # Title

# Add legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
