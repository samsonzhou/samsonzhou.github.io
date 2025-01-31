import matplotlib.pyplot as plt
from collections import Counter
import hashlib

filename = "130900_ip_timestamps.csv"
m = 1000000
#39900564
n = 100*m
probs = [i+2 for i in range(10)]
data = dict()
receivers = set()
senders = set()

#String to integer hash function
def string_to_hash(s, n):
    """Hashes a string to an integer in the range [0, n-1]."""
    hash_value = int(hashlib.sha256(s.encode()).hexdigest(), 16)  # Convert hash to integer
    return hash_value % n  # Map to range [0, n-1]

#Integer to binary hash function with probability p
def int_to_binary(x, p):
    """Hashes an integer to {0,1}, returning 1 with probability p."""
    hash_value = int(hashlib.sha256(str(x).encode()).hexdigest(), 16)  # Convert hash to integer
    normalized = hash_value % 10**8 / 10**8  # Normalize to [0,1)
    return 1 if normalized < p else 0  # Return 1 with probability p

#Count the number of items that hash to 1
def count_hashed_ones(senders, p):
    """Counts the number of items in senders that hash to 1."""
    return sum(int_to_binary(sender, p) for sender in senders)

# Task 1: Read and print the first three lines
with open(filename, "r") as file:
    file.readline()
    for _ in range(m):
        line = file.readline().strip()
        if not line:
            break
        line_data = line.split(",")
        receiver_ip = string_to_hash(line_data[3],n)
        sender_ip = string_to_hash(line_data[2],n)
        receivers.add(string_to_hash(line_data[3],n))
        senders.add(string_to_hash(line_data[2],n))
#        receiver_ip = line_data[3]
#        sender_ip = line_data[2]
        if receiver_ip not in data:
            data[receiver_ip] = {sender_ip:1}
        else:
            if sender_ip not in data[receiver_ip]:
                data[receiver_ip][sender_ip] = 1
            else:
                data[receiver_ip][sender_ip] += 1

# Compute the lengths of each nested dictionary
lengths = [len(value) for value in data.values()]

# Count occurrences of each length
length_counts = Counter(lengths)

# Print histogram distribution
print("Histogram Distribution:")
for length, count in sorted(length_counts.items()):
    print(f"{length} keys: {count} occurrences")

# Find the key with the longest inner dictionary
longest_key = max(data, key=lambda k: len(data[k]))
longest_dict = data[longest_key]
longest_length = len(data[longest_key])

print(f"The dictionary with the most keys is '{longest_key}' with {longest_length} keys.")

# Extract values from the longest dictionary
values = list(longest_dict.values())

# Print histogram distribution
value_counts = Counter(values)
print(f"Histogram Distribution for '{longest_key}':")
for value, count in sorted(value_counts.items()):
    print(f"{value}: ({count} occurrences)")

all_naive = []
all_ours = []

for denom in probs:
    prob = 1/denom
    p = 1/prob**2 * 1/len(senders)
    naive = count_hashed_ones(senders, p) * len(receivers)
    ours = 0
    all_naive.append(naive)
    for receiver in data.keys():
        ours += sum(int_to_binary(sender, p) for sender in data[receiver].keys())    
    all_ours.append(ours)

# Plot the histogram
plt.hist(lengths, bins=range(1, max(lengths) + 2), edgecolor='black', alpha=0.7)

# Plot the distribution
plt.hist(lengths, bins=range(1, max(lengths) + 2), edgecolor='black', alpha=0.7)

# Set y-axis to log scale
plt.yscale("log")
plt.xscale("log")

# Labels and title
plt.xlabel("# of Unique Senders")
plt.ylabel("Number of Receivers")
plt.title("Histogram of Senders per Receiver")

# Show plot
plt.show()


print("-" * 40)  # Separator for clarity

# Task 2: Count and print the total number of lines
with open(filename, "r") as file:
    total_lines = sum(1 for _ in file)

print("Total number of lines:", total_lines)
39900564
