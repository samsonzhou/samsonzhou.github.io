import matplotlib.pyplot as plt
from collections import Counter
import hashlib

#seed = 30
seed = 40

filename = "130900_ip_timestamps.csv"
m = 1000000
#39900564
n = 100*m
probs = [i+2 for i in range(10)]

#String to integer hash function
def string_to_hash(s, n):
    """Hashes a string to an integer in the range [0, n-1]."""
    hash_value = int(hashlib.sha256(s.encode()).hexdigest(), 16)  # Convert hash to integer
    return (hash_value + seed) % n  # Map to range [0, n-1]

#Integer to binary hash function with probability p
def int_to_binary(x, p):
    """Hashes an integer to {0,1}, returning 1 with probability p."""
    hash_value = int(hashlib.sha256(str(x).encode()).hexdigest(), 16)  # Convert hash to integer
    normalized = (hash_value + seed) % 10**8 / 10**8  # Normalize to [0,1)
    return 1 if normalized < p else 0  # Return 1 with probability p

#Count the number of items that hash to 1
def count_hashed_ones(senders, p):
    """Counts the number of items in senders that hash to 1."""
    return sum(int_to_binary(sender, p) for sender in senders)

#scales=[1,2,4,8,16,32,64,128,256,512,1024]
scales=[1,4,16,64,256,1024]

all_errs_all_rounds = []

# Read m lines
with open(filename, "r") as file:
    file.readline() #remove header
    for roundVal in range(20):
        data = dict()
        receivers = set()
        senders = set()
        all_errs = []
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

        all_naive = []
        all_ours = []
        all_ests = []

        true_senders = set()
        for receiver in data.keys():
            for sender in data[receiver].keys():
                true_senders.add(sender)
        true_count=len(true_senders)

        #for denom in probs:
        for scale in scales:
            all_senders = set()
            #prob = 1/denom
            #p = 1/prob**2 * 1/len(senders)
            p = min(1, 10 * scale * 1/len(senders))
            naive = count_hashed_ones(senders, p) * len(receivers)
            ours = 0
            all_naive.append(naive)
            for receiver in data.keys():
                ours += sum(int_to_binary(sender, p) for sender in data[receiver].keys())
                for sender in data[receiver].keys():
                    if int_to_binary(sender, p) == 1:
                        all_senders.add(sender)
            all_ours.append(ours)
            all_ests.append(len(all_senders)*1/p)

        all_errs = [min(all_ests[i]/true_count,true_count/all_ests[i]) for i in range(len(scales))]
        all_errs_all_rounds.append(all_errs)
        
# Create the plot
plt.figure(figsize=(8, 5))  # Set figure size

# Set x-axis to logarithmic scale
plt.xscale("log")

# Plot each line
#plt.plot(scales, all_errs_all_rounds[0], marker='o', linestyle='-', label="Ours", color='blue')
for i in range(len(all_errs_all_rounds)):
    plt.plot(scales, all_errs_all_rounds[i], marker='o', linestyle='-', color='blue')
#plt.plot(probs, all_naive, marker='o', linestyle='-', label="Naive", color='blue')
#plt.plot(probs, all_ours, marker='s', linestyle='--', label="Ours", color='red')

# Add labels and title
plt.xlabel("p")
plt.ylabel("Error")
plt.title("Probability vs. Error")

# Add legend and grid
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
