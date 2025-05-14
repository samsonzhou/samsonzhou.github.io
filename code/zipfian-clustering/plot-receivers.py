import matplotlib.pyplot as plt
from collections import Counter
import hashlib
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

filename = "125910_ip_timestamps.csv"
m = 1048576
#1048576
#1036077 IP4
n = 100*m
probs = [i+2 for i in range(10)]
data = dict()
receivers = set()
senders = set()

def ip_to_domain(ip_str):
    """
    Convert an IP address string to an integer of the form:
    A
    where A.B.C.D are the components of the IP address.
    """
    parts = list(map(int, ip_str.strip().split('.')))
    if len(parts) != 4:
        raise ValueError("Invalid IP address format")
    return parts[0]

def ip_to_custom_integer(ip_str):
    """
    Convert an IP address string to an integer of the form:
    A*1000^3 + B*1000^2 + C*1000 + D
    where A.B.C.D are the components of the IP address.
    """
    parts = list(map(int, ip_str.strip().split('.')))
    if len(parts) != 4:
        raise ValueError("Invalid IP address format")
    #return parts[0] * 1000**3 + parts[1] * 256**2 + parts[2] * 256 + parts[3]
    return parts[0] * 256**3 + parts[1] * 256 + parts[2]

def cluster_1d_kmeans_pp(data, k):
    """
    Clusters a 1D list of real numbers into k clusters using KMeans++.
    
    Parameters:
        data (list of float): The list of real numbers to cluster.
        k (int): The number of clusters.
    
    Returns:
        labels (list of int): Cluster labels for each point.
        centers (list of float): Cluster centers.
    """
    data = np.array(data).reshape(-1, 1)  # Convert to 2D array for sklearn
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_.flatten()

def clustering_cost_per_center(data, labels, centers):
    """
    Compute the clustering cost (sum of squared distances) for each center.
    
    Parameters:
        data (list of float): The list of real points (1D).
        labels (list of int): Cluster label assigned to each point.
        centers (list of float): The cluster centers.
    
    Returns:
        list of float: Cost per cluster center.
    """
    data = np.array(data)
    labels = np.array(labels)
    centers = np.array(centers)
    
    k = len(centers)
    costs = []
    for i in range(k):
        cluster_points = data[labels == i]
        center = centers[i]
        cost = np.sum((cluster_points - center) ** 2)
        costs.append(cost)
    
    return costs

def plot_histogram(data, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Plots a histogram of the given list of numbers.

    Parameters:
        data (list of float): The data to plot.
        bins (int): Number of histogram bins.
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Define Zipf's law function for fitting
def zipf_rank_function(rank, s, C):
    """Zipf's law: frequency ~ C * rank^(-s)"""
    return C * rank**(-s)

# Function to estimate Zipf parameters
def estimate_zipf_parameters(lengths):
    # Sort the list (if not already sorted)
    lengths = sorted(lengths, reverse=True)

    # Generate ranks (1, 2, 3, ..., N)
    ranks = np.arange(1, len(lengths) + 1)

    # Perform a least squares fit to estimate 's' and 'C'
    params, _ = curve_fit(zipf_rank_function, ranks, lengths, p0=[1, max(lengths)])

    # Return the estimated parameters
    return {"s": params[0], "C": params[1]}

def count_unique_items(L):
    """
    Compute the number of unique items in a list L.
    
    Parameters:
        L (list): The list of items.
    
    Returns:
        int: The number of unique items in the list.
    """
    return len(set(L))

def generate_zipfian_list(C, s, length):
    """
    Generate a list of values based on Zipf's law with given parameters.
    
    Parameters:
        C (float): The constant.
        s (float): The Zipfian exponent.
        length (int): Length of the list to generate.
    
    Returns:
        list: A list of values following Zipf's law.
    """
    ranks = np.arange(1, length + 1)
    values = C / (ranks ** s)
    return values

data = []
domains = []
k = 201 #201 unique domains in the dataset

# Task 1: Read and print the first three lines
with open(filename, "r") as file:
    file.readline()
    for _ in range(m):
        line = file.readline().strip()
        if not line:
            break
        try:
            line_int = ip_to_custom_integer(line)
            data.append(line_int)
            domain = ip_to_domain(line)
            domains.append(domain)
            # do something with line_int
        except ValueError as e:
            continue

labels, centers = cluster_1d_kmeans_pp(data, k)
costs = clustering_cost_per_center(data, labels, centers)
#print("Costs per center:", costs)
#plot_histogram(domains, bins=100)

costs = sorted(costs,reverse=True)
zipf_params = estimate_zipf_parameters(costs)
C = 28899942464646.934
s = 0.8373525258185955
L = generate_zipfian_list(C, s, k)
print(zipf_params)
plt.figure(figsize=(8, 5))
plt.plot(costs, marker='o', markersize=6, markerfacecolor='none', linestyle='-', label=f"Clustering Costs", alpha=0.9)
plt.plot(L, marker='x', markersize=4, linestyle='-', label=f"Zipf's Law", alpha=0.9)
plt.yscale('log')

# Add title and labels
title="Clustering Costs"
xlabel="Cluster Index (i)"
ylabel="Cost"
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)

# Show legend
plt.legend()

# Show grid and the plot
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#plot_histogram(costs, bins=256)
