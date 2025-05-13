import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt

#https://realpython.com/k-means-clustering-python/
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

def proj(X):
    return np.transpose(X)@np.linalg.pinv(X@np.transpose(X))@X

def frob_cost_of_proj(A, B):
    U, S, V = np.linalg.svd(B, full_matrices=False)
    rankB = np.linalg.matrix_rank(B)
    Ak = A@proj(V[0:rankB])
    Atop = A - Ak
    return np.linalg.norm(Atop)**2

def generate_random_matrix(n, d, min_value=0, max_value=100):
    """
    Generate a random n by d integer matrix.

    Parameters:
    - n: Number of rows.
    - d: Number of columns.
    - min_value: Minimum value for the matrix elements (default is 0).
    - max_value: Maximum value for the matrix elements (default is 100).

    Returns:
    - A list of lists representing the random matrix.
    """
    matrix = [[random.randint(min_value, max_value) for _ in range(d)] for _ in range(n)]
    return matrix

#DATASET EXTRACTION
file = open("Skin_NonSkin.txt", "r")

dataset_str=[]
for line in file.readlines():
    line_str=line.split("\t")
    thisLine = []
    for term in line_str:
        thisLine.append(int(term))
    thisLine = thisLine[:-1]
    dataset_str.append(thisLine)
dataset = np.array(dataset_str)
dataset_normalized = preprocessing.scale(dataset)

n=len(dataset)
#Too long for SVD
n=200
d=len(dataset[0])
##n=10
##d=4
##randA = generate_random_matrix(n, d)
##dataset_normalized=np.array(randA)
##
##nums = n
##dataset_normalized = dataset_normalized[0:nums]

k = 2
c = 10
count = -1
factors = []
our_costs = []
true_costs = []
ratios = []
r = 0
old_r = 0
kcurr = 0
for t in range(n):
    At = dataset_normalized[0:t]
    sq_norm_At = np.linalg.norm(At)**2
    U, S, V = np.linalg.svd(At, full_matrices=True)
    r = len(V)
    kcurr = min(r,k)
##    if(r > old_r and r <= k):
##        kcurr = min(r, k)
##        count = sq_norm_At
##        factors = V[0:kcurr] 
    if(sq_norm_At>c*count):
        count = sq_norm_At
        factors = V[0:kcurr]  
    #old_r = r
    our_cost = frob_cost_of_proj(At, factors)
    if(our_cost < 10**-15):
        #Probably rounding error
        our_cost = 0
    true_cost = frob_cost_of_proj(At, V[0:kcurr])
    if(true_cost < 10**-15):
        #Probably rounding error
        true_cost=0
    our_costs.append(our_cost)
    true_costs.append(true_cost)
    if true_cost==0:
        ratio = 1
    else:
        ratio=our_cost/true_cost
    ratios.append(ratio)
    if t%100==0:
        print(t)
