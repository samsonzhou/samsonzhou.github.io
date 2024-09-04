#import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn import preprocessing
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
#import matplotlib.pyplot as plt

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

def topK(A, k):
    U, S, V = np.linalg.svd(A, full_matrices=False)
    return V[0:k]

def leverageIterate(A, w, p):
    wp = [wx**(1-2/p) for wx in w]
    M = np.linalg.pinv(np.transpose(A)@np.diag(wp)@A)
    weights = [np.transpose(ai)@M@ai for ai in A]
    wout = [wx**(p/2) for wx in weights]
    return wout

def lewisWeights(A, p):
    w = [1 for a in A]
    for i in range(100):
        w = leverageIterate(A, w, p)
    return w

def sampleRow(A, w):
    if np.sum(w) == 0:
        return A[np.random.choice(len(w))]
    probs = w / np.sum(w)
    return A[np.random.choice(len(w), p=probs)]

def lewisSample(A, p, numSamples):
    w = lewisWeights(A, p)
    outRows = []
    for i in range(numSamples):
        outRow = sampleRow(A,w)/numSamples
        outRows.append(outRow)
    return outRows

#Matrix for M
A1 = []
#Matrix for F
A2 = []

file = open("abalone.data", "r")
for line in file.readlines():
    l = line.split(",")[:-1]
    marker = l[0]
    l = l[1:-1]
    thisLine = []
    for term in l:
        thisLine.append(float(term))
    if marker == 'M':
        A1.append(thisLine)
    elif marker == 'F':
        A2.append(thisLine)
#Truncate dataset size for each
data_size_A1 = 1000
data_size_A2 = 50
normalize_A1 = 1
normalize_A2 = 1
normalize_A1 = data_size_A1**2
normalize_A2 = data_size_A2**2
A1 = A1[0:data_size_A1]
A2 = A2[0:data_size_A2]
A1 = np.array(A1)/normalize_A1
A2 = np.array(A2)/normalize_A2
#Stacked matrices
A = np.vstack((A1, A2))

#Generate matrices G and H for Dvoretzky's Theorem
birows = 50
#Number of rows in G
rows = birows
#Number of cols in H
cols = birows

numTrials = 200
for i in range(numTrials):
    G = gaussian_matrix = np.random.normal(0, 1/rows, (rows, len(A)))
    H = gaussian_matrix = np.random.normal(0, 1/cols, (len(A[0]), cols))
    GAH = G@A@H
    #rank of bicriteria solution
    sol_rank = 1
    #parameter for lewis weight sampling
    p = 2
    TGAH = lewisSample(GAH, p, sol_rank)

    #rank of solution to A
    k = 1
    #top k singular vectors of A
    A_sol = topK(A,k)
    #socially fair cost of best solution to A
    outcost1a = frob_cost_of_proj(A1, A_sol)
    outcost1b = frob_cost_of_proj(A2, A_sol)
    outcost1 = max(outcost1a, outcost1b)

    #our algorithm solution
    bicrit_mat = np.linalg.pinv(TGAH)@TGAH@np.linalg.pinv(H)
    #socially fair cost of our algorithm
    outcost2a = frob_cost_of_proj(A1, bicrit_mat)
    outcost2b = frob_cost_of_proj(A2, bicrit_mat)
    outcost2 = max(outcost2a, outcost2b)

    #print(outcost1, outcost2)
    print(outcost2/outcost1)
