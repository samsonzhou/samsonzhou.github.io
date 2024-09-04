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
    probs = w / np.sum(w)
    return A[np.random.choice(len(w), p=probs)]

def lewisSample(A, p, numSamples):
    w = lewisWeights(A, p)
    outRows = []
    for i in range(numSamples):
        outRow = sampleRow(A,w)/numSamples
        outRows.append(outRow)
    return outRows

A1 = np.array([[2, 0, 0, 0],
              [0, 2, 0, 0]])

A2 = np.array([[0, 0, 1.99, 0],
              [0, 0, 0, 1.99]])

A = np.vstack((A1, A2))

file = open("Skin_NonSkin.txt", "r")

A1 = []
A2 = []

dataset_str=[]
for line in file.readlines():
    line_str=line.split("\t")
    thisLine = []
    for term in line_str:
        thisLine.append(int(term))
    marker = thisLine[-1]
    thisLine = thisLine[:-1]
    if marker == 1:
        A1.append(thisLine)
    else:
        A2.append(thisLine)
    dataset_str.append(thisLine)
#dataset = np.array(dataset_str)
#dataset_normalized = preprocessing.scale(dataset)
data_size = 100
A1 = A1[0:data_size]
A2 = A2[0:data_size]
A1 = np.array(A1)
A2 = np.array(A2)
A = np.vstack((A1, A2))

birows = 10
#rows = 10
#cols = 10
rows = birows
cols = birows

G = gaussian_matrix = np.random.normal(0, 1/rows, (rows, len(A)))

H = gaussian_matrix = np.random.normal(0, 1/cols, (len(A[0]), cols))

GAH = G@A@H
#rank of bicriteria solution
sol_rank = 2
TGAH = lewisSample(GAH, 2, sol_rank)

A_sol = topK(A,1)
outcost1a = frob_cost_of_proj(A1, A_sol)
outcost1b = frob_cost_of_proj(A2, A_sol)
outcost1 = max(outcost1a, outcost1b)

bicrit_mat = np.linalg.pinv(TGAH)@TGAH@np.linalg.pinv(H)

outcost2a = frob_cost_of_proj(A1, bicrit_mat)
outcost2b = frob_cost_of_proj(A2, bicrit_mat)
outcost2 = max(outcost2a, outcost2b)

print(outcost1, outcost2)
