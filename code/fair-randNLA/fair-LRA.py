#import matplotlib.pyplot as plt
#from kneed import KneeLocator
#from sklearn import preprocessing
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt

#https://realpython.com/k-means-clustering-python/
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html

def proj(X):
    return np.transpose(X)@np.linalg.pinv(X@np.transpose(X))@X

def frob_cost_of_proj(A, B):
    U, S, V = np.linalg.svd(B, full_matrices=True)
    Ak = A@proj(V[0:len(S)])
    Atop = A - Ak
    return np.linalg.norm(Atop)**2

def topK(A, k):
    U, S, V = np.linalg.svd(A, full_matrices=True)
    return V[0:k]

def leverageIterate(A, w, p):
    wp = [wx**(1-2/p) for wx in w]
    M = np.linalg.pinv(np.transpose(A)@np.diag(wp)@A)
    weights = [np.transpose(ai)@M@ai for ai in A]
    wout = [wx**(p/2) for wx in weights]
    return wout

def lewisWeights(A, p):
    w = [1 for a in A]
    for i in range(20):
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



numTrials = 100
flag = False
min_ratio = []
mean_ratio = []
## iterate over number of rows in bicriteria
bi_range = [i+1 for i in range(20)]
for bi_rows in bi_range:
## iterate over values of p for Lewis weight sampling
#p_range = [i+1 for i in range(10)]
#for p in p_range:
    #bi_rows = 3
    rows = bi_rows
    cols = bi_rows
    new_min = 1000
    run_sum = 0
    for i in range(numTrials):
        G = gaussian_matrix = np.random.normal(0, 1/rows, (rows, len(A)))
        H = gaussian_matrix = np.random.normal(0, 1/cols, (len(A[0]), cols))
        GAH = G@A@H
        #rank of bicriteria solution
        sol_rank = 1
        #parameter for lewis weight sampling
        p = 1
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
        bicrit_mat = TGAH@np.linalg.pinv(H)
        #bicrit_mat = np.linalg.pinv(TGAH)@TGAH@np.linalg.pinv(H)
        #socially fair cost of our algorithm
        outcost2a = frob_cost_of_proj(A1, bicrit_mat)
        outcost2b = frob_cost_of_proj(A2, bicrit_mat)
        outcost2 = max(outcost2a, outcost2b)

        ratio = outcost2/outcost1
        run_sum += ratio
        if ratio < new_min:
            new_min = ratio
    new_mean = run_sum / numTrials
    min_ratio.append(new_min)
    mean_ratio.append(new_mean)

## iterate over number of rows in bicriteria
plt.plot(bi_range, min_ratio, label="min ratio")
plt.plot(bi_range, mean_ratio, label="avg ratio")
plt.xticks(bi_range)
plt.xlabel('Number of rows in Gaussian sketch')

## iterate over values of p for Lewis weight sampling
#plt.plot(p_range, min_ratio, label="min ratio")
#plt.plot(p_range, mean_ratio, label="avg ratio")
#plt.xlabel('Value of p for Lewis sample')
plt.legend(loc="right")
plt.ylabel('Ratio of costs')

plt.show()
