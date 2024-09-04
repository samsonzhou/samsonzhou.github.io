#import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn import preprocessing
#from sklearn.datasets import make_blobs
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
import matplotlib.pyplot as plt
import re
import time

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
G1 = []
#Matrix for F
G2 = []

file = open("credit-cards.csv", "r")
lines = file.readlines()[2:]
for line in lines:
    l = re.split("[,\n]",line)[:-1]
    marker = float(l[2])
    #marker = float(l[-1])
    l = l[6:-2]
    thisLine = []
    for term in l:
        thisLine.append(float(term))
    if marker == 1:
    #if marker == 0:
        G1.append(thisLine)
    elif marker == 2:
    #elif marker == 1:
        G2.append(thisLine)

#Truncate dataset size for each
data_size_A1 = 10000
data_size_A2 = 10000
#data_size_A1 = data_size
#data_size_A2 = data_size
#Sample indices from each group for more efficient runtime
A1_indices = np.random.choice(len(G1), size=data_size_A1, replace=False)
A2_indices = np.random.choice(len(G2), size=data_size_A2, replace=False)
A1 = [G1[i] for i in A1_indices]
A2 = [G2[i] for i in A2_indices]
#normalization for group size
normalize_A1 = 1
normalize_A2 = 1
normalize_A1 = data_size_A1**2
normalize_A2 = data_size_A2**2
A1 = np.array(A1)/normalize_A1
A2 = np.array(A2)/normalize_A2
#normalization along columns
A1 = preprocessing.scale(np.array(A1))
A2 = preprocessing.scale(np.array(A2))
#Stacked matrices
A = np.vstack((A1, A2))

numTrials = 100

flag = False
min_ratio = []
mean_ratio = []
## iterate over number of rows in bicriteria
#bi_range = [i+1 for i in range(20)]
#for bi_rows in bi_range:

## iterate over dataset sample size
#data_sizes = [i+2 for i in range(20)]
#for data_size in data_sizes:

## iterate over values of p for Lewis weight sampling
#p_range = [i+1 for i in range(10)]
#for p in p_range:

#avg_costs
avgcosts1 = []
avgcosts2 = []

time_svd = []
time_bicrit1 = []
time_bicrit2 = []

k_range = [i+1 for i in range(10)]
for k in k_range:
    #Generate matrices G and H for Dvoretzky's Theorem
    birows = 50
    #Number of rows in G
    rows = birows
    #Number of cols in H
    cols = birows
    new_min = 1000
    run_sum = 0
    
    t_svd = 0
    t_bicrit1 = 0
    t_bicrit2 = 0
    
    avgcost1 = 0
    avgcost2 = 0
    for i in range(numTrials):
        #rank of solution to A
        #k = 1
        #top k singular vectors of A
        start_svd = time.time()
        A_sol = topK(A,k)
        stop_svd = time.time()
        t_svd+=(stop_svd-start_svd)*1000
        
        #socially fair cost of best solution to A
        outcost1a = frob_cost_of_proj(A1, A_sol)
        outcost1b = frob_cost_of_proj(A2, A_sol)
        outcost1 = max(outcost1a, outcost1b)

        start_bicrit1 = time.time()
        G = gaussian_matrix = np.random.normal(0, 1/rows, (rows, len(A)))
        H = gaussian_matrix = np.random.normal(0, 1/cols, (len(A[0]), cols))
        GAH = G@A@H
        #rank of bicriteria solution
        sol_rank = k
        #parameter for lewis weight sampling
        p = 1
        TGAH = lewisSample(GAH, p, sol_rank)
        #our algorithm solution
        start_bicrit2 = time.time()
        bicrit_mat = TGAH@np.linalg.pinv(H)
        #bicrit_mat = np.linalg.pinv(TGAH)@TGAH@np.linalg.pinv(H)
        #socially fair cost of our algorithm

        stop_bicrit = time.time()
        t_bicrit1+=(stop_bicrit-start_bicrit1)*1000
        t_bicrit2+=(stop_bicrit-start_bicrit2)*1000
        
        outcost2a = frob_cost_of_proj(A1, bicrit_mat)
        outcost2b = frob_cost_of_proj(A2, bicrit_mat)
        outcost2 = max(outcost2a, outcost2b)

        avgcost1 += outcost1
        avgcost2 += outcost2

        ratio = outcost2/outcost1
        run_sum += ratio
        if ratio < new_min:
            new_min = ratio
    time_svd.append(t_svd/numTrials)
    time_bicrit1.append(t_bicrit1/numTrials)
    time_bicrit2.append(t_bicrit2/numTrials)
    avgcost1 /= numTrials
    avgcost2 /= numTrials
    avgcosts1.append(avgcost1)
    avgcosts2.append(avgcost2)
    new_mean = run_sum / numTrials
    min_ratio.append(new_min)
    mean_ratio.append(new_mean)

## iterate over number of rows in bicriteria
#plt.plot(bi_range, min_ratio, label="min ratio")
#plt.plot(bi_range, mean_ratio, label="avg ratio")
#plt.xticks(bi_range)
#plt.xlabel('Number of rows in Gaussian sketch')

## iterate over number of sampled rows
#plt.plot(data_sizes, min_ratio, label="min ratio")
#plt.plot(data_sizes, mean_ratio, label="avg ratio")
#plt.xticks(data_sizes)
#plt.xlabel('Number of sampled observations in dataset')

## iterate over rank of solution
plt.plot(k_range, min_ratio, label="min ratio")
plt.plot(k_range, mean_ratio, label="avg ratio")
#plt.plot(k_range, avgcosts1, label="SVD")
#plt.plot(k_range, avgcosts2, label="bicrit")

##runtime plots
#plt.plot(k_range, time_svd, label="SVD")
#plt.plot(k_range, time_bicrit1, label="bicrit1")
#plt.plot(k_range, time_bicrit2, label="bicrit2")
#plt.ylim(0.7,1.1)
plt.xlabel('Rank')


#plt.plot(p_range, min_ratio, label="min ratio")
#plt.plot(p_range, mean_ratio, label="avg ratio")
#plt.xlabel('Value of p for Lewis sample')

#plt.legend(loc="lower left")
plt.legend(loc="right")
plt.ylabel('Ratio of costs')
#plt.ylabel('Runtime (ms)')
plt.grid(True)
plt.show()
#prev_ratios = [0.5687403908757838, 0.7846220211631423, 0.8673724887605413, 0.7876882784752204,
# 0.9668737845489807, 0.9775537162201142, 0.981754678820433, 0.9832153881067941,
# 0.9206360075800274, 0.9319599161600274, 0.9910652090626644, 0.9505181711803966, 0.9940928839072521,
# 0.9513950358880342, 0.9916753603383022, 0.9636617213066128,
# 0.970282474340901, 0.9868749855367684, 0.9908062828323211, 0.9970956173875936]
