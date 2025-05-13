import scipy
from scipy.linalg import hadamard
from numpy import linalg as LA
import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import random
import matplotlib.pyplot as plt

k=3
Amat = [[200,0,0,0,0],
        [0,200,0,0,0],
        [0,0,100,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]]
A = np.array(Amat)
u, sone, v = np.linalg.svd(A, full_matrices=True)
vtop = v[0:k]
Pv = np.matmul(np.transpose(vtop),vtop)
projA = np.matmul(A,Pv)
lossA = np.linalg.norm(A-projA)
frobLossA = lossA**2


Bmat = [[200,0,0,0,0],
        [0,200,0,0,0],
        [0,0,100,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,100,50,50]]
B = np.array(Bmat)
projB = np.matmul(B,Pv)
lossB = np.linalg.norm(B-projB)
frobLossB = lossB**2

w, stwo, x = np.linalg.svd(B, full_matrices=True)
xtop = x[0:k]
Px = np.matmul(np.transpose(xtop),xtop)
projBtwo = np.matmul(B,Px)
lossBtwo = np.linalg.norm(B-projBtwo)
frobLossBtwo = lossBtwo**2

newVec = [0,0,100,50,50]
uNew = newVec / np.linalg.norm(newVec)
Cmat = [[1,0,0,0,0],
        uNew]
C = np.array(Cmat)
projC = np.matmul(np.transpose(C),C)
lossBthree = np.linalg.norm(B-np.matmul(B,projC))
frobLossBthree = lossBthree**2

print(frobLossA, frobLossB, frobLossBtwo, frobLossBthree)
