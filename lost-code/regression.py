import scipy
from scipy.linalg import hadamard
from numpy import linalg as LA
import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import random


n = 100 # number of rows in A
d = 20 # number of columns in A, need n > d
m = 40 # number of rows in sketch size

Amat = []
bvec = []
# generate matrix A and vector b at random
for i in range(n):
    a = []
    for j in range(d):
        # generate each row of A at random
        a.append(random.randint(-10,10))
    Amat.append(a)
    #generate each entry of b at random
    bvec.append(random.randint(-10,10))

A = np.array(Amat)
b = np.array(bvec)

##S = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0]])
Smat = []
Tmat = []

# generate Gaussian sketch matrix S and
# other (probably bad) sketch matrix T at random

for i in range(m):
    s = []
    t = []
    for j in range(n):
        # generate each row of S at random
        s.append(1/m**0.5*np.random.normal())
        # generate each row of T at random
        t.append(random.randint(-10,10))
    Smat.append(s)
    Tmat.append(t)
S = np.array(Smat)
T = np.array(Tmat)

# form U to be the concatenation of S and T
ST = np.concatenate((S, T), axis=0)
U = ST

# create sketch matrices SA, TA, and UA
SA = np.matmul(S,A)
TA = np.matmul(T,A)
UA = np.matmul(U,A)

# compute optimal vector
ans1 = np.matmul(A,np.matmul(np.linalg.pinv(A),b))
# compute optimal vector in sketched space S
ans2 = np.matmul(A,np.matmul(np.linalg.pinv(SA),np.matmul(S,b)))
# compute optimal vector in sketched space T
ans3 = np.matmul(A,np.matmul(np.linalg.pinv(TA),np.matmul(T,b)))
# compute optimal vector in sketched space U
ans4 = np.matmul(A,np.matmul(np.linalg.pinv(UA),np.matmul(U,b)))

# compute optimal loss
print("optimal loss: "+str(np.linalg.norm(ans1-b)))
# compute loss of optimal vector in sketched space S
print("loss in sketched space S: "+str(np.linalg.norm(ans2-b)))
# compute loss of optimal vector in sketched space T
print("loss in sketched space T: "+str(np.linalg.norm(ans3-b)))
# compute loss of optimal vector in sketched space U
print("loss in sketched space U, where U is the concatenation of S and T: "
      +str(np.linalg.norm(ans4-b)))
