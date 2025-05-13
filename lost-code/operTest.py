import scipy
from scipy.linalg import hadamard
from numpy import linalg as LA
import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds

n=2**7
m=2
s=90
sm=m*s

D=scipy.sparse.identity(n).toarray()*np.random.normal(0,1,n)
H=1/(n**0.5)*1/(s**0.5)*hadamard(n)
allmats=H.dot(D)
Hmats=H

for i in range(s-1):
    D=scipy.sparse.identity(n).toarray()*np.random.normal(0,1,n)
    H=1/(n**0.5)*1/(s**0.5)*hadamard(n)
    Hmats=np.vstack((Hmats,H))
    allmats=np.vstack((allmats,H.dot(D)))
bigHD=np.array(allmats)
S=np.random.permutation(scipy.sparse.identity(s*n).toarray())[0:m]
SHD=S.dot(bigHD)
oper=LA.norm(SHD,2)
sing=scipy.sparse.linalg.svds(SHD,1,which='SM')
print(oper)
print(sing[1])
