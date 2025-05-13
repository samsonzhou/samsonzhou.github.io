import scipy
from scipy.linalg import hadamard
from numpy import linalg as LA
import numpy as np
from scipy.stats import ortho_group
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import svds
import random
import matplotlib.pyplot as plt

def proj(X):
    return np.transpose(X)@np.linalg.pinv(X@np.transpose(X))@X

def is_row_in_span(input_vector, set_of_vectors):
    a = input_vector
    b = set_of_vectors
    old_rank = np.linalg.matrix_rank(b)
    new_rank = np.linalg.matrix_rank(np.vstack((b,a)))
    
    return old_rank == new_rank

def frob_cost_of_proj(A, B):
    U, S, V = np.linalg.svd(B, full_matrices=True)
    Ak = A@proj(V)
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

# Example usage:
#n = 100  # Number of rows
n=5
#d = 40  # Number of columns
d=4
#k = 35
k=2
randA = generate_random_matrix(n, d)
randA.append([200]*d)
#randA = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0.7,0.7]]
#randB = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0.7,0.7]]
#randA = [[2.1,0,0,0],[0,1.9,0,0],[0,0,1.1,0],[0,0,0,1],[0,0,0.7,1]]
#randA = [[1.3,0,0,0],[0,1.2,0,0],[0,0,1.1,0],[0,0,0,1],[0.1,0,0.7,1]]
randA = [[1,0,0,0],[1,0,0,0],[0,0,1,0],[2,0,0,0],[1,0,3,0]]
randB = randA[:-1]
matA = np.array(randA)
u, s, vA = np.linalg.svd(matA, full_matrices=True)
topA = vA[0:k]
matB = np.array(randB)
u, sone, vB = np.linalg.svd(matB, full_matrices=True)
topB = vB[0:k]
topB = np.append(topB,[vA[-1]],axis=0)
U, S, V = np.linalg.svd(matA@proj(topB), full_matrices=True)
#U, S, V = np.linalg.svd(randA@(np.transpose(topB)@topB), full_matrices=True)
topB = V[0:k]

#topB = [[1,0,0,0],[0,0,0.7,0.7]]
#topB = [[1,0,0,0],[0,0,0.573462,0.819232]]

## Determine the overlap between the top k space of A and the top k space of B
#counter = 0
#for vec in topA:
#    if is_row_in_span(vec,topB)==False:
#        counter+=1
#print(counter)




#B_cost_on_top_B = frob_cost_of_proj(matB,topB)
A_cost_on_top_A = frob_cost_of_proj(matA,topA)
A_cost_on_top_B = frob_cost_of_proj(matA,topB)
#print(B_cost_on_top_B)
print(A_cost_on_top_A)
print(A_cost_on_top_B)
