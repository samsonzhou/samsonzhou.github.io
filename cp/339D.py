import sys
#from collections import defaultdict, deque, Counter
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    N=2**n
    tree=[0]*(2*N)
    for i in range(N):
        tree[i]=a[i]
    newd=N
    oldd=0
    p=1
    b=0
    for i in range(n):
        for j in range(N//(2**p)):
            if b==0:
                tree[newd+j]=tree[oldd+2*j]|tree[oldd+2*j+1]
            else:
                tree[newd+j]=tree[oldd+2*j]^tree[oldd+2*j+1]
        oldd=newd
        newd=newd+N//(2**p)
        p+=1
        b=1-b
        
    for i in range(m):
        q, x = map(int, input().split())
        q-=1
        tree[q]=x
        b=0
        newd=N
        oldd=0
        for j in range(n):
            if b==0:
                tree[newd+q//2]=tree[oldd+q]|tree[1^(oldd+q)]
            else:
                tree[newd+q//2]=tree[oldd+q]^tree[1^(oldd+q)]
            b=1-b
            oldd=newd
            newd=newd+N//2**(j+1)
            q//=2
        print(tree[-2])
    
#Booth's algorithm
#Finds first lexicographically ordered cyclic shift of a string s
#Essentially iterate over ss (s repeated twice) and KMP-style search
def least_rotation(s):
    s = s + s
    n = len(s) // 2

    i, j, k = 0, 1, 0

    while i < n and j < n and k < n:
        if s[i + k] == s[j + k]:
            k += 1
            continue

        if s[i + k] > s[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1

        k = 0

    start = min(i, j)
    return s[start:start + n]

#Subroutine for creating a list of all primes up to n
#Sieve approach, removes all multiples, runtime O(n log log n)
def all_divs_up_to(n):
    ls=[-1]*n
    for i in range(2,big):
        if ls[i]==-1:
            for j in range(i*i,big,i):
                if ls[j]==-1:
                    ls[j]=i
    ls[0]=0
    ls[1]=1
    primes=[j for j in range(big) if ls[j]==-1]
    return ls

#from tryingoutcp
def dfs_iterative(graph, root, parent, sz):
    n = len(graph)
 
    visited = [0] * n
    order = []
    stack = [root]
    visited[root] = 1
    while stack:
        node = stack.pop()
        order.append(node)
 
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = 1
                parent[neighbor] = node
                stack.append(neighbor)
 
    for node in reversed(order):
        sz[node] = 1
        for neighbor in graph[node]:
            if neighbor != parent[node]:
                sz[node] += sz[neighbor]
