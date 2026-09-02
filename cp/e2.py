import sys
from collections import defaultdict, deque, Counter
#import heapq
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

from functools import cache

MOD = 998244353
# precompute twiddle tables once, like doc2 -- this is the piece that replaces
# find_nth_root_of_unity's per-call search
_g = 3
_ig = pow(_g, MOD - 2, MOD)
W  = [pow(_g,  (MOD - 1) >> i, MOD) for i in range(24)]
iW = [pow(_ig, (MOD - 1) >> i, MOD) for i in range(24)]
 
def fft(k, f):
    """Forward transform, in-place, mutates f. Replaces doc1's recursive fft()."""
    for l in range(k, 0, -1):        # l = level, replaces recursion depth
        d = 1 << (l - 1)             # distance between butterfly partners at this level
        U = [1]
        for _ in range(d):
            U.append(U[-1] * W[l] % MOD)
        for i in range(1 << (k - l)):    # which block
            base = i * 2 * d
            for j in range(d):            # which element within the block
                s = base + j
                f[s], f[s + d] = (f[s] + f[s + d]) % MOD, U[j] * (f[s] - f[s + d]) % MOD
 
def ifft(k, f):
    """Inverse transform, in-place, mutates f."""
    for l in range(1, k + 1):
        d = 1 << (l - 1)
        for i in range(1 << (k - l)):
            base = i * 2 * d
            u = 1
            for j in range(d):
                s = base + j
                f[s + d] *= u
                f[s], f[s + d] = (f[s] + f[s + d]) % MOD, (f[s] - f[s + d]) % MOD
                u = u * iW[l] % MOD
 
def convolve(a, b, mod=998244353):
    a = a[:]; b = b[:]
    nn = len(a) + len(b) - 1
    k = (nn - 1).bit_length()
    n = 1 << k
    a += [0] * (n - len(a))
    b += [0] * (n - len(b))
    fft(k, a)          # was: wa = fft(a, mod=mod)   -- now mutates a directly
    fft(k, b)          # was: wb = fft(b, mod=mod)
    w = [x * y % mod for x, y in zip(a, b)]
    ifft(k, w)          # was: p = fft(w, mod=mod); reverse trick
    inv = pow(n, mod - 2, mod)
    w = [x * inv % mod for x in w]
    return w[:nn]


def solve():
    """
    Main logic for a single test case.
    """
    # 1. Read a single integer
    # n = int(input())
    
    # 2. Read multiple integers on a single line
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    n, k = map(int, input().split())
    MOD=998244353
    if k>2*n:
        print(0)
    else:
        q=deque()
        for i in range(n):
            invk=pow(i+1,-1,MOD)
            p=[invk,(i*invk)%MOD]
            q.append(p)
        while len(q)>1:
            u=q.popleft()
            v=q.popleft()
            p=convolve(u,v)
            if len(p)>k-n+1:
                q.append(p[:k-n+1])
            else:
                q.append(p)
        #print(p)
        p=q.popleft()
        print(p[k-n])
        
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
    for i in range(2,n):
        if ls[i]==-1:
            for j in range(i*i,n,i):
                if ls[j]==-1:
                    ls[j]=i
    ls[0]=0
    ls[1]=1
    primes=[j for j in range(n) if ls[j]==-1]
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

def factorial_list(n, MOD):
    factorial = [1]
    for i in range(1,n+1):
        factorial.append(factorial[-1] * i % MOD)
    return factorial

def factorial_val(n, MOD):
    factorial = 1
    for i in range(1,n+1):
        factorial = (factorial*i)% MOD
    return factorial
    
def choose(n, k, MOD):
    num = factorial_val(n)
    denom = factorial_val(k) * factorial_val(n-k) % MOD
    return (num * pow(denom, -1, MOD)) % MOD
