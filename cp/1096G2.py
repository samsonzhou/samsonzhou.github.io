import sys
#from collections import defaultdict, deque, Counter
#import heapq
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

from functools import cache

@cache
def find_nth_root_of_unity(n: int, p: int) -> int:
    assert (p-1) % n == 0
    
    for x in range(2, p):
        omega = pow(x, (p-1) // n, p)
        if pow(omega, n // 2, p) == 1:
            continue
        return omega
    
    assert False

def fft(a: list[int], start: int = 0, step: int = 1, mod: int = 998244353) -> list[int]:
    n = len(a) // step
    
    if n == 1:
        return [a[start]]
    
    # assert n.bit_count() == 1
    assert (mod - 1) % n == 0
    
    omega = find_nth_root_of_unity(n, mod)
    
    p = fft(a, start, step * 2, mod)
    q = fft(a, start + step, step * 2, mod)
    
    w = [0] * n
    
    curr_omega = 1
    curr_other_omega = pow(omega, n//2, mod)
    
    for i in range(n//2):
        w[i] = (p[i] + q[i] * curr_omega) % mod # curr_omega = pow(omega, i, mod)
        w[n//2+i] = (p[i] + q[i] * curr_other_omega) % mod # curr_other_omega = pow(omega, n//2+i, mod)
        curr_omega = curr_omega * omega % mod
        curr_other_omega = curr_other_omega * omega % mod
    
    return w

def convolve(a: list[int], b: list[int], mod: int = 998244353) -> list[int]:
    a = a[:]
    b = b[:]
    
    nn = len(a) + len(b) - 1
    n = 1 << (nn-1).bit_length()
    
    a = a + [0] * (n-len(a))
    b = b + [0] * (n-len(b))
    
    wa = fft(a, mod=mod)
    wb = fft(b, mod=mod)
    
    w = [x * y for x, y in zip(wa, wb)]
    p = fft(w, mod=mod)
    p = p[:1] + p[:0:-1]
    p = [pp * pow(n, -1, mod) % mod for pp in p]
    return p[:nn]

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
    ds = list(map(int, input().split()))
    p=[]
    for i in range(10):
        if i in ds:
            p.append(1)
        else:
            p.append(0)

    t=n.bit_length()-1
    out=[1]
    m=n//2
    for i in range(t+1):
        if m>>i&1==1:
            out=convolve(out,p)
        if i==t:
            continue
        p=convolve(p,p)
    tot=0
    for i in out:
        tot=(tot+pow(i,2,998244353))%998244353
    print(tot)
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
