import sys
#from collections import defaultdict, deque, Counter
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

def solve():
    """
    Main logic for a single test case.
    """
    # 1. Read a single integer
    n = int(input())

    ints=[]
    for i in range(n):
        ai,bi= map(int, input().split())
        ints.append((ai,bi))
    ints.sort()

    def cnt_invs(arr):
        if len(arr)==1:
            return 0
        else:
            L=len(arr)
            larr=arr[:L//2]
            rarr=arr[L//2:]
            lfin = [larr[i][1] for i in range(len(larr))]
            rfin = [rarr[i][1] for i in range(len(rarr))]
            lfin.sort()
            rfin.sort()
            cnt=0
            i=0
            j=0
            while i < len(lfin) and j<len(rfin):
                if lfin[i]<rfin[j]:
                    cnt+=j
                    i+=1
                else:
                    while j<len(rfin) and lfin[i]>=rfin[j]:
                        j+=1
                    if j<len(rfin) and lfin[i]<rfin[j]:
                        cnt+=j
                        i+=1
            if i<len(lfin) and j==len(rfin):
                thiscnt=(len(lfin)-i)*len(rfin)
                cnt+=thiscnt
            return cnt_invs(larr)+cnt_invs(rarr)+cnt

    out=cnt_invs(ints)
    print(out)    
    
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
    t = int(input())
    for _ in range(t):
        solve()

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
