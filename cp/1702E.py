import sys
from collections import defaultdict, deque, Counter
#import heapq
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

    pts=[[] for _ in range(n)]
    for i in range(n):
        ai,bi=map(int,input().split())
        pts[ai-1].append(i)
        pts[bi-1].append(i)

    jvr=False
    adj=[[] for _ in range(n)]
    for i in range(n):
        if len(pts[i])>=3:
            jvr=True
        elif len(pts[i])==2:
            u,v=pts[i][0],pts[i][1]
            adj[u].append(v)
            adj[v].append(u)

    color=[-1]*n
    exp=[-1]*n
    i=0
    exp[0]=1
    color[0]=0
    while i<n and not jvr:
        q=deque()
        q.append(i)
        while q and not jvr:
            v=q.popleft()
            c=color[v]
            for u in adj[v]:
                if color[u]!=-1 and color[u]==color[v]:
                    jvr=True
                    break
                color[u]=1-color[v]
                if exp[u]==-1:
                    exp[u]=1
                    q.append(u)
        i+=1
        while i<n and exp[i]==1:
            i+=1
    if jvr:
        print("NO")
    else:
        print("YES")
        

    
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
