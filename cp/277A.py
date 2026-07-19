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
    # n = int(input())
    
    # 2. Read multiple integers on a single line
    
    
    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    n, m = map(int, input().split())

    L=[[i] for i in range(m+1)]
    p=[-1 for i in range(m+1)]
    
    def dsuget(a):
        return p[a]
    
    def dsuunion(a,b):
        if p[a]==p[b]:
            return
        if len(L[p[a]])>len(L[p[b]]):
            c=p[b]
            L[p[a]].extend(L[p[b]])
            for u in L[c]:
                p[u]=p[a]
            L[c].clear()
        else:
            c=p[a]
            L[p[b]].extend(L[p[a]])
            for u in L[c]:
                p[u]=p[b]
            
            L[c].clear()
    z=0
    for i in range(n):
        a = list(map(int, input().split()))
        if len(a)==1:
            z+=1
        else:
            a=a[1:]
            for j in range(len(a)):
                if p[a[j]]==-1:
                    p[a[j]]=a[j]
            if len(a)==0:
                continue
            else:
                for j in range(len(a)-1):
                    dsuunion(a[j],a[j+1])

    cnt=0
    for i in range(m+1):
        if len(L[i])!=0:
            if len(L[i])!=1 or p[L[i][0]]!=-1:
                cnt+=1
    print(max(0,cnt-1)+z)

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
