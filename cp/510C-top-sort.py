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
    n = int(input())

    adj=[[] for _ in range(26)]
    old=input().strip()
    flag=False
    for i in range(n-1):
        new=input().strip()
        j=0
        while j<len(old) and j<len(new) and old[j]==new[j]:
            j+=1
        if j>=len(old) and j<len(new):
            continue
        elif j<len(old) and j>=len(new):
            flag=True
        else:
            x=ord(old[j])-97
            y=ord(new[j])-97
            if x not in adj[y]:
                adj[y].append(x)
            #if ord(new[j])-97 not in adj[ord(old[j])-97]:
            #    adj[ord(old[j])-97].append(ord(new[j])-97)
        old=new
    explored=[0]*26 # 0=unvisited, 1=in-progress, 2=finished
    order=[]

    def explore(v):
        global flag
        if explored[v]==2:
            pass
        else:
            explored[v]=1
            if len(adj[v])==0:
                order.append(v)
            else:
                for u in adj[v]:
                    if explored[u]==1:
                        flag=True
                    else:
                        explore(u)
                order.append(v)
            explored[v]=2
    
    for i in range(26):
        if explored[i]==2:
            i+=1
        else:
            explore(i)
    out=[]
    if flag:
        print("Impossible")
    else:
        for i in order:
            out.append(chr(97+i))
        print(*out,sep="")

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
