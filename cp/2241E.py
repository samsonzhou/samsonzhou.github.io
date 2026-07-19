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
    
    # 2. Read multiple integers on a single line
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    a = list(map(int, input().split()))

    adj=[[] for _ in range(n+1)]
    parent=[-1]*(n+1)
    
    for i in range(n-1):
        u, v = map(int, input().split())
        adj[u].append(v)
        adj[v].append(u)

    search=[1]
    order=[]
    while search:
        node=search.pop()
        order.append(node)
        for neigh in adj[node]:
            if neigh!=parent[node]:
                parent[neigh]=node
                search.append(neigh)
    subcnt=[0]*(n+1)
    for node in reversed(order):
        if len(adj[node])==1 and adj[node][0]==parent[node]:
            subcnt[node]=1
        else:
            tot=0
            for i in adj[node]:
                if i != parent[node]:
                    tot+=(subcnt[i])
            subcnt[node]=tot+1

    tot=0
    for i in range(n):
        if math.sqrt(a[i])==math.isqrt(a[i]):
            deg=0
            sub=0
            curr=0
            #print(adj[i+1])
            if len(adj[i+1])<2:
                continue
            for j in adj[i+1]:
                if subcnt[j]>subcnt[i+1]:
                    k=n-subcnt[i+1]
                    deg+=k
                else:
                    deg+=subcnt[j]
            curr=(((deg)*(deg-1)//2)+(deg*(deg-1)*(deg-2)//6))
            for j in adj[i+1]:
                if subcnt[j]>subcnt[i+1]:
                    k=n-subcnt[i+1]
                else:
                    k=subcnt[j]
                #print(k)
                sub+=((deg-k)*(k-1)*k//2)
                sub+=(k*(k-1)//2)
                sub+=(k*(k-1)*(k-2)//6)
            tot+=(curr-sub)
            #print(tot,deg,curr,sub)
    #print(subcnt)
    print(tot)
            
    
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
