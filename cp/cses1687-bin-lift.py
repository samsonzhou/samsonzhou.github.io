import sys
#from collections import defaultdict, deque, Counter
import heapq
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
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    n, qs = map(int, input().split())
    adj = [[] for _ in range(n)]
    a = list(map(int, input().split()))
    for i in range(n-1):
        adj[i+1].append(a[i]-1)
        adj[a[i]-1].append(i+1)

    order = []
    exp = [0]*n
    q=[0]
    exp[0]=1
    par = [-1]*n
    while q:
        v=q.pop()
        for u in adj[v]:
            if exp[u]==0:
                q.append(u)
                exp[u]=1
                par[u]=v
        order.append(v)
    order.reverse()
    ms=n.bit_length()
    pars=[[-1]*ms for i in range(n)]
    for i in range(n):
        pars[i][0]=par[i]
    for i in range(1,ms):
        for j in range(n):
            if pars[j][i-1]==-1:
                pars[j][i]=-1
            else:
                pars[j][i]=pars[pars[j][i-1]][i-1]
    for _ in range(qs):
        v,k = map(int, input().split())
        v-=1
        ks=k.bit_length()
        jvr=False
        for l in range(ks-1,-1,-1):
            if (k>>l)&1==1:
                if pars[v][l]==-1:
                    jvr=True
                else:
                    v=pars[v][l]
        if jvr:
            print(-1)
        else:
            print(v+1)
    

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

def dijkstra(n,start,edges):
    MAX = pow(10,10)
    dist = [MAX]*n
    adj=[[] for _ in range(n)]
    for (u,v,w) in edges:
        adj[u-1].append((v-1,w))
        adj[v-1].append((u-1,w))
    q=[(0,start-1)]
    heapq.heapify(q)
    fin=[0]*n
    while q:
        (d,v)=heapq.heappop(q)
        if fin[v]==0:
            fin[v]=1
            dist[v]=d
            for (u,w) in adj[v]:
                if fin[u]==0:
                    heapq.heappush(q,(d+w,u))
    return dist
                
