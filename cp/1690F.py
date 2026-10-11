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
    n = int(input())
    
    # 2. Read multiple integers on a single line
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    s = input().strip()
    p = list(map(int, input().split()))
    ps = [0]*n
    for i in range(n):
        ps[p[i]-1]=i
    cycs = []
    exp=[0]*n
    for i in range(n):
        if exp[i]==1:
            continue
        arr = []
        arr.append(i)
        c = ps[i]
        exp[i]=1
        while c!=i:
            arr.append(c)
            c = ps[c]
            exp[c]=1
        cycs.append(arr)
    ans=1
    for cyc in cycs:
        target = len(cyc)
        fnd = False
        i = 1
        while not fnd:
            jvr = False
            for j in range(len(cyc)):
                if s[cyc[j]]!=s[cyc[(j+i)%target]]:
                    jvr = True
                    break
            if not jvr:
                ans = math.lcm(i, ans)
                fnd = True
            i+=1
    print(ans)
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    t = int(input())
    for _ in range(t):
        solve()


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
            heapq.heappush(q,(d+w,u))
    return dist
                
