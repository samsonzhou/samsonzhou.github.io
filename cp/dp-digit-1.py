import sys
#from collections import defaultdict, deque, Counter
import heapq
import math
sys.setrecursionlimit(10000)
# Overwrite standard input for fast I/O
input = sys.stdin.readline


if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    mod,dmag = map(int, input().split())
    a = input().strip()
    L = len(a)
    MOD = pow(10,9)+7
    dp = [[[-1]*mod for _ in range(2)] for _ in range(L+1)]
    def go(pos,tar,tight,m):
        """
        Main logic for a single test case.
        """
        n=len(tar)
        if pos>=n:
            return 1 if m==0 else 0
        if dp[pos][tight][m]!=-1:
            return dp[pos][tight][m]
        lim = int(tar[pos]) if tight else 9
        tot = 0
        if pos+1==n:
            for d in range(lim+1):
               if d!=dmag and (m*10+d)%mod==0:
                   tot+=1
        else:
            for d in range(lim+1):
                if d==dmag:
                    continue
                new_tight = tight and d==lim
                if new_tight:
                    lim2 = int(tar[pos+1]) if tight else 9
                    if dmag>lim2:
                        continue
                    new_tight = (dmag == lim2)
                new_m = (100*m+d*10+dmag)%mod
                tot += go(pos+2,tar,new_tight,new_m)
        dp[pos][tight][m]=tot%MOD
        return tot%MOD
    
    #dp = [[[-1]*mod for _ in range(2)] for _ in range(L+1)]
    jvr=False
    if int(a)%mod!=0:
        jvr=True
    else:
        for i in range(len(a)):
            if i&1==0 and int(a[i])==dmag:
                jvr=True
                break
            if i&1==1 and int(a[i])!=dmag:
                jvr=True
                break
    
    aans = go(0,a,True,0)
    for p in range(L+1):          # reset only the tight layer; non-tight results are reused for b
        dp[p][1] = [-1]*mod
    
    b = input().strip()
    bans = go(0,b,True,0)
    if jvr:
        off=0
    else:
        off=1
    #print(bans)
    print((bans-aans+off)%MOD)

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
                
