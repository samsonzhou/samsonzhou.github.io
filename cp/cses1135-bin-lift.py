import sys
#from collections import defaultdict, deque, Counter

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

n, qs = map(int, input().split())
par=[-1]*n
adj=[[] for _ in range(n)]
for _ in range(n-1):
    a, b = map(int, input().split())
    adj[a-1].append(b-1)
    adj[b-1].append(a-1)
lvls=n.bit_length()

def pval(v,k):
    u=v
    for i in range(lvls-1,-1,-1):
        if (k>>i)&1==1:
            u=pars[i][u]
            if u==-1:
                break
    return(u)
    
deep=[-1]*n
deep[0]=0
q=[0]
while q:
    v=q.pop()
    for u in adj[v]:
        if deep[u]==-1:
            q.append(u)
            deep[u]=deep[v]+1
            par[u]=v
pars=[[-1]*n for _ in range(lvls)]
pars[0]=par[:]
for i in range(1,lvls):
    for j in range(1,n):
        if pars[i-1][j]==-1:
            pars[i][j]=-1
        else:
            pars[i][j]=pars[i-1][pars[i-1][j]]

for _ in range(qs):
    uo,vo = map(int, input().split())
    u=uo-1
    v=vo-1
    if deep[u]>deep[v]:
        h=deep[u]-deep[v]
        u=pval(u,h)
    else:
        h=deep[v]-deep[u]
        v=pval(v,h)
    if u!=v:
        for i in range(lvls-1,-1,-1):
            x,y=pars[i][u],pars[i][v]
            if x!=y:
                u=x
                v=y
        u=pars[0][u]
    print(deep[uo-1]+deep[vo-1]-2*deep[u])
        
