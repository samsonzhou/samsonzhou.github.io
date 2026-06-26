n=10**4
p=[0]*n
L=[[] for _ in range(n)]
for i in range(n):
    p[i]=i
    L[i].append(i)

def dsuget(i):
    if p[i] != i:
        p[i]=dsuget(p[i])
    return p[i]

def dsuunion(a,b):
    a=dsuget(a)
    b=dsuget(b)
    if p[a]!=p[b]:
        if len(L[a])>len(L[b]):
            for i in L[b]:
                p[i]=p[a]
            L[a].extend(L[b])
            L[b].clear()
        else:
            for i in L[a]:
                p[i]=p[b]
            L[b].extend(L[a])
            L[a].clear()
