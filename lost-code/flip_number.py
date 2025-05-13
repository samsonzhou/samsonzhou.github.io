data_stream=[1,1,1,1,1,2,3,4,1,5,1,2,1,2,1,2,1,1,1,1,6,1,1,1,7]
f = [0 for i in range(100)]
eps = 1/100

p=1.5

prev_fp=0
prev_resid=0

fp=0
resid=0

c_fp=0
c_resid=0

list_hh = []

for u in data_stream:
    f[u]+=1
    fp+=(f[u]**p-(f[u]-1)**p)
    if fp > (1+eps)*prev_fp:
        prev_fp=fp
        c_fp+=1
    if u not in list_hh:
        #u is new HH
        if f[u]>=eps*fp:
            resid -= (f[u]-1)**p
            list_hh.append(u)
        else:
            resid+=(f[u]**p-(f[u]-1)**p)
    for other_hh in list_hh:
        if f[other_hh]<eps*fp:
            list_hh.remove(other_hh)
            resid+=f[other_hh]**p
    if resid > (1+eps)*prev_resid:
        prev_resid=resid
        c_resid+=1

print(c_fp,c_resid)
            
    
