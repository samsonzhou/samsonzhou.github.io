subvec = [1,2,3,4,5,6,7,8,9,10]
power = 1
vec = []
for i in range(power):
    vec+=[4**i*j for j in subvec]
N=len(vec)
s=10
#N=10000
#s=500

sets=set()

M=max(vec)*10
#M=40320
numRemove = 0
for i in range(M):
    x = i/M
    total = 0
    for j in vec:
        total += ((j*x)%1)**2
    if total <= s:
        rounded_vec = [round((j*x)%1) for j in vec]
        set_vec = tuple(rounded_vec)
        if set_vec not in sets:
            print(x)
            print([j*x for j in vec])
            print(set_vec)
            input()
            sets.add(set_vec)
        numRemove += 1

list_vec = list(sets)
print(len(list_vec))
