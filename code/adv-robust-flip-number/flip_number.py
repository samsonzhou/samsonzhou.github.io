import matplotlib.pyplot as plt
import time

#universe size
n = 1000000

#p=1.5

file = open("125910_ip_timestamps.csv","r")
lines = file.readlines()[1:]

flip_fp = []
flip_resid = []
ratios = []

#acc_list = [1,2,3,4,5]
#eps = 1/10000
#for acc_power in acc_list:
    #acc = 10**(-1*acc_power)

#eps_list = [1,2,3,4,5,6,7,8,9,10]
#acc=0.001
#for eps_power in eps_list:
#    eps = 4**(-1*eps_power)

acc = 0.00001
eps = 0.00001
#p_list = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
p_list=[1.9]
for p in p_list:
    prev_fp=0
    prev_resid=0

    fp=0
    resid=0

    c_fp=0
    c_resid=0

    list_hh = []
    steps = 0
    f = [0 for i in range(n)]
    for line in lines:
        steps += 1
        s = line[:-1]
        u = hash(s)%n
        f[u]+=1
        fp+=(f[u]**p-(f[u]-1)**p)
        if fp > (1+acc)*prev_fp:
            prev_fp=fp
            c_fp+=1
        if u not in list_hh:
            #u is new HH
            if f[u]**p>=eps*fp:
                resid -= (f[u]-1)**p
                list_hh.append(u)
            else:
                resid+=(f[u]**p-(f[u]-1)**p)
        if steps > 1/eps:
            steps = 0
            for other_hh in list_hh:
                if f[other_hh]**p<eps*fp:
                    list_hh.remove(other_hh)
                    resid+=f[other_hh]**p
        if resid > (1+acc)*prev_resid:
            prev_resid=resid
            c_resid+=1
    flip_fp.append(c_fp)
    flip_resid.append(c_resid)
    if c_resid == 0:
        ratio = 0
    else:
        ratio = c_fp/c_resid
    ratios.append(ratio)

#plt.plot(acc_list, flip_fp, label="Fp")
#plt.plot(acc_list, flip_resid, label="Residual")
#plt.xticks(acc_list)
#plt.plot(eps_list, flip_fp, label="Fp")
#plt.plot(eps_list, flip_resid, label="Residual")
#plt.xlim(1,10)
#plt.xticks(eps_list)
#plt.yscale('log')
#plt.xlabel('HH parameter (inverse log)')
plt.plot(p_list, flip_fp, label="Fp")
plt.plot(p_list, flip_resid, label="Residual")
plt.xlim(1.1,1.9)
plt.xticks(p_list)
plt.xlabel('p')
plt.ylabel('Flip number')
plt.legend(loc="best")
plt.grid(True)
plt.show()

#ratios = [1.1741573033707866, 1.2704309063893016,
#1.4867452135493373, 1.6550998163438997, 1.7491483904690608]
#FOR ACC=0.1
#flip_fp=[208, 208, 208, 208, 208, 208, 208, 208, 208, 208]
#flip_resid=[201, 194, 199, 194, 191, 183, 173, 165, 157, 149]
#FOR ACC=0.001
#flip_fp=[14017, 14017, 14017, 14017, 14017, 14017, 14017, 14017, 14017, 14017]
#flip_resid=[13988, 12657, 12508, 11379, 11001, 10237, 9521, 9088, 8941, 8845]
#FOR P
#flip_fp=[11530, 12165, 12809, 13436, 14013, 14589, 15175, 15745, 16320]
#flip_resid=[9286, 9716, 9970, 10575, 10917, 11597, 11952, 12277, 12466]
