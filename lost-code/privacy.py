import math
import random
from itertools import permutations

def hamDist(alist, blist):
    numInd = min(len(alist),len(blist))
    count = 0
    for i in range(numInd):
        if alist[i] != blist[i]:
            count+=1
    return count

def assignProbs(alist):
    probs = {}
    val = alist[0]
    probs[val] = 1.0
    totalProb = 1
    i = 1
    while i < len(alist):
        j = 0
        scale = hamDist(alist[i],alist[j])
        upper = probs[alist[j]]*math.pow(eeps,scale)
        lower = probs[alist[j]]/math.pow(eeps,scale)
        j = 1
        while j < i:
            scale = hamDist(alist[i],alist[j])
            upper = min(upper, probs[alist[j]]*math.pow(eeps,scale))
            lower = max(lower, probs[alist[j]]/math.pow(eeps,scale))
            j+=1
        prob = random.uniform(lower, upper)
        val = alist[i]
        probs[val] = prob
        totalProb += prob
        i+=1
    for i in range(len(alist)):
        temp = probs[alist[i]]
        probs[alist[i]] = temp/totalProb
    return probs

def applyPerm(alist, blist):
    clist = []
    for i in range(len(alist)):
        clist.append(blist[alist[i]-1])
    return clist

def probOnce(tlist, slist, permlist, probs):
    total = 0
    for perm in permlist:
        if list(tlist) == applyPerm(slist,perm):
            total += probs[perm]
    return total

def probTwice(tlist, slist, permlist, probs):
    total = 0
    for perm1 in permlist:
        for perm2 in permlist:
            if list(tlist) == applyPerm(slist,applyPerm(perm1,perm2)):
                total += (probs[perm1]*probs[perm2])
    return total

numEle = 4

eps = 1
eeps = math.exp(eps)

permList = list(permutations(range(1, numEle+1)))
probList = assignProbs(permList)

allProbsOne = 0
allProbsTwo = 0

val = permList[0]
tlist = val
for perm in permList:
    slist = perm
    probOne = probOnce(tlist, slist, permList, probList)
    probTwo = probTwice(tlist, slist, permList, probList)
    allProbsOne += (probOne/math.factorial(numEle))
    allProbsTwo += (probTwo/math.factorial(numEle))

print(probList)
print(allProbsOne)
print(allProbsTwo)
