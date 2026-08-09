import sys
#from collections import defaultdict, deque, Counter
#import heapq
import math
import bisect

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

class SegTree:
    def __init__(self, n):
        self.n = n
        self.s = [0] * (2 * n)

    def update(self, pos, val):
        pos += self.n
        self.s[pos] = val
        pos //= 2
        while pos:
            self.s[pos] = max(self.s[pos * 2], self.s[pos * 2 + 1])
            pos //= 2

    def query(self, l, r):
        # max of a[l], a[l+1], ..., a[r-1]
        res = 0
        l += self.n
        r += self.n
        while l < r:
            if l & 1:
                res = max(res,self.s[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res,self.s[r])
            l //= 2
            r //= 2
        return res

    def root(self):
        return self.s[1]

    def upper_bound(self,i):
        L=0
        U=self.n
        while L<=U:
            M=(L+U)//2
            if self.query(0,M)<=i:
                L=M+1
            else:
                U=M-1
        return L-1
    
def solve():
    """
    Main logic for a single test case.
    """
    # 1. Read a single integer
    n = int(input())
    
    # 2. Read multiple integers on a single line
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    b = list(map(int, input().split()))
    pos=[]
    neg=[]
    neg.append(0)
    z=0
    tot=0
    for i in b:
        if i<0:
            neg.append(abs(i))
        elif i>0:
            pos.append(i)
        else:
            z+=1
        tot+=i

    if tot<=0:
        print(-1)
    else:
        pos.sort()
        neg.sort()
        out=[]
        tot=0
        j=0
        out.append(pos[j])
        tot=pos[j]
        j+=1
        k=len(neg)
        avail = SegTree(k)
        for i in range(k):
            avail.update(i,i)
        ind=bisect.bisect_right(neg,tot-1)
        ind=avail.query(0,ind)
        while ind!=0:
            val=neg[ind]
            tot-=val
            avail.update(ind,-1*10**9)
            out.append(tot)
            ind=bisect.bisect_right(neg,tot-1)
            ind=avail.query(0,ind)
        for i in range(z):
            out.append(tot)
        while len(out)<n:
            tot+=pos[j]
            j+=1
            out.append(tot)
            ind=bisect.bisect_right(neg,tot-1)
            ind=avail.query(0,ind)
            while ind!=0:
                val=neg[ind]
                tot-=val
                avail.update(ind,-1*10**9)
                out.append(tot)
                ind=bisect.bisect_right(neg,tot-1)
                ind=avail.query(0,ind)
        print(*out)
        
            
        
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    t = int(input())
    for _ in range(t):
        solve()
