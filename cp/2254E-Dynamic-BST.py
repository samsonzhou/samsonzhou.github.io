import sys
#from collections import defaultdict, deque, Counter
#import heapq
import math
import bisect

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)



"""
The "sorted list" data-structure, with amortized O(n^(1/3)) cost per insert and pop.

Example:

A = SortedList()
A.insert(30)
A.insert(50)
A.insert(20)
A.insert(30)
A.insert(30)

print(A) # prints [20, 30, 30, 30, 50]

print(A.lower_bound(30), A.upper_bound(30)) # prints 1 4

print(A[-1]) # prints 50
print(A.pop(1)) # prints 30

print(A) # prints [20, 30, 30, 50]
print(A.count(30)) # prints 2

"""

from bisect import bisect_left as lower_bound
from bisect import bisect_right as upper_bound


class FenwickTree:
    def __init__(self, x):
        bit = self.bit = list(x)
        size = self.size = len(bit)
        for i in range(size):
            j = i | (i + 1)
            if j < size:
                bit[j] += bit[i]

    def update(self, idx, x):
        """updates bit[idx] += x"""
        while idx < self.size:
            self.bit[idx] += x
            idx |= idx + 1

    def __call__(self, end):
        """calc sum(bit[:end])"""
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def find_kth(self, k):
        """Find largest idx such that sum(bit[:idx]) <= k"""
        idx = -1
        for d in reversed(range(self.size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k


class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        iterable = sorted(iterable)
        self.micros = [iterable[i:i + self.block_size - 1] for i in range(0, len(iterable), self.block_size - 1)] or [[]]
        self.macro = [i[0] for i in self.micros[1:]]
        self.micro_size = [len(i) for i in self.micros]
        self.fenwick = FenwickTree(self.micro_size)
        self.size = len(iterable)

    def insert(self, x):
        i = lower_bound(self.macro, x)
        j = upper_bound(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self.fenwick.update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i:i + 1] = self.micros[i][:self.block_size >> 1], self.micros[i][self.block_size >> 1:]
            self.micro_size[i:i + 1] = self.block_size >> 1, self.block_size >> 1
            self.fenwick = FenwickTree(self.micro_size)
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.fenwick.update(i, -1)
        return self.micros[i].pop(j)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.upper_bound(x) - self.lower_bound(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def lower_bound(self, x):
        i = lower_bound(self.macro, x)
        return self.fenwick(i) + lower_bound(self.micros[i], x)

    def upper_bound(self, x):
        i = upper_bound(self.macro, x)
        return self.fenwick(i) + upper_bound(self.micros[i], x)

    def _find_kth(self, k):
        return self.fenwick.find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))


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
    neg=SortedList()
    neg.insert(0)
    z=0
    tot=0
    for i in b:
        if i<0:
            neg.insert(abs(i))
        elif i>0:
            pos.append(i)
        else:
            z+=1
        tot+=i

    if tot<=0:
        print(-1)
    else:
        pos.sort()
        out=[]
        tot=0
        j=0
        out.append(pos[j])
        tot=pos[j]
        j+=1
        ind=neg.upper_bound(tot-1)-1
        while ind!=0:
            tot-=neg.pop(ind)
            out.append(tot)
            ind=neg.upper_bound(tot-1)-1
        for i in range(z):
            out.append(tot)
        while len(out)<n:
            tot+=pos[j]
            j+=1
            out.append(tot)
            ind=neg.upper_bound(tot-1)-1
            while ind!=0:
                tot-=neg.pop(ind)
                out.append(tot)
                ind=neg.upper_bound(tot-1)-1
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
