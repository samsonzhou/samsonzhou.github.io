import sys
#from collections import defaultdict, deque, Counter
#import heapq
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

mod = 998244353
imag = 911660635
iimag = 86583718
rate2 = (911660635, 509520358, 369330050, 332049552, 983190778, 123842337, 238493703, 975955924, 603855026, 856644456, 131300601,
              842657263, 730768835, 942482514, 806263778, 151565301, 510815449, 503497456, 743006876, 741047443, 56250497, 867605899)
irate2 = (86583718, 372528824, 373294451, 645684063, 112220581, 692852209, 155456985, 797128860, 90816748, 860285882, 927414960,
               354738543, 109331171, 293255632, 535113200, 308540755, 121186627, 608385704, 438932459, 359477183, 824071951, 103369235)
rate3 = (372528824, 337190230, 454590761, 816400692, 578227951, 180142363, 83780245, 6597683, 70046822, 623238099,
              183021267, 402682409, 631680428, 344509872, 689220186, 365017329, 774342554, 729444058, 102986190, 128751033, 395565204)
irate3 = (509520358, 929031873, 170256584, 839780419, 282974284, 395914482, 444904435, 72135471, 638914820, 66769500,
               771127074, 985925487, 262319669, 262341272, 625870173, 768022760, 859816005, 914661783, 430819711, 272774365, 530924681)
 
def butterfly(a):
    n = len(a)
    h = (n - 1).bit_length()
    len_ = 0
    while len_ < h:
        if h - len_ == 1:
            p = 1 << (h - len_ - 1)
            rot = 1
            for s in range(1 << len_):
                offset = s << (h - len_)
                for i in range(p):
                    l = a[i + offset]
                    r = a[i + offset + p] * rot % mod
                    a[i + offset] = (l + r) % mod
                    a[i + offset + p] = (l - r) % mod
                if s + 1 != 1 << len_:
                    rot *= rate2[(~s & -~s).bit_length() - 1]
                    rot %= mod
            len_ += 1
        else:
            p = 1 << (h - len_ - 2)
            rot = 1
            for s in range(1 << len_):
                rot2 = rot * rot % mod
                rot3 = rot2 * rot % mod
                offset = s << (h - len_)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p] * rot
                    a2 = a[i + offset + p * 2] * rot2
                    a3 = a[i + offset + p * 3] * rot3
                    a1na3imag = (a1 - a3) % mod * imag
                    a[i + offset] = (a0 + a2 + a1 + a3) % mod
                    a[i + offset + p] = (a0 + a2 - a1 - a3) % mod
                    a[i + offset + p * 2] = (a0 - a2 + a1na3imag) % mod
                    a[i + offset + p * 3] = (a0 - a2 - a1na3imag) % mod
                if s + 1 != 1 << len_:
                    rot *= rate3[(~s & -~s).bit_length() - 1]
                    rot %= mod
            len_ += 2
 
def butterfly_inv(a):
    n = len(a)
    h = (n - 1).bit_length()
    len_ = h
    while len_:
        if len_ == 1:
            p = 1 << (h - len_)
            irot = 1
            for s in range(1 << (len_ - 1)):
                offset = s << (h - len_ + 1)
                for i in range(p):
                    l = a[i + offset]
                    r = a[i + offset + p]
                    a[i + offset] = (l + r) % mod
                    a[i + offset + p] = (l - r) * irot % mod
                if s + 1 != (1 << (len_ - 1)):
                    irot *= irate2[(~s & -~s).bit_length() - 1]
                    irot %= mod
            len_ -= 1
        else:
            p = 1 << (h - len_)
            irot = 1
            for s in range(1 << (len_ - 2)):
                irot2 = irot * irot % mod
                irot3 = irot2 * irot % mod
                offset = s << (h - len_ + 2)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p]
                    a2 = a[i + offset + p * 2]
                    a3 = a[i + offset + p * 3]
                    a2na3iimag = (a2 - a3) * iimag % mod
                    a[i + offset] = (a0 + a1 + a2 + a3) % mod
                    a[i + offset + p] = (a0 - a1 + a2na3iimag) * irot % mod
                    a[i + offset + p * 2] = (a0 + a1 - a2 - a3) * irot2 % mod
                    a[i + offset + p * 3] = (a0 - a1 - a2na3iimag) * irot3 % mod
                if s + 1 != (1 << (len_ - 2)):
                    irot *= irate3[(~s & -~s).bit_length() - 1]
                    irot %= mod
            len_ -= 2
 
def convolve(a, b):
    a = a.copy()
    b = b.copy()
    n = len(a)
    m = len(b)
    z = 1 << (n + m - 2).bit_length()
    a += [0] * (z - n)
    butterfly(a)
    b += [0] * (z - m)
    butterfly(b)
    for i in range(z):
        a[i] = a[i] * b[i] % mod
    butterfly_inv(a)
    a = a[:n + m - 1]
    iz = pow(z, mod - 2, mod)
    for i in range(n + m - 1):
        a[i] = a[i] * iz % mod
    return a


def solve():
    """
    Main logic for a single test case.
    """
    # 1. Read a single integer
    # n = int(input())
    
    # 2. Read multiple integers on a single line
    # n, m = map(int, input().split())
    
    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    n, k = map(int, input().split())
    ds = list(map(int, input().split()))
    p=[]
    for i in range(10):
        if i in ds:
            p.append(1)
        else:
            p.append(0)

    t=n.bit_length()-1
    out=[1]
    m=n//2
    for i in range(t+1):
        if m>>i&1==1:
            out=convolve(out,p)
        if i==t:
            continue
        p=convolve(p,p)
    tot=0
    for i in out:
        tot=(tot+pow(i,2,998244353))%998244353
    print(tot)
#Booth's algorithm
#Finds first lexicographically ordered cyclic shift of a string s
#Essentially iterate over ss (s repeated twice) and KMP-style search
def least_rotation(s):
    s = s + s
    n = len(s) // 2

    i, j, k = 0, 1, 0

    while i < n and j < n and k < n:
        if s[i + k] == s[j + k]:
            k += 1
            continue

        if s[i + k] > s[j + k]:
            i = i + k + 1
            if i <= j:
                i = j + 1
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1

        k = 0

    start = min(i, j)
    return s[start:start + n]

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

#from tryingoutcp
def dfs_iterative(graph, root, parent, sz):
    n = len(graph)
 
    visited = [0] * n
    order = []
    stack = [root]
    visited[root] = 1
    while stack:
        node = stack.pop()
        order.append(node)
 
        for neighbor in graph[node]:
            if not visited[neighbor]:
                visited[neighbor] = 1
                parent[neighbor] = node
                stack.append(neighbor)
 
    for node in reversed(order):
        sz[node] = 1
        for neighbor in graph[node]:
            if neighbor != parent[node]:
                sz[node] += sz[neighbor]

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
