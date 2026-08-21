#TEST CODE FOR NTT
import sys
 
# import time
import bisect
# import functools
import math
import random
# import re
from collections import Counter, defaultdict, deque

# Overwrite standard input for fast I/O
input = sys.stdin.readline

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


# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

class FFT:
    def __init__(self, MOD=998244353):
        FFT.MOD = MOD
        g = self.primitive_root_constexpr()
        ig = pow(g, FFT.MOD - 2, FFT.MOD)
        FFT.W = [pow(g, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
        FFT.iW = [pow(ig, (FFT.MOD - 1) >> i, FFT.MOD) for i in range(30)]
 
    def primitive_root_constexpr(self):
        if FFT.MOD == 998244353:
            return 3
        elif FFT.MOD == 200003:
            return 2
        elif FFT.MOD == 167772161:
            return 3
        elif FFT.MOD == 469762049:
            return 3
        elif FFT.MOD == 754974721:
            return 11
        divs = [0] * 20
        divs[0] = 2
        cnt = 1
        x = (FFT.MOD - 1) // 2
        while x % 2 == 0:
            x //= 2
        i = 3
        while i * i <= x:
            if x % i == 0:
                divs[cnt] = i
                cnt += 1
                while x % i == 0:
                    x //= i
            i += 2
        if x > 1:
            divs[cnt] = x
            cnt += 1
        g = 2
        while 1:
            ok = True
            for i in range(cnt):
                if pow(g, (FFT.MOD - 1) // divs[i], FFT.MOD) == 1:
                    ok = False
                    break
            if ok:
                return g
            g += 1
 
    def fft(self, k, f):
        for l in range(k, 0, -1):
            d = 1 << l - 1
            U = [1]
            for i in range(d):
                U.append(U[-1] * FFT.W[l] % FFT.MOD)
 
            for i in range(1 << k - l):
                for j in range(d):
                    s = i * 2 * d + j
                    f[s], f[s + d] = (f[s] + f[s + d]) % FFT.MOD, U[j] * (f[s] - f[s + d]) % FFT.MOD
 
    def ifft(self, k, f):
        for l in range(1, k + 1):
            d = 1 << l - 1
            for i in range(1 << k - l):
                u = 1
                for j in range(i * 2 * d, (i * 2 + 1) * d):
                    f[j+d] *= u
                    f[j], f[j + d] = (f[j] + f[j + d]) % FFT.MOD, (f[j] - f[j + d]) % FFT.MOD
                    u = u * FFT.iW[l] % FFT.MOD
 
    def convolve(self, A, B):
        A = A[:]
        B = B[:]
        n0 = len(A) + len(B) - 1
        k = (n0).bit_length()
        n = 1 << k
        A += [0] * (n - len(A))
        B += [0] * (n - len(B))
        self.fft(k, A)
        self.fft(k, B)
        A = [a * b % FFT.MOD for a, b in zip(A, B)]
        self.ifft(k, A)
        inv = pow(n, FFT.MOD - 2, FFT.MOD)
        A = [a * inv % FFT.MOD for a in A]
        del A[n0:]
        return A

a=[1,2,3] #a(x) = 1 + 2x + 3x^2
b=[4,5] #b(x) = 4 + 5x
fft=FFT()
c=fft.convolve(a,b) #c(x) = a(x)*b(x) = 4 + 13x + 22x^2 + 15x^3
print(c)
#[4, 13, 22, 15]
