import sys
#from collections import defaultdict, deque, Counter
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

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
    t = int(input())
    for _ in range(t):
        solve()

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
def all_primes_up_to(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            # mark multiples starting from p*p
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False

    primes = [i for i, val in enumerate(is_prime) if val]
    return primes

#Subroutine for creating a list of all divisors of all integers up to n
#Sieve approach, adds each integer to the list of the divisors of its multiples
def all_divisors_up_to(n):
    divs = [[] for _ in range(n+1)]

    for y in range(1, n+1):
        for multiple in range(y, n+1, y):
            divs[multiple].append(y)
    return divs

#Subroutine for finding prime divisors of an integer n
#Runs in time O(sqrt(n))
def prime_divisors(n):
    res = []

    d = 2
    while d * d <= n:
        if n % d == 0:
            res.append(d)
            while n % d == 0:
                n //= d
        d += 1

    if n > 1:
        res.append(n)

    return res

#Subroutine for returning median of a list b
#Runs in O(n log n) time due to sort
def median(b):
    a=sorted(b)
    if len(a)&1==1:
        return(a[len(a)//2])
    else:
        return((a[len(a)//2]+a[len(a)//2-1])/2)

#code for primes
big=10**6
ls=[-1]*big
for i in range(2,big):
    if ls[i]==-1:
        for j in range(i*i,big,i):
            if ls[j]==-1:
                ls[j]=i
ls[0]=0
ls[1]=1
primes=[j for j in range(big) if ls[j]==-1]
