import sys
from collections import defaultdict, deque, Counter
import math

# Overwrite standard input for fast I/O
input = sys.stdin.readline

# Increase recursion depth for deep trees/graphs (Codeforces default is often too low)
#sys.setrecursionlimit(200000)

def solve():
    """
    Main logic for a single test case.
    """

    # 2. Read multiple integers on a single line
    n, m = map(int, input().split())
    adjlist=[[] for i in range(n)]
    for i in range(m):
        u, v = map(int, input().split())
        u-=1
        v-=1
        (adjlist[u]).append(v)
        (adjlist[v]).append(u)
    print(bigbfs(n,m,adjlist))

    # 3. Read a list of integers
    # a = list(map(int, input().split()))
    
    # 4. Read a string (strip to remove the trailing newline character '\n')
    # s = input().strip()
    
    pass

def bigbfs(n,m,adjlist):
    visited = [-1]*n # denotes whether vertex has been found
    color = [-1]*n
    total = 0
    for i in range(n):
        if visited[i]==-1:
            cnt = [0,0]
            q = deque()
            q.append(i)
            visited[i] = 1
            bipartite = True
            while(len(q)>=1):
                v = q.pop()
                vc = color[v]
                if vc == -1:
                    vc = 0
                    color[v] = 0
                    cnt[0]+=1
                for u in adjlist[v]:
                    if color[u]!=-1 and color[u]!=(1-vc):
                        bipartite = False
                    elif visited[u]==-1:
                        q.append(u)
                        color[u] = 1-vc
                        cnt[color[u]]+=1
                        visited[u] = 1
                        #print(color, visited)
                        #print(cnt)
            if(bipartite):
                total+=max(cnt)
    return total

if __name__ == '__main__':
    # Most Codeforces problems have multiple test cases.
    # If a problem only has one test case, remove the loop and just call solve() once.
    t = int(input())
    for _ in range(t):
        solve()
