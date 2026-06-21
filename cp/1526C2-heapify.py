import heapq

n = int(input())
arr = list(map(int, input().split()))

heap = []
total = 0

for x in arr:
    total += x

    if total >= 0:
        # keep x
        heapq.heappush(heap, x)
    else:
        # total became negative, so remove smallest chosen element
        removed = heapq.heappushpop(heap, x)
        total -= removed

print(len(heap))
