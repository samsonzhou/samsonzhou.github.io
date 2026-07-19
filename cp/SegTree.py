class SegTree:
    def __init__(self, n):
        self.n = n
        self.s = [0] * (2 * n)

    def update(self, pos, val):
        pos += self.n
        self.s[pos] = val
        pos //= 2
        while pos:
            self.s[pos] = self.s[pos * 2] + self.s[pos * 2 + 1]
            pos //= 2

    def query(self, l, r):
        # sum of a[l], a[l+1], ..., a[r-1]
        res = 0
        l += self.n
        r += self.n
        while l < r:
            if l & 1:
                res += self.s[l]
                l += 1
            if r & 1:
                r -= 1
                res += self.s[r]
            l //= 2
            r //= 2
        return res

    def root(self):
        return self.s[1]

a = [1, 2, 4, 3, 5, 1, 0, -1, 1]
st = SegTree(len(a))
for i in range(len(a)):
    st.update(i, a[i])

print(st.query(0, len(a)))  # 16  (total sum, same as st.root())
print(st.query(2, 5))       # 4 + 3 + 5 = 12
print(st.query(6, 9))       # 0 + -1 + 1 = 0
