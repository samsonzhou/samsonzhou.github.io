#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using pii = pair<int, int>;
using pll = pair<ll, ll>;

#define all(x) begin(x), end(x)
#define rall(x) rbegin(x), rend(x)

void solve() {

}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Most Codeforces problems have multiple test cases.
    // For a single test case, replace this with: solve();
        /*
    Main logic for one test case.
    */

    // 1. Read a single integer
    // int n;
    // cin >> n;

    // 4. Read a string
    string s;
    cin >> s;
    string t;
    cin >> t;
    int n = s.length();
    vector<int> a(n);
    for(int i=0; i<n; i++){
        a[i]=s[i]^t[i];
    }
    for (int x : a){
        cout << x;
    }
    cout << '\n';

    // 5. Read a full line, including spaces
    // string s;
    // getline(cin >> ws, s);
    return 0;
}