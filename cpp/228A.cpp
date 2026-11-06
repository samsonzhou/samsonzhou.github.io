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

    // 2. Read multiple integers
    // int n, m;
    // cin >> n >> m;

    // 3. Read a list of integers
    vector<int> a(4);
    for (int& x : a) cin >> x;
    sort(a.begin(), a.end());
    int c=0;
    for (int i=1; i<4; i++){
        if(a[i]==a[i-1]){
            c++;
        }
    }
    cout << c << '\n';

    // 4. Read a string
    // string s;
    // cin >> s;

    // 5. Read a full line, including spaces
    // string s;
    // getline(cin >> ws, s);
    return 0;
}