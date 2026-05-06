import numpy as np
from scipy.optimize import linprog
import math
import matplotlib.pyplot as plt

def generate_data(s, n_i, d):
    """Generates Chebyshev (l_inf) robust regression data."""
    D = d + 1
    c = np.zeros(D)
    c[-1] = 1.0 # Minimize maximum error t
    
    servers_A, servers_b = [], []
    w_true = np.random.randn(d)
    
    for _ in range(s):
        X = np.random.randn(n_i, d)
        y = X @ w_true + 0.1 * np.random.randn(n_i)
        
        # Chebyshev constraints: Xw - t <= y, -Xw - t <= -y
        A1 = np.hstack([X, -np.ones((n_i, 1))])
        A2 = np.hstack([-X, -np.ones((n_i, 1))])
        
        servers_A.append(np.vstack([A1, A2]))
        servers_b.append(np.concatenate([y, -y]))
        
    return servers_A, servers_b, D, c

def simulate_vww20(servers_A, servers_b, D, c):
    """Simulates VWW20 (Standard Distributed Clarkson) baseline."""
    flat_A = np.vstack(servers_A)
    flat_b = np.concatenate(servers_b)
    N = len(flat_b)
    weights = np.ones(N)
    
    global_pings = 0
    A_box = np.vstack([np.eye(D), -np.eye(D)])
    b_box = 1000 * np.ones(2 * D)
    
    while True:
        global_pings += 1
        W = np.sum(weights)
        
        sample_size = min(N, 9 * D**2)
        idx = np.random.choice(N, sample_size, p=weights/W, replace=True)
        
        A_sub = np.vstack([flat_A[idx], A_box])
        b_sub = np.concatenate([flat_b[idx], b_box])
        
        res = linprog(c, A_ub=A_sub, b_ub=b_sub, bounds=(None, None), method='highs')
        if not res.success: return global_pings
            
        x_star = res.x
        violations = (flat_A @ x_star - flat_b > 1e-6)
        violating_indices = np.where(violations)[0]
        
        if len(violating_indices) == 0:
            return global_pings
            
        W_V = np.sum(weights[violating_indices])
        if W_V <= (2 / (9*D + 1)) * W:
            weights[violating_indices] *= 2
        
        if global_pings > 150: break # Failsafe
            
    return global_pings

def check_violation(A, b, x, tol=1e-6):
    return np.any(A @ x - b > tol)

def simulate_hybrid_clarkson(servers_A, servers_b, D, c):
    """Simulates the proposed HybridClarkson method (Algorithms 8 & 9)."""
    s = len(servers_A)
    T_indices = set()
    global_pings = 0
    
    A_box = np.vstack([np.eye(D), -np.eye(D)])
    b_box = 1000 * np.ones(2 * D)
    
    for _ in range(50): # Failsafe outer loop
        sample_size = min(s, math.ceil(D * math.sqrt(s)))
        avail = list(set(range(s)) - T_indices)
        take = min(len(avail), sample_size)
        if take > 0:
            R_indices = set(np.random.choice(avail, take, replace=False))
        else:
            R_indices = set()
            
        S_prime = list(R_indices.union(T_indices))
        if len(S_prime) == 0: S_prime = list(range(s))
        
        w = np.ones(len(S_prime))
        
        for _ in range(100): # Inner WeightedClarkson
            W = np.sum(w)
            B_size = min(len(S_prime), 6 * D**2)
            B_indices = np.random.choice(len(S_prime), B_size, p=w/W, replace=True)
            B_servers = [S_prime[i] for i in B_indices]
            
            A_sub = np.vstack([servers_A[i] for i in B_servers] + [A_box])
            b_sub = np.concatenate([servers_b[i] for i in B_servers] + [b_box])
            
            res = linprog(c, A_ub=A_sub, b_ub=b_sub, bounds=(None, None), method='highs')
            if not res.success: break
            
            V_inner, W_V = [], 0
            for idx, s_idx in enumerate(S_prime):
                if check_violation(servers_A[s_idx], servers_b[s_idx], res.x):
                    V_inner.append(idx)
                    W_V += w[idx]
                    
            if len(V_inner) == 0: break
            if W_V <= W / (3 * D):
                for idx in V_inner: w[idx] *= 2
        
        global_pings += 1 # Global DSO Check
        
        V_global = set()
        for i in range(s):
            if check_violation(servers_A[i], servers_b[i], res.x):
                V_global.add(i)
                
        if len(V_global) == 0:
            return global_pings
            
        if len(V_global) <= 2 * D * math.sqrt(s):
            T_indices = T_indices.union(V_global)
            
    return global_pings

if __name__ == "__main__":
    print("Running Experiment A: Varying Local Dataset Size (n_i)...")
    s_fixed, d_fixed = 50, 3
    n_list = [10, 50, 100, 200, 500]
    
    ours_pings_A, vww_pings_A = [], []
    np.random.seed(42)
    for n in n_list:
        A, b, D, c = generate_data(s_fixed, n, d_fixed)
        p_o = np.mean([simulate_hybrid_clarkson(A, b, D, c) for _ in range(3)])
        p_v = np.mean([simulate_vww20(A, b, D, c) for _ in range(3)])
        ours_pings_A.append(p_o); vww_pings_A.append(p_v)
        print(f"  n={n} | Ours: {p_o:.1f} | VWW20: {p_v:.1f}")
        
    print("\nRunning Experiment B: Varying Number of Servers (s)...")
    n_fixed = 50
    s_list = [20, 50, 100, 200, 300]
    
    ours_pings_B, vww_pings_B = [], []
    for s_val in s_list:
        A, b, D, c = generate_data(s_val, n_fixed, d_fixed)
        p_o = np.mean([simulate_hybrid_clarkson(A, b, D, c) for _ in range(3)])
        p_v = np.mean([simulate_vww20(A, b, D, c) for _ in range(3)])
        ours_pings_B.append(p_o); vww_pings_B.append(p_v)
        print(f"  s={s_val} | Ours: {p_o:.1f} | VWW20: {p_v:.1f}")
        
    # --- Plotting ---
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    axes[0].plot(n_list, ours_pings_A, marker='o', linewidth=2, label='HybridClarkson (Ours)', color='#348ABD')
    axes[0].plot(n_list, vww_pings_A, marker='s', linewidth=2, label='VWW20 (Baseline)', color='#E24A33')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Local Constraints per Server ($n_i$)')
    axes[0].set_ylabel('Global Synchronization Rounds')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Exp A: Decoupling $n$ from Sync Rounds')

    axes[1].plot(s_list, ours_pings_B, marker='o', linewidth=2, label='HybridClarkson (Ours)', color='#348ABD')
    axes[1].plot(s_list, vww_pings_B, marker='s', linewidth=2, label='VWW20 (Baseline)', color='#E24A33')
    axes[1].set_xlabel('Number of Distributed Servers ($s$)')
    axes[1].set_ylabel('Global Synchronization Rounds')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Exp B: Scaling with Network Size')

    plt.tight_layout()
    plt.savefig('neurips_experiments.pdf', bbox_inches='tight')
    print("\nSuccess! Plots saved to 'neurips_experiments.pdf'.")
