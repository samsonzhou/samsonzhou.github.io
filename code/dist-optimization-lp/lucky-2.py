import numpy as np
from scipy.optimize import linprog
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_data(s, n_i, d):
    """Generates federated Chebyshev (l_inf) robust regression data."""
    D = d + 1
    c = np.zeros(D); c[-1] = 1.0 
    servers_A, servers_b = [], []
    w_true = np.random.randn(d)
    
    for _ in range(s):
        X = np.random.randn(n_i, d)
        y = X @ w_true + 0.1 * np.random.randn(n_i)
        servers_A.append(np.vstack([np.hstack([X, -np.ones((n_i, 1))]), np.hstack([-X, -np.ones((n_i, 1))])]))
        servers_b.append(np.concatenate([y, -y]))
        
    return servers_A, servers_b, D, c

def simulate_vww20(servers_A, servers_b, D, c, is_global=True):
    """Simulates VWW20. Acts as the Baseline AND the recursive FindOPT oracle."""
    s = len(servers_A)
    flat_A = np.vstack(servers_A)
    flat_b = np.concatenate(servers_b)
    N = len(flat_b)
    weights = np.ones(N)
    
    metrics = {'pings': 0, 'constraints_sent': 0, 'oracle_queries': 0}
    A_box = np.vstack([np.eye(D), -np.eye(D)])
    b_box = 1000 * np.ones(2 * D)
    
    for _ in range(200): # Failsafe
        if is_global: metrics['pings'] += 1
        metrics['oracle_queries'] += s  # All servers evaluate local constraints
        
        W = np.sum(weights)
        sample_size = min(N, 9 * D**2)
        
        # BANDWIDTH ACCOUNTING: Coordinator fetches constraints from servers
        metrics['constraints_sent'] += sample_size
        
        idx = np.random.choice(N, sample_size, p=weights/W, replace=True)
        A_sub = np.vstack([flat_A[idx], A_box])
        b_sub = np.concatenate([flat_b[idx], b_box])
        
        res = linprog(c, A_ub=A_sub, b_ub=b_sub, bounds=(None, None), method='highs')
        if not res.success: return res.x if hasattr(res, 'x') and res.x is not None else np.zeros(D), metrics
        x_star = res.x
        
        violations = (flat_A @ x_star - flat_b > 1e-6)
        v_idx = np.where(violations)[0]
        if len(v_idx) == 0: return x_star, metrics
            
        W_V = np.sum(weights[v_idx])
        if W_V <= (2 / (9*D + 1)) * W: weights[v_idx] *= 2
            
    return np.zeros(D), metrics

def check_violation(A, b, x, tol=1e-6):
    return np.any(A @ x - b > tol)

def simulate_hybrid_clarkson(servers_A, servers_b, D, c):
    """Simulates HybridClarkson using recursive FindOPT tracking."""
    s = len(servers_A)
    T_indices = set()
    metrics = {'pings': 0, 'constraints_sent': 0, 'oracle_queries': 0}
    
    for _ in range(50): 
        sample_size = min(s, math.ceil(D * math.sqrt(s)))
        avail = list(set(range(s)) - T_indices)
        take = min(len(avail), sample_size)
        R_indices = set(np.random.choice(avail, take, replace=False)) if take > 0 else set()
        S_prime = list(R_indices.union(T_indices)) or list(range(s))
        w = np.ones(len(S_prime))
        
        for _ in range(100): 
            W = np.sum(w)
            B_size = min(len(S_prime), 6 * D**2)
            B_indices = np.random.choice(len(S_prime), B_size, p=w/W, replace=True)
            B_servers = [S_prime[i] for i in B_indices]
            
            # --- RECURSION: FindOPT(B) using VWW20 natively ---
            B_A = [servers_A[i] for i in B_servers]; B_b = [servers_b[i] for i in B_servers]
            x_star, inner_metrics = simulate_vww20(B_A, B_b, D, c, is_global=False)
            metrics['constraints_sent'] += inner_metrics['constraints_sent']
            metrics['oracle_queries'] += inner_metrics['oracle_queries']
            
            # Inner loop Separation Oracle queries on active S_prime
            metrics['oracle_queries'] += len(S_prime)
            V_inner, W_V = [], 0
            for idx, s_idx in enumerate(S_prime):
                if check_violation(servers_A[s_idx], servers_b[s_idx], x_star):
                    V_inner.append(idx); W_V += w[idx]
            if len(V_inner) == 0: break
            if W_V <= W / (3 * D):
                for idx in V_inner: w[idx] *= 2
        
        # --- Outer loop Global DSO Check ---
        metrics['pings'] += 1
        metrics['oracle_queries'] += s          
        
        V_global = set()
        for i in range(s):
            if check_violation(servers_A[i], servers_b[i], x_star): V_global.add(i)
        if len(V_global) == 0: return metrics
        if len(V_global) <= 2 * D * math.sqrt(s): T_indices = T_indices.union(V_global)
            
    return metrics

if __name__ == "__main__":
    print("Running Simulator... (takes ~30 seconds)")
    d_fixed = 3
    
    # --- Experiment A: Scale local dataset constraints (n) ---
    s_fixed = 100
    n_list = [10, 50, 100, 200, 500]
    res_A_ours, res_A_base = [], []
    np.random.seed(42)
    for n in n_list:
        A, b, D, c = generate_data(s_fixed, n, d_fixed)
        res_A_ours.append(np.mean([list(simulate_hybrid_clarkson(A, b, D, c).values()) for _ in range(3)], axis=0))
        res_A_base.append(np.mean([list(simulate_vww20(A, b, D, c)[1].values()) for _ in range(3)], axis=0))
        print(f"Exp A (n={n}) evaluated.")

    # --- Experiment B: Scale network size (s) ---
    n_fixed = 20
    s_list = [50, 100, 200, 300, 500]
    res_B_ours, res_B_base = [], []
    for s_val in s_list:
        A, b, D, c = generate_data(s_val, n_fixed, d_fixed)
        res_B_ours.append(np.mean([list(simulate_hybrid_clarkson(A, b, D, c).values()) for _ in range(3)], axis=0))
        res_B_base.append(np.mean([list(simulate_vww20(A, b, D, c)[1].values()) for _ in range(3)], axis=0))
        print(f"Exp B (s={s_val}) evaluated.")

    # --- Plotting the 2x3 Grid ---
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    c_ours, m_ours = '#348ABD', 'o'
    c_base, m_base = '#E24A33', 's'

    # Row 1: Experiment A (Varying n)
    axes[0,0].plot(n_list, [r[0] for r in res_A_ours], marker=m_ours, lw=2, label='HybridClarkson', c=c_ours)
    axes[0,0].plot(n_list, [r[0] for r in res_A_base], marker=m_base, lw=2, label='VWW20', c=c_base)
    axes[0,0].set_xscale('log'); axes[0,0].set_title('Global Sync Rounds vs. $n_i$')
    axes[0,0].set_ylabel('Global Barriers (Pings)'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(n_list, [r[2]/1000 for r in res_A_ours], marker=m_ours, lw=2, c=c_ours)
    axes[0,1].plot(n_list, [r[2]/1000 for r in res_A_base], marker=m_base, lw=2, c=c_base)
    axes[0,1].set_xscale('log'); axes[0,1].set_title('Separation Queries vs. $n_i$')
    axes[0,1].set_ylabel('Local Computations (x$10^3$)'); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(n_list, [r[1]/1000 for r in res_A_ours], marker=m_ours, lw=2, c=c_ours)
    axes[0,2].plot(n_list, [r[1]/1000 for r in res_A_base], marker=m_base, lw=2, c=c_base)
    axes[0,2].set_xscale('log'); axes[0,2].set_title('Bandwidth Trade-Off vs. $n_i$')
    axes[0,2].set_ylabel('Constraints Transmitted (x$10^3$)'); axes[0,2].grid(True, alpha=0.3); axes[0,0].legend()

    # Row 2: Experiment B (Varying s)
    axes[1,0].plot(s_list, [r[0] for r in res_B_ours], marker=m_ours, lw=2, c=c_ours)
    axes[1,0].plot(s_list, [r[0] for r in res_B_base], marker=m_base, lw=2, c=c_base)
    axes[1,0].set_xlabel('Distributed Servers ($s$)'); axes[1,0].set_ylabel('Global Barriers (Pings)')
    axes[1,0].set_title('Global Sync Rounds vs. $s$'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(s_list, [r[2]/1000 for r in res_B_ours], marker=m_ours, lw=2, c=c_ours)
    axes[1,1].plot(s_list, [r[2]/1000 for r in res_B_base], marker=m_base, lw=2, c=c_base)
    axes[1,1].set_xlabel('Distributed Servers ($s$)'); axes[1,1].set_ylabel('Local Computations (x$10^3$)')
    axes[1,1].set_title('Separation Queries vs. $s$'); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(s_list, [r[1]/1000 for r in res_B_ours], marker=m_ours, lw=2, c=c_ours)
    axes[1,2].plot(s_list, [r[1]/1000 for r in res_B_base], marker=m_base, lw=2, c=c_base)
    axes[1,2].set_xlabel('Distributed Servers ($s$)'); axes[1,2].set_ylabel('Constraints Transmitted (x$10^3$)')
    axes[1,2].set_title('Bandwidth Trade-Off vs. $s$'); axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neurips_tradeoffs.pdf', bbox_inches='tight')
    print("\nSuccess! Saved to 'neurips_tradeoffs.pdf'.")
