"""
Proof-of-concept for Algorithm 2: Distributed ClarksonCoordinator
from "On Distributed Separation Oracles and the Communication Cost of Optimization".
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import random

def generate_feasible_lp(d, s, seed=42):
    """
    Step 0: Generate a synthetic, random Linear Program distributed across s servers.
    Each server `i` holds a single linear constraint (a half-space) C_i: A_i * x <= b_i.
    
    Args:
        d (int): Dimension of the problem space.
        s (int): Total number of distributed servers (constraints).
        seed (int): Random seed for reproducibility.
        
    Returns:
        phi (np.ndarray): The linear objective vector to minimize.
        A (np.ndarray): The constraint normals (s x d matrix).
        b (np.ndarray): The constraint offsets (vector of size s).
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate a random objective vector phi
    phi = np.random.uniform(-1, 1, size=d)
    
    # Generate random constraint directions (the hyperplanes)
    A = np.random.uniform(-1, 1, size=(s, d))
    
    # We guarantee x=0 is strictly feasible by setting b > 0.
    # This ensures the intersection of all C_i is not empty.
    b = np.random.uniform(0.1, 1.0, size=s)
    
    return phi, A, b


def find_opt(d, phi, A, b, active_indices):
    """
    Subroutine: FindOPT(T U R, phi)
    This simulates the coordinator solving the subproblem defined ONLY by the 
    constraints that have been sampled (R) or accumulated as previous violations (T).
    
    Args:
        d (int): Dimension.
        phi (np.ndarray): Objective vector.
        A, b: Global constraints (only accessed via active_indices).
        active_indices (list/set): The indices in T U R.
        
    Returns:
        x_star (np.ndarray): The optimal point for this subproblem.
        obj_val (float): The objective value at x_star.
    """
    # Initialize Gurobi environment (suppressing terminal output for clean logs)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    m = gp.Model("FindOPT", env=env)
    m.setParam('Method', 1) # Use Dual Simplex to ensure we find a basic feasible solution
    
    # Theorem 3.5 in the paper mentions a global convex constraint G known to all servers.
    # We enforce a bounding box [-10, 10]^d here as G to ensure intermediate 
    # FindOPT queries don't become unbounded before enough constraints are sampled.
    x = m.addMVar(d, lb=-10.0, ub=10.0, name="x")
    m.setObjective(phi @ x, GRB.MINIMIZE)
    
    # Add ONLY the active constraints (T U R) to the model. 
    # The coordinator never sees the full massive set of s constraints!
    if len(active_indices) > 0:
        indices = list(active_indices)
        m.addConstr(A[indices] @ x <= b[indices])
        
    m.optimize()
    
    if m.status == GRB.OPTIMAL:
        return x.X, m.ObjVal
    return None, None


def check_violations(A, b, x_star, tol=1e-6):
    """
    Simulate the broadcast step: The coordinator broadcasts x_star to all s servers.
    Each server checks locally if x_star violates its constraint.
    
    Args:
        A, b: Constraint data.
        x_star: The candidate optimum from the current round.
        tol: Floating point tolerance.
        
    Returns:
        V (set): The set of server indices where x_star is NOT feasible (x_* \notin C_i).
    """
    # Calculate how much x_star violates each constraint: (A * x_star) - b
    violations = A @ x_star - b
    
    # Find all indices where the violation is strictly positive (greater than tolerance)
    V = set(np.where(violations > tol)[0])
    return V


def centralized_opt(d, phi, A, b):
    """
    Verification Subroutine.
    Solves the full problem centrally using all s constraints at once. 
    This is just to prove that Algorithm 2 actually found the true global optimum.
    """
    x_star, obj_val = find_opt(d, phi, A, b, range(len(b)))
    return x_star, obj_val


def run_distributed_clarkson(d, s):
    """
    Main implementation of Algorithm 2: Distributed ClarksonCoordinator.
    """
    # 1. Setup the distributed problem
    phi, A, b = generate_feasible_lp(d, s)
    
    # 2. Initialize tracking variables
    T = set() # T: Tracks constraints that were violated in "successful" rounds
    
    # Sample size per round based on Lemma 6.2: |R| = ceil(d * sqrt(s))
    sample_size = int(math.ceil(d * math.sqrt(s)))
    
    # Threshold for a "successful" round: |V| <= 2 * sqrt(s)
    max_violations = 2 * math.sqrt(s)
    
    print("=" * 70)
    print(" Distributed Clarkson Coordinator (Algorithm 2)")
    print("=" * 70)
    print(f"Dimension (d)                : {d}")
    print(f"Total Servers (s)            : {s}")
    print(f"Subproblem Sample Size |R|   : {sample_size} constraints")
    print(f"Violation Threshold 2*sqrt(s): {max_violations:.2f}")
    print("-" * 70)
    
    iteration = 0
    successful_iterations = 0
    
    while True:
        iteration += 1
        
        # Algorithm 2, Step 1: Sample R uniformly at random from server indices
        R = set(random.sample(range(s), min(sample_size, s)))
        
        # Algorithm 2, Step 2: Coordinator computes x_* over T U R
        active_indices = T.union(R)
        x_star, obj_val = find_opt(d, phi, A, b, active_indices)
        
        if x_star is None:
            print("Subproblem is unbounded/infeasible.")
            break
            
        # Algorithm 2, Step 3: Servers verify x_star locally and report violations (V)
        V = check_violations(A, b, x_star)
        
        print(f"Iter {iteration:>2}: |T U R|={len(active_indices):>4} constraints used -> Violations |V|={len(V):>4}")
        
        # Algorithm 2, Step 4: If V is empty, x_star is feasible globally! We are done.
        if len(V) == 0:
            print("-" * 70)
            print(f" => Algorithm converged to global optimum in {iteration} total iterations!")
            print(f" => Successful iterations (|V| <= threshold): {successful_iterations}")
            print(f" => Theorem 6.3 Guarantee: expected successful iterations is O(d).")
            print("-" * 70)
            
            # Verify correctness against a standard centralized solve
            _, true_obj = centralized_opt(d, phi, A, b)
            print(f" Distributed Objective : {obj_val:.6f}")
            print(f" Centralized Objective : {true_obj:.6f}")
            print(f" Optima Match          : {abs(obj_val - true_obj) < 1e-5}")
            print("=" * 70)
            break
            
        # Algorithm 2, Step 5: Check if the round was "successful"
        # If the number of violations is small enough, add them to T.
        if len(V) <= max_violations:
            T = T.union(V)
            successful_iterations += 1
            print(f"          -> SUCCESS. Added {len(V)} constraints to T.")
        else:
            # If V is too large, the round is discarded (Markov's inequality bounds the probability of this).
            print(f"          -> FAIL. |V| exceeded threshold. Discarding and resampling...")


if __name__ == "__main__":
    # Test case scaled for the free Gurobi Size-Limited License.
    # The Gurobi free license allows a maximum of 2,000 variables/constraints.
    # 
    # With s = 1500 and d = 10:
    # - The centralized verification step will load exactly 1500 constraints (Under 2000 -> PASS).
    # - The distributed rounds will sample |R| = ceil(10 * sqrt(1500)) = 388 constraints.
    run_distributed_clarkson(d=10, s=1500)
