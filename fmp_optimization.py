
import numpy as np
import time
import sys

try:
    import cplex
except ImportError:
    print("Error: cplex module not found. Please ensure you are running in the 'py_cplex' environment.")
    sys.exit(1)

"""
Factor Mimicking Portfolio (FMP) Optimization
---------------------------------------------

Problem Formulation:
We aim to construct a portfolio 'w' (N x 1) that mimics a specific target factor exposure 't' (K x 1),
while minimizing the Total Portfolio Risk.

Model:
    Covariance Matrix Sigma = B * F * B^T + D
    where:
        B: (N x K) Factor Loadings
        F: (K x K) Factor Covariance Matrix
        D: (N x N) Idiosyncratic Variance Matrix (Diagonal)

Objective:
    Minimize Total Variance:
    min (1/2) * w^T * Sigma * w
    = min (1/2) * w^T * (B * F * B^T + D) * w
    = min (1/2) * (w^T B) * F * (B^T w) + (1/2) * w^T D w

Constraints:
    1. Factor Exposure Constraint:
       B^T * w = t

Optimization Logic:
    Substituting the constraint (B^T w = t) into the objective function:
    Objective = (1/2) * t^T * F * t  +  (1/2) * w^T * D * w

    Since t and F are constants, the term (1/2) * t^T * F * t is a constant.
    Therefore, minimizing Total Variance subject to fixed factor exposures is MATHEMATICALLY EQUIVALENT
    to minimizing the Idiosyncratic Variance:
    
    Reduced Problem:
        Minimize: (1/2) * w^T * D * w
        Subject to: B^T * w = t
    
    This reduction allows us to solve a separable QP with diagonal Hessian, which is 
    extremely fast and sparse, avoiding the need to handle the dense Sigma matrix directly.
"""

def solve_fmp_cplex(B, D, F, target_exposure, time_limit=60):
    """
    Solves the FMP problem using CPLEX Python API.
    
    Args:
        B (np.ndarray): Factor loadings (N, K).
        D (np.ndarray): Idiosyncratic variances (N,).
        F (np.ndarray): Factor Covariance Matrix (K, K).
        target_exposure (np.ndarray): Target exposures (K,).
        time_limit (float): Time limit in seconds.
        
    Returns:
        dict: Results including weights, objective (Total Variance), and timings.
    """
    t0 = time.time()
    n, k = B.shape
    
    # Validation
    if len(D) != n: return {"error": "D dimension mismatch"}
    if len(target_exposure) != k: return {"error": "Target dimension mismatch"}
    if F.shape != (k, k): return {"error": "F dimension mismatch"}

    try:
        # Initialize CPLEX
        prob = cplex.Cplex()
        
        # Performance Tuning
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        
        # Barrier method is best for QPs of this structure
        prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
        
        # Threads: 0 means auto (use all cores).
        prob.parameters.threads.set(0)
        
        # Time limit
        prob.parameters.timelimit.set(time_limit)

        # Objective: Minimize
        prob.objective.set_sense(prob.objective.sense.minimize)

        # 1. Variables
        # w_i: continuous, unbounded
        prob.variables.add(
            obj=[0.0] * n,
            lb=[-cplex.infinity] * n,
            ub=[cplex.infinity] * n,
            names=[f"w_{i}" for i in range(n)]
        )

        # 2. Quadratic Objective: 0.5 * w^T * D * w
        # Since we optimized to minimize idiosynchratic risk (see docstring), we input D.
        # CPLEX API expects coeffs for 0.5 * xQx, so we pass (i, i, D[i]).
        q_triplets = [(i, i, float(D[i])) for i in range(n)]
        prob.objective.set_quadratic_coefficients(q_triplets)

        # 3. Constraints: B^T * w = t
        # Iterating over columns of B (Factors)
        lin_exprs = []
        rhs = []
        senses = []
        names = []
        
        for j in range(k):
            # Column j of B
            col_vals = B[:, j].tolist()
            lin_exprs.append(cplex.SparsePair(ind=list(range(n)), val=col_vals))
            rhs.append(float(target_exposure[j]))
            senses.append("E") # Equality
            names.append(f"fac_{j}")
            
        prob.linear_constraints.add(
            lin_expr=lin_exprs,
            senses=senses,
            rhs=rhs,
            names=names
        )

        setup_time = time.time() - t0

        # Solve
        solve_start = time.time()
        prob.solve()
        solve_end = time.time()
        
        # Extract results
        status = prob.solution.get_status_string()
        if prob.solution.get_status() in [1, 101, 102]: # Optimal statuses
            w_opt = np.array(prob.solution.get_values())
            
            # The solver minimized 0.5 * w'Dw
            solver_obj = prob.solution.get_objective_value()
            
            # Calculate Constant Term: 0.5 * t^T * F * t
            # This represents the systematic risk component
            systematic_risk = 0.5 * (target_exposure.T @ F @ target_exposure)
            
            # Total Objective = Solver Obj (Specific Risk) + Systematic Risk
            total_objective = solver_obj + systematic_risk
            
        else:
            w_opt = None
            total_objective = None
            solver_obj = None
            
        return {
            "weights": w_opt,
            "objective": total_objective, # Total Variance (0.5 wSigmaw)
            "specific_risk_component": solver_obj,
            "status": status,
            "solve_time": solve_end - solve_start,
            "total_time": time.time() - t0
        }

    except cplex.CplexError as exc:
        return {"error": str(exc), "status": "Error"}


def generate_data(n, k):
    np.random.seed(42)  # Fixed for reproducibility
    
    # Factor Loadings B
    B = np.random.randn(n, k)
    
    # Idiosyncratic Variance D (diagonal)
    D = np.random.uniform(0.1, 0.5, n)
    
    # Factor Covariance F
    # Create a random positive definite matrix
    temp = np.random.randn(k, k)
    F = temp @ temp.T + np.eye(k) * 0.1
    
    # Target exposure: Unit exposure to Factor 0
    t = np.zeros(k)
    t[0] = 1.0
    
    return B, D, F, t

def run_benchmark():
    test_sizes = [
        (50, 5),
        (5000, 10),
        (10000, 20),
        (20000, 25)
    ]
    
    print(f"{'Assets':<8} | {'Factors':<8} | {'Time (s)':<10} | {'Total Obj':<12} | {'ExpErr':<10}")
    print("-" * 60)
    
    for n, k in test_sizes:
        B, D, F, t = generate_data(n, k)
        
        res = solve_fmp_cplex(B, D, F, t)
        
        if 'error' in res:
             print(f"{n:<8} | {k:<8} | {'ERROR':<10} | {res['error']}")
        elif res['weights'] is None:
             print(f"{n:<8} | {k:<8} | {'FAIL':<10} | {res['status']}")
        else:
            w = res['weights']
            
            # Verify constraints
            realized_exp = B.T @ w
            exp_err = np.max(np.abs(realized_exp - t))
            
            print(f"{n:<8} | {k:<8} | {res['total_time']:<10.4f} | {res['objective']:<12.6f} | {exp_err:<10.2e}")
            
    print("-" * 60)

if __name__ == "__main__":
    run_benchmark()
