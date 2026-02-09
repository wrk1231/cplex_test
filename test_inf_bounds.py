
import numpy as np
import cplex
import warnings
import sys
warnings.filterwarnings("ignore")
from optimizer_soft_constraint_regular_to import solve_mvo_soft_constraint

def log(msg):
    print(msg)
    sys.stdout.flush()

def main():
    log("Testing solve_mvo_soft_constraint with mixed infinite bounds like [-0.3, np.inf]")
    log("================================================================================")

    n_assets = 5
    n_factors = 2
    np.random.seed(42)
    alpha = np.random.randn(n_assets) * 0.05
    D = np.random.rand(n_assets) * 0.02
    B = np.random.randn(n_assets, n_factors)
    F = np.diag(np.random.rand(n_factors) * 0.1)

    # Factors: 
    # Factor 0: [-0.01, 0.01] (Fully Bounded)
    # Factor 1: [-0.3, np.inf] (Lower Bounded, Upper Infinite)
    factor_bounds = np.array([
        [-0.01, 0.01],
        [-0.3, np.inf]
    ])

    log(f"Bounds:\n {factor_bounds}")

    # Case 1: Soft Constraints (penalty > 0)
    log("\n[Case 1] Soft Constraints with [-0.3, np.inf]")
    try:
        log("Calling solve_mvo_soft_constraint...")
        res = solve_mvo_soft_constraint(
            alpha=alpha, B=B, F=F, D=D,
            factor_bounds=factor_bounds,
            penalty_l2_factor=100.0
        )
        if res:
             log("SUCCESS: Soft constraints with mixed infinite bounds worked.")
             val = res['factor_exposures'][1]
             log(f"Factor 1 Exposure: {val}")
             if val >= -0.3 - 1e-5:
                 log("PASS: Factor 1 >= -0.3 constraint respected (or close enough).")
             else:
                 log(f"WARNING: Factor 1 < -0.3. Value: {val}")

        else:
             log("Result is None (CPLEX Error caught inside)")
    except Exception as e:
        log(f"FAILED with Exception: {e}")

    # Case 2: Hard Constraints (penalty = 0)
    log("\n[Case 2] Hard Constraints with [-0.3, np.inf]")
    try:
        log("Calling solve_mvo_soft_constraint...")
        res = solve_mvo_soft_constraint(
            alpha=alpha, B=B, F=F, D=D,
            factor_bounds=factor_bounds,
            penalty_l2_factor=0.0
        )
        if res:
             log("SUCCESS: Hard constraints with mixed infinite bounds worked.")
        else:
             log("Result is None (CPLEX Error caught inside)")
    except Exception as e:
        log(f"FAILED with Exception: {e}")

if __name__ == "__main__":
    main()
