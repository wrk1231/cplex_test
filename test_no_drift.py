
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from optimizer_soft_constraint_regular_to import solve_mvo_soft_constraint

def main():
    print("Testing solve_mvo_soft_constraint with w_drift=None")
    print("====================================================")

    # 1. Setup Data
    n_assets = 5
    n_factors = 2
    np.random.seed(42)
    alpha = np.random.randn(n_assets) * 0.05
    D = np.random.rand(n_assets) * 0.02
    B = np.random.randn(n_assets, n_factors)
    F = np.diag(np.random.rand(n_factors) * 0.1)

    # -------------------------------------------------------------
    # CASE A: gamma=None, w_drift=None
    # -------------------------------------------------------------
    print("\n[Case A] gamma=None, w_drift=None")
    try:
        res_a = solve_mvo_soft_constraint(
            alpha=alpha, B=B, F=F, D=D,
            gamma=None, 
            w_drift=None
        )
        if res_a:
            print(f"  Status: {res_a['status']}")
            print(f"  TC:     {res_a['transaction_cost']:.6f}")
            print("  SUCCESS: Optimization ran without w_drift.")
        else:
            print("  FAILED: Optimizer returned None.")
    except Exception as e:
        print(f"  ERROR: {e}")

    # -------------------------------------------------------------
    # CASE B: gamma=0.0, w_drift=None
    # -------------------------------------------------------------
    print("\n[Case B] gamma=0.0, w_drift=None")
    try:
        res_b = solve_mvo_soft_constraint(
            alpha=alpha, B=B, F=F, D=D,
            gamma=0.0, 
            w_drift=None
        )
        if res_b:
            print(f"  Status: {res_b['status']}")
            print(f"  TC:     {res_b['transaction_cost']:.6f}")
            print("  SUCCESS: Optimization ran with gamma=0.0 and no w_drift.")
        else:
            print("  FAILED: Optimizer returned None.")
    except Exception as e:
        print(f"  ERROR: {e}")

    # -------------------------------------------------------------
    # CASE C: gamma=0.1, w_drift=None (Expected Failure)
    # -------------------------------------------------------------
    print("\n[Case C] gamma=0.1, w_drift=None (Expect Error)")
    try:
        res_c = solve_mvo_soft_constraint(
            alpha=alpha, B=B, F=F, D=D,
            gamma=0.1,
            w_drift=None
        )
        print("  UNEXPECTED SUCCESS: Should have failed.")
    except ValueError as e:
        print(f"  CAUGHT EXPECTED ERROR: {e}")
    except Exception as e:
        print(f"  CAUGHT UNEXPECTED ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
