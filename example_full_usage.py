
import numpy as np
import warnings
# Suppress CPLEX implementation warnings if any
warnings.filterwarnings("ignore")

# Import the specific optimizer function
from optimizer_soft_constraint_regular_to import solve_mvo_soft_constraint

def main():
    print("Optimization with All Parameters Example")
    print("========================================")

    # -------------------------------------------------------------
    # 1. Setup Synthetic Data
    # -------------------------------------------------------------
    n_assets = 10
    n_factors = 3
    np.random.seed(42)

    # Assets
    alpha = np.random.randn(n_assets) * 0.05       # Expected returns
    D = np.random.rand(n_assets) * 0.02            # Idiosyncratic risk

    # Factor Model
    B = np.random.randn(n_assets, n_factors)       # Factor loadings
    F = np.diag(np.random.rand(n_factors) * 0.1)   # Factor covariance

    # Previous State (for transaction costs)
    w_drift = np.random.uniform(-0.1, 0.1, n_assets)
    w_drift = w_drift / np.sum(np.abs(w_drift))    # Normalize for realism

    # -------------------------------------------------------------
    # 2. Define Constraints & Penalties
    # -------------------------------------------------------------
    
    # Factor Bounds (Soft Constraints)
    # We want Factor 0 in [-0.01, 0.01], others wider
    factor_bounds = np.array([
        [-0.01, 0.01],  # Factor 0: Tight range
        [-0.50, 0.50],  # Factor 1: Loose
        [-0.50, 0.50],  # Factor 2: Loose
    ])

    # Parameters
    params = {
        # Basic Inputs
        "alpha": alpha,
        "B": B,
        "F": F,
        "D": D,
        
        # Risk Aversion (Obj = Alpha - lambda * Risk - Costs - Penalties)
        "lambda_risk": 2.0,
        
        # Weight Penalties (Regularization)
        "penalty_l1_w": 0.005,      # Sparse weights
        "penalty_l2_w": 1.0,        # Small weights (Ridge)
        
        # Factor Constraint Configuration
        "factor_bounds": factor_bounds,
        "penalty_l2_factor": 100.0, # High penalty for violating factor_bounds
        
        # Transaction Costs (New formula: gamma * 0.5 * |w - w_drift|)
        "w_drift": w_drift,
        "gamma": 0.02               # 20 bps transaction cost parameter
    }

    # -------------------------------------------------------------
    # 3. Run Optimization
    # -------------------------------------------------------------
    print("\nRunning solver with parameters:")
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: [Array shape {v.shape}]")
        else:
            print(f"  {k}: {v}")

    result = solve_mvo_soft_constraint(**params)

    # -------------------------------------------------------------
    # 4. Analyze Results
    # -------------------------------------------------------------
    if result:
        print("\nOptimization Successful!")
        print(f"  Status:           {result['status']}")
        print(f"  Objective Value:  {result['objective']:.6f}")
        print(f"  Transaction Cost: {result['transaction_cost']:.6f}")
        print(f"  Factor Penalty:   {result['factor_diff_sq']:.6f}")
        print("-" * 30)
        
        print("\nWeights vs Drift:")
        print(f"{'Asset':<5} {'w_drift':<10} {'w_opt':<10} {'Diff':<10}")
        for i in range(n_assets):
            print(f"{i:<5} {w_drift[i]:.4f}     {result['weights'][i]:.4f}     {result['weights'][i]-w_drift[i]:.4f}")

        print("\nFactor Exposures vs Bounds:")
        y = result['factor_exposures']
        print(f"{'Factor':<6} {'Min':<8} {'Max':<8} {'Actual':<8} {'Violation'}")
        for j in range(n_factors):
            lb, ub = factor_bounds[j]
            val = y[j]
            # visual check for violation
            violation = ""
            if val < lb: violation = f"Low by {lb-val:.4f}"
            if val > ub: violation = f"High by {val-ub:.4f}"
            print(f"{j:<6} {lb:<8.2f} {ub:<8.2f} {val:<8.4f} {violation}")

    else:
        print("\nOptimization Failed.")

if __name__ == "__main__":
    main()
