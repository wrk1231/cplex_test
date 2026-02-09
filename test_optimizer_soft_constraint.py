

import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)
from optimizer_soft_constraint import solve_mvo_soft_constraint

def generate_synthetic_data(n=20, k=3, seed=42):
    np.random.seed(seed)
    alpha = np.random.randn(n) * 0.05
    B = np.random.randn(n, k)
    F = np.diag(np.random.rand(k) * 0.1)
    D = np.random.rand(n) * 0.05
    w_drift = np.random.dirichlet(np.ones(n))
    return alpha, B, F, D, w_drift

def run_tests():
    print("="*60)
    print("Testing solve_mvo_soft_constraint")
    print("="*60)
    
    n, k = 50, 5
    alpha, B, F, D, w_drift = generate_synthetic_data(n, k)
    
    # 1. Baseline: No penalties
    print("\n--- Test 1: Baseline (No penalties) ---")
    res_base = solve_mvo_soft_constraint(alpha, B, F, D)
    if not res_base:
        print("Baseline failed!")
        return
    print(f"Status: {res_base['status']}")
    print(f"L1 Norm: {res_base['L1_norm']:.4f}")
    print(f"L2 Sq:   {res_base['L2_norm_sq']:.4f}")

    # 2. L1 Penalty
    print("\n--- Test 2: L1 Penalty (coeff=0.1) ---")
    res_l1 = solve_mvo_soft_constraint(alpha, B, F, D, penalty_l1_w=0.1)
    print(f"Status: {res_l1['status']}")
    print(f"L1 Norm: {res_l1['L1_norm']:.4f} (Baseline: {res_base['L1_norm']:.4f})")
    if res_l1['L1_norm'] < res_base['L1_norm']:
        print("PASS: L1 norm decreased.")
    else:
        print("FAIL: L1 norm did not decrease.")


    # 3. L2 Penalty on w
    print("\n--- Test 3: L2 Penalty on w (coeff=5.0) ---")
    res_l2 = solve_mvo_soft_constraint(alpha, B, F, D, penalty_l2_w=5.0)
    print(f"Status: {res_l2['status']}")
    print(f"L2 Sq:   {res_l2['L2_norm_sq']:.4f} (Baseline: {res_base['L2_norm_sq']:.4f})")
    if res_l2['L2_norm_sq'] < res_base['L2_norm_sq']:
        print("PASS: L2 norm sq decreased.")
    else:
        print("FAIL: L2 norm sq did not decrease.")

    # 4. Factor Bounds Penalty (Replaces Factor Target)
    print("\n--- Test 4: Factor Bounds (Soft Interval) ---")
    # Base factors
    base_factors = res_base['factor_exposures']
    print(f"Base Factors: {base_factors}")
    
    # Create bounds that are tighter than base factors for Factor 0
    # e.g., if F0 is -0.05, make bounds [-0.01, 0.01]
    bounds = []
    for val in base_factors:
        bounds.append((-0.001, 0.001)) # Very tight around 0
    bounds = np.array(bounds)
    
    print("Applying tight bounds [-0.001, 0.001] via soft penalty (coeff=100.0)")
    res_fac = solve_mvo_soft_constraint(alpha, B, F, D, 
                                        factor_bounds=bounds, 
                                        penalty_l2_factor=100.0)
    
    if res_fac:
        new_factors = res_fac['factor_exposures']
        dist_sq = res_fac['factor_diff_sq'] # Weighted penalty metric
        print(f"New Factors: {new_factors}")
        print(f"Factor Penalty Metric: {dist_sq:.6f}")
        
        # Check if factors moved closer to range compared to baseline
        # Baseline deviation from strict [-0.001, 0.001]
        base_dev_sq = 0.0
        for i, val in enumerate(base_factors):
            over = max(0, val - 0.001)
            under = max(0, -0.001 - val)
            base_dev_sq += (over**2 + under**2)
            
        print(f"Baseline Deviation Sq: {base_dev_sq:.6f}")
        
        if dist_sq < base_dev_sq:
            print("PASS: Factors moved towards bounds.")
        else:
            print("FAIL: Factors did not improve compliance with bounds (or already compliant).")


    # 5. Transaction Cost
    print("\n--- Test 5: Transaction Cost (gamma=0.1) ---")
    res_tc = solve_mvo_soft_constraint(alpha, B, F, D, w_drift=w_drift, gamma=0.1)
    print(f"Status:   {res_tc['status']}")
    print(f"TC Value: {res_tc['transaction_cost']:.6f}")
    if res_tc['transaction_cost'] > 0:
        print("PASS: TC calculated.")
    else:
         # It's possible TC is 0 if w sticks to w_drift, but usually non-zero.
         # Check if weights differ from w_drift
         diff = np.sum(np.abs(res_tc['weights'] - w_drift))
         if diff < 1e-6:
             print("PASS: Weights stuck to drift (TC minimized).")
         else:
             print(f"WARNING: TC is 0 but weights differ? diff={diff}")

if __name__ == "__main__":
    run_tests()
