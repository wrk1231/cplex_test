
import cplex
import numpy as np


def solve_mvo_soft_constraint(alpha, B, F, D, lambda_risk=1.0, 
                              penalty_l1_w=0.0, penalty_l2_w=0.0, 
                              penalty_l2_factor=0.0,
                              w_drift=None, gamma=None,
                              factor_bounds=None):
    """
    Solves MVO with soft constraints (penalties) instead of hard limits.
    
    Objective (Maximize):
      alpha^T w 
      - lambda_risk * (y^T F y + w^T D w)
      - Transaction cost (based on w_drift and gamma)
      - penalty_l1_w * ||w||_1
      - penalty_l2_w * ||w||_2^2
      
      If factor_bounds is provided (min, max):
          - penalty_l2_factor * ( sum(max(0, y - max)^2) + sum(max(0, min - y)^2) )
          (Deadzone penalty outside bounds)
          If penalty_l2_factor == 0 and factor_bounds used, hard constraints are applied.
    
    Subject to:
      y = B^T w
      
    Transaction cost formula (modified):
      gamma * 0.5 * (|w_i - w_drift_i|)
      This is subtracted from the objective.
      
    w_drift and gamma must be provided if gamma > 0.
    """
    try:
        n, k = B.shape

        # --- Handle Transaction Cost Config ---
        apply_drift_penalty = False
        gamma_arr = np.zeros(n)
        w_drift_arr = np.zeros(n)

        if gamma is not None:
             # Check if gamma has any non-zero elements
            gamma_check = np.asarray(gamma)
            if np.any(gamma_check != 0):
                if w_drift is None:
                    raise ValueError("w_drift must be provided when gamma is non-zero")
                apply_drift_penalty = True
                gamma_arr = np.full(n, gamma) if np.isscalar(gamma) else np.asarray(gamma)
                w_drift_arr = np.asarray(w_drift)
                
                if gamma_arr.shape != (n,):
                     raise ValueError(f"gamma shape mismatch: {gamma_arr.shape} vs ({n},)")
                if w_drift_arr.shape != (n,):
                     raise ValueError(f"w_drift shape mismatch: {w_drift_arr.shape} vs ({n},)")

        # --- Handle Factor Bounds ---
        use_factor_bounds = False
        factor_bounds_arr = None

        if factor_bounds is not None:
            use_factor_bounds = True
            factor_bounds_arr = np.asarray(factor_bounds)
            if factor_bounds_arr.shape != (k, 2):
                raise ValueError(f"factor_bounds shape mismatch: {factor_bounds_arr.shape} vs ({k}, 2)")

        # --- CPLEX Init ---
        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)

        # --- Variables ---
        
        # 1. Weights w (n)
        prob.variables.add(
            obj=alpha.tolist(),
            lb=[-cplex.infinity] * n,
            ub=[cplex.infinity] * n,
            names=[f"w_{i}" for i in range(n)]
        )
        # Var indices for w: 0 to n-1

        # 2. Factor exposures y (k)
        # y = B^T w
        
        y_obj_coeffs = [0.0] * k
        y_lb = [-cplex.infinity] * k
        y_ub = [cplex.infinity] * k

        if use_factor_bounds:
            # Factor bounds logic
            if penalty_l2_factor == 0:
                # Hard constraints
                for j in range(k):
                    if factor_bounds_arr[j, 0] <= -1e20:
                         y_lb[j] = -cplex.infinity
                    else:
                         y_lb[j] = float(factor_bounds_arr[j, 0])
                    
                    if factor_bounds_arr[j, 1] >= 1e20:
                         y_ub[j] = cplex.infinity
                    else:
                         y_ub[j] = float(factor_bounds_arr[j, 1])
            # If soft constraints (penalty > 0), we use aux vars, y bounds are open.

        
        prob.variables.add(
            obj=y_obj_coeffs,
            lb=y_lb,
            ub=y_ub,
            names=[f"y_{j}" for j in range(k)]
        )
        # Var indices for y: n to n+k-1
        
        var_offset = n + k

        # 3. Auxiliary variables for L1 of w (z)
        z_start_idx = None
        # Modified: z is only needed for penalty_l1_w, NOT for transaction cost (which only uses drift now)
        use_z = (penalty_l1_w > 0)
        
        if use_z:
            z_start_idx = var_offset
            z_coeffs = []
            for i in range(n):
                coeff = 0.0
                # Modified: removed gamma contribution to z coefficient
                if penalty_l1_w > 0:
                    coeff -= penalty_l1_w
                z_coeffs.append(coeff)
            
            prob.variables.add(
                obj=z_coeffs,
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"z_{i}" for i in range(n)]
            )
            var_offset += n

        # 4. Auxiliary variables for drift (d)
        d_start_idx = None
        if apply_drift_penalty:
            d_start_idx = var_offset
            d_coeffs = (-0.5 * gamma_arr).tolist()
            prob.variables.add(
                obj=d_coeffs,
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"d_{i}" for i in range(n)]
            )
            var_offset += n

        # 5. Auxiliary variables for soft factor bounds (e_pos, e_neg)
        e_pos_idx = None
        e_neg_idx = None
        use_soft_bounds = use_factor_bounds and (penalty_l2_factor > 0)

        if use_soft_bounds:
            # e_pos: amount y exceeds max
            e_pos_idx = var_offset
            prob.variables.add(
                obj=[0.0]*k,
                lb=[0.0]*k,
                ub=[cplex.infinity]*k,
                names=[f"e_pos_{j}" for j in range(k)]
            )
            var_offset += k

            # e_neg: amount y is below min
            e_neg_idx = var_offset
            prob.variables.add(
                obj=[0.0]*k,
                lb=[0.0]*k,
                ub=[cplex.infinity]*k,
                names=[f"e_neg_{j}" for j in range(k)]
            )
            var_offset += k


        # --- Constraints ---

        # 1. Factor definition: y = B^T w  =>  B^T w - y = 0
        idx_y_start = n
        for j in range(k):
            # sum(B[i,j] * w[i]) - y[j] = 0
            indices = list(range(n)) + [idx_y_start + j]
            values = B[:, j].tolist() + [-1.0]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["E"],
                rhs=[0.0]
            )

        # 2. z constraints: z_i >= |w_i|
        if z_start_idx is not None:
             for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, z_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"],
                    rhs=[0.0]
                )
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, z_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"],
                    rhs=[0.0]
                )

        # 3. d constraints: d_i >= |w_i - w_drift_i|
        if d_start_idx is not None:
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, d_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"],
                    rhs=[float(w_drift_arr[i])]
                )
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, d_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"],
                    rhs=[float(-w_drift_arr[i])]
                )

        # 4. Soft Bound constraints
        if use_soft_bounds:
            for j in range(k):
                # e_pos_j >= y_j - ub_j  =>  y_j - e_pos_j <= ub_j
                ub_val = factor_bounds_arr[j, 1]
                if ub_val < 1e20: # Skip if infinite
                    prob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[idx_y_start + j, e_pos_idx + j], val=[1.0, -1.0])],
                        senses=["L"],
                        rhs=[float(ub_val)]
                    )
                
                # e_neg_j >= lb_j - y_j  =>  -y_j - e_neg_j <= -lb_j
                lb_val = factor_bounds_arr[j, 0]
                if lb_val > -1e20: # Skip if infinite
                    prob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(ind=[idx_y_start + j, e_neg_idx + j], val=[-1.0, -1.0])],
                        senses=["L"],
                        rhs=[float(-lb_val)]
                    )

        # --- Quadratic Objective ---
        
        quad_coeffs = []

        # 1. w^T D w  AND  penalty_l2_w * w^T w
        D_diag = D if D.ndim == 1 else np.diag(D)
        for i in range(n):
            coeff = -2.0 * lambda_risk * D_diag[i]
            if penalty_l2_w > 0:
                coeff -= 2.0 * penalty_l2_w
            if abs(coeff) > 1e-10:
                quad_coeffs.append((i, i, coeff))

        # 2. y^T F y
        # Base F logic
        F_diag = None
        is_F_diag = (F.ndim == 1 or (F.ndim == 2 and np.allclose(F, np.diag(np.diag(F)))))
        if is_F_diag:
             F_diag = F if F.ndim == 1 else np.diag(F)

        # We construct effective Q for y block
        for i in range(k):
            for j in range(k):
                coeff = 0.0
                # From risk model
                if is_F_diag:
                    if i == j:
                        coeff -= 2.0 * lambda_risk * F_diag[i]
                else:
                    coeff -= 2.0 * lambda_risk * F[i, j]
                
                if abs(coeff) > 1e-10:
                    quad_coeffs.append((idx_y_start + i, idx_y_start + j, coeff))

        # 3. Soft Bound Quadratic Penalty
        # Minimize penalty * (sum e_pos^2 + sum e_neg^2)
        # Maximize - penalty * ...
        if use_soft_bounds:
            for j in range(k):
                q_val = -2.0 * penalty_l2_factor
                # e_pos
                quad_coeffs.append((e_pos_idx + j, e_pos_idx + j, q_val))
                # e_neg
                quad_coeffs.append((e_neg_idx + j, e_neg_idx + j, q_val))

        prob.objective.set_quadratic_coefficients(quad_coeffs)

        # --- Parameters ---
        prob.parameters.threads.set(1)
        prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
        prob.parameters.barrier.convergetol.set(1e-6)
        prob.parameters.preprocessing.presolve.set(1)
        prob.parameters.preprocessing.reduce.set(3)

        # --- Solve ---
        prob.solve()

        # --- Result ---
        sol_values = prob.solution.get_values()
        weights = np.array(sol_values[:n])
        factor_exposures = np.array(sol_values[n:n+k])
        
        # Derived metrics
        obj_val = prob.solution.get_objective_value()
        status = prob.solution.get_status_string()
        
        l1_norm = np.sum(np.abs(weights))
        l2_norm_sq = np.sum(weights**2)
        
        factor_diff_norm_sq = 0.0
        if use_factor_bounds:
            # Calculate distance to bounds
            # max(0, y - max)^2 + max(0, min - y)^2
            dist_sq = 0.0
            for j in range(k):
                v_y = factor_exposures[j]
                lb, ub = factor_bounds_arr[j]
                over = max(0.0, v_y - ub)
                under = max(0.0, lb - v_y)
                dist_sq += (over**2 + under**2)
            factor_diff_norm_sq = dist_sq
            
        trans_cost = 0.0
        if apply_drift_penalty:
            drift_dev = np.abs(weights - w_drift_arr)
            # Modified: Transaction cost formula is just gamma * 0.5 * drift_dev
            tc_vec = gamma_arr * 0.5 * drift_dev
            trans_cost = np.sum(tc_vec)

        return {
            "weights": weights,
            "factor_exposures": factor_exposures,
            "objective": obj_val,
            "status": status,
            "L1_norm": l1_norm,
            "L2_norm_sq": l2_norm_sq,
            "factor_diff_sq": factor_diff_norm_sq,
            "transaction_cost": trans_cost
        }


    except cplex.CplexError as exc:
        print(f"CPLEX Error: {exc}")
        return None
