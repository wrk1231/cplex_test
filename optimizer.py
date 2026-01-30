import cplex
import numpy as np

def solve_mvo_factor_auxiliary(alpha, B, F, D, lambda_risk=1, L1_limit=None, factor_exposure_limits=None, w_drift=None, gamma=None):
    """
    Transaction cost formula: gamma * 1/2 * (|w_i - w_drift_i| + |w_i| - |w_drift_i|)
    """
    try:
        n, k = B.shape

        # Check if drift penalty should be applied (gamma > 0 and w_drift provided)
        apply_drift_penalty = False
        gamma_arr = None
        w_drift_arr = None

        if gamma is not None and not (np.isscalar(gamma) and gamma == 0):
            # gamma is non-zero, check if it's a scalar 0 or array of zeros
            gamma_test = np.asarray(gamma)
            if np.any(gamma_test != 0):
                # We have non-zero gamma values
                if w_drift is None:
                    raise ValueError("w_drift must be provided when gamma is non-zero")
                apply_drift_penalty = True
                gamma_arr = np.full(n, gamma) if np.isscalar(gamma) else np.asarray(gamma)
                if gamma_arr.shape != (n,):
                    raise ValueError(f"gamma must be scalar or array of shape ({n},), got {gamma_arr.shape}")
                w_drift_arr = np.asarray(w_drift)
                if w_drift_arr.shape != (n,):
                    raise ValueError(f"w_drift must be array of shape ({n},), got {w_drift_arr.shape}")

        # Initialize CPLEX problem
        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)

        # Variables: w (n) and y (k)
        # Asset weights
        prob.variables.add(
            obj=alpha.tolist(),
            lb=[-cplex.infinity] * n,
            ub=[cplex.infinity] * n,
            names=[f"w_{i}" for i in range(n)]
        )

        # y: factor exposures (y = B'w)
        # Apply factor exposure bounds if provided
        y_lb = [-cplex.infinity] * k
        y_ub = [cplex.infinity] * k
        if factor_exposure_limits is not None:
            if 'bounds' in factor_exposure_limits:
                # User provided list of (lower, upper) tuples
                bounds = factor_exposure_limits['bounds']
                y_lb = [b[0] if b[0] is not None else -cplex.infinity for b in bounds]
                y_ub = [b[1] if b[1] is not None else cplex.infinity for b in bounds]
            else:
                # User provided 'lower' and 'upper' arrays
                y_lb = factor_exposure_limits.get('lower', [-cplex.infinity] * k)
                y_ub = factor_exposure_limits.get('upper', [cplex.infinity] * k)
            # Convert None values to infinities
            y_lb = [lb if lb is not None else -cplex.infinity for lb in y_lb]
            y_ub = [ub if ub is not None else cplex.infinity for ub in y_ub]
        else:
            y_lb = [-cplex.infinity] * k
            y_ub = [cplex.infinity] * k

        prob.variables.add(
            obj=[0.0] * k,
            lb=y_lb,
            ub=y_ub,
            names=[f"y_{j}" for j in range(k)]
        )
        var_offset = n + k

        # If L1 constraint is active OR drift penalty is active, add auxiliary variables for |w_i|
        z_start_idx = None
        if L1_limit is not None or apply_drift_penalty:
            z_start_idx = var_offset
            # Add z_i >= |w_i| for each i, where z_i >= 0
            # If drift penalty is active, add objective coefficient: -0.5 * gamma_i
            if apply_drift_penalty:
                obj_coeffs = (-0.5 * gamma_arr).tolist()
            else:
                obj_coeffs = [0.0] * n
            prob.variables.add(
                obj=obj_coeffs,
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"z_{i}" for i in range(n)]
            )
            var_offset += n

        # If drift penalty is active, add auxiliary variables for |w_i - w_drift_i|
        d_start_idx = None
        if apply_drift_penalty:
            d_start_idx = var_offset
            # Add d_i >= |w_i - w_drift_i| for each i, where d_i >= 0
            # Objective coefficient: -0.5 * gamma_i
            prob.variables.add(
                obj=(-0.5 * gamma_arr).tolist(),
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"d_{i}" for i in range(n)]
            )
            var_offset += n

        # Constraints: y = B'w => Bw - y = 0
        for j in range(k):
            indices = list(range(n)) + [n + j]
            values = B[:, j].tolist() + [-1.0]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["E"],
                rhs=[0.0]
            )

        # Constraints for |w_i| auxiliary variables (z_i >= |w_i|)
        if z_start_idx is not None:
            # Constraint: z_i >= w_i for all i
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, z_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"],
                    rhs=[0.0]
                )
            # Constraint: z_i >= -w_i for all i
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, z_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"],
                    rhs=[0.0]
                )

        # L1 constraint: sum(|w_i|) <= L1_limit
        if L1_limit is not None:
            # Constraint: sum(z_i) <= L1_limit
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=list(range(z_start_idx, z_start_idx + n)),
                    val=[1.0] * n
                )],
                senses=["L"],
                rhs=[L1_limit]
            )

        # Drift penalty constraints: d_i >= |w_i - w_drift_i|
        if apply_drift_penalty:
            # Constraint: d_i >= w_i - w_drift_i for all i
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, d_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"],
                    rhs=[float(w_drift_arr[i])]
                )
            # Constraint: d_i >= -(w_i - w_drift_i) for all i
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(ind=[i, d_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"],
                    rhs=[float(-w_drift_arr[i])]
                )

        # Quadratic objective: -lambda_risk * (y'Fy + w'Dw)
        quad_coeffs = []

        # Factor risk: y'Fy
        if F.ndim == 1 or (F.ndim == 2 and np.allclose(F, np.diag(np.diag(F)))):
            # F is diagonal
            F_diag = F if F.ndim == 1 else np.diag(F)
            for j in range(k):
                if abs(F_diag[j]) > 1e-10:
                    quad_coeffs.append((n + j, n + j, -2.0 * lambda_risk * F_diag[j]))
        else:
            # F is full matrix
            for i in range(k):
                for j in range(k):
                    if abs(F[i, j]) > 1e-10:
                        quad_coeffs.append((n + i, n + j, -2.0 * lambda_risk * F[i, j]))

        # Specific risk: w'Dw
        D_diag = D if D.ndim == 1 else np.diag(D)
        for i in range(n):
            if abs(D_diag[i]) > 1e-10:
                quad_coeffs.append((i, i, -2.0 * lambda_risk * D_diag[i]))

        prob.objective.set_quadratic_coefficients(quad_coeffs)

        # Performance tuning
        prob.parameters.threads.set(0)
        prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
        prob.parameters.barrier.convergetol.set(1e-6)
        prob.parameters.preprocessing.presolve.set(1)
        prob.parameters.preprocessing.reduce.set(3)

        prob.solve()

        # Extract only w (first n variables)
        weights = np.array(prob.solution.get_values()[:n])
        factor_exposures = np.array(prob.solution.get_values()[n:n + k])

        # Calculate drift deviation and transaction cost if applicable
        drift_deviation = None
        transaction_cost = None
        if apply_drift_penalty:
            drift_deviation = np.abs(weights - w_drift_arr)
            transaction_cost = gamma_arr * 0.5 * (drift_deviation + np.abs(weights) - np.abs(w_drift_arr))
            transaction_cost = np.sum(transaction_cost)

        return {
            "weights": weights,
            "objective": prob.solution.get_objective_value(),
            "status": prob.solution.get_status_string(),
            "L1_norm": np.sum(np.abs(weights)),
            "factor_exposures": factor_exposures,
            "drift_deviation": drift_deviation,
            "transaction_cost": transaction_cost
        }

    except cplex.CplexError as exc:
        print(f"CPLEX Error: {exc}")
        return None