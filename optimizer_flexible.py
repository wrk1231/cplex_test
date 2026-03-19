import cplex
import numpy as np


def solve_mvo_flexible(alpha, B, F, D, lambda_risk=1.0,
                       penalty_l1_w=0.0, penalty_l2_w=0.0,
                       penalty_l2_factor=0.0,
                       w_drift=None, gamma=None,
                       factor_bounds=None,
                       l1_limit=None, l2_limit=None, turnover_limit=None):
    """
    Solves MVO with flexible soft/hard constraints on all penalty terms.

    Objective (Maximize):
      alpha^T w
      - lambda_risk * (y^T F y + w^T D w)
      - [soft penalties, depending on mode]

    Each penalty parameter accepts:
      - A numeric value (> 0): soft penalty subtracted from the objective
      - The string 'h': enforces the corresponding hard constraint
      - 0 or None: term is disabled

    Parameters
    ----------
    alpha : ndarray (n,)
        Expected returns / alpha signal.
    B : ndarray (n, k)
        Factor loading matrix.
    F : ndarray (k, k) or (k,)
        Factor covariance matrix (full or diagonal).
    D : ndarray (n, n) or (n,)
        Idiosyncratic variance (full diagonal or 1-d).
    lambda_risk : float
        Risk aversion coefficient.

    penalty_l1_w : float or 'h'
        float > 0  -> soft:  subtract rho_1 * ||w||_1 from objective
        'h'        -> hard:  enforce ||w||_1 <= l1_limit
    penalty_l2_w : float or 'h'
        float > 0  -> soft:  subtract rho_2 * ||w||_2^2 from objective
        'h'        -> hard:  enforce ||w||_2^2 <= l2_limit  (QCQP)

    penalty_l2_factor : float or 'h'
        float > 0  -> soft:  quadratic deadzone penalty on factor-bound violations
        0          -> hard:  lb_j <= y_j <= ub_j when factor_bounds is provided
        'h'        -> hard:  lb_j <= y_j <= ub_j
    factor_bounds : ndarray (k, 2) or None
        [lb, ub] per factor.  With penalty_l2_factor=0, behaves like the legacy
        soft-constraint solver and enforces hard bounds.

    w_drift : ndarray (n,) or None
        Reference (current) portfolio for turnover.
    gamma : float, ndarray, or 'h'
        float/arr > 0  -> soft:  subtract gamma * 0.5 * |w - w_drift| from objective
        'h'             -> hard:  enforce sum_i |w_i - w_drift_i| <= turnover_limit
    l1_limit : float or None
        Required when penalty_l1_w='h'.  L1-norm budget.
    l2_limit : float or None
        Required when penalty_l2_w='h'.  Squared-L2 budget.
    turnover_limit : float or None
        Required when gamma='h'.  Total turnover budget.

    Returns
    -------
    dict with keys:
        weights, factor_exposures, objective, status,
        L1_norm, L2_norm_sq, factor_diff_sq, transaction_cost
    """
    try:
        n, k = B.shape

        # ------------------------------------------------------------------
        # Parse modes
        # ------------------------------------------------------------------
        l1_mode = 'off'
        l1_coeff = 0.0
        if penalty_l1_w == 'h':
            l1_mode = 'hard'
            if l1_limit is None:
                raise ValueError("l1_limit required when penalty_l1_w='h'")
        elif isinstance(penalty_l1_w, (int, float)) and penalty_l1_w > 0:
            l1_mode = 'soft'
            l1_coeff = float(penalty_l1_w)

        l2_mode = 'off'
        l2_coeff = 0.0
        if penalty_l2_w == 'h':
            l2_mode = 'hard'
            if l2_limit is None:
                raise ValueError("l2_limit required when penalty_l2_w='h'")
        elif isinstance(penalty_l2_w, (int, float)) and penalty_l2_w > 0:
            l2_mode = 'soft'
            l2_coeff = float(penalty_l2_w)

        factor_mode = 'off'
        factor_coeff = 0.0
        factor_bounds_arr = None
        if factor_bounds is not None:
            factor_bounds_arr = np.asarray(factor_bounds)
            if factor_bounds_arr.shape != (k, 2):
                raise ValueError(
                    f"factor_bounds shape mismatch: {factor_bounds_arr.shape} vs ({k}, 2)")
            if penalty_l2_factor == 'h':
                factor_mode = 'hard'
            elif isinstance(penalty_l2_factor, (int, float)) and penalty_l2_factor > 0:
                factor_mode = 'soft'
                factor_coeff = float(penalty_l2_factor)
            elif penalty_l2_factor == 0:
                factor_mode = 'hard'
        else:
            if penalty_l2_factor == 'h' or (
                    isinstance(penalty_l2_factor, (int, float)) and penalty_l2_factor > 0):
                raise ValueError(
                    "factor_bounds must be provided when penalty_l2_factor is active")

        drift_mode = 'off'
        gamma_arr = np.zeros(n)
        w_drift_arr = np.zeros(n)
        if isinstance(gamma, str):
            if gamma == 'h':
                drift_mode = 'hard'
                if turnover_limit is None:
                    raise ValueError("turnover_limit required when gamma='h'")
                if w_drift is None:
                    raise ValueError("w_drift required when gamma='h'")
                w_drift_arr = np.asarray(w_drift)
                if w_drift_arr.shape != (n,):
                    raise ValueError(
                        f"w_drift shape mismatch: {w_drift_arr.shape} vs ({n},)")
            else:
                raise ValueError("gamma must be numeric, None/0, or 'h'")
        elif gamma is not None:
            gamma_check = np.asarray(gamma)
            if not np.issubdtype(gamma_check.dtype, np.number):
                raise ValueError("gamma must be numeric, None/0, or 'h'")
            if np.any(gamma_check != 0):
                if w_drift is None:
                    raise ValueError(
                        "w_drift must be provided when gamma is non-zero")
                drift_mode = 'soft'
                gamma_arr = (np.full(n, gamma) if np.isscalar(gamma)
                             else np.asarray(gamma))
                w_drift_arr = np.asarray(w_drift)
                if gamma_arr.shape != (n,):
                    raise ValueError(
                        f"gamma shape mismatch: {gamma_arr.shape} vs ({n},)")
                if w_drift_arr.shape != (n,):
                    raise ValueError(
                        f"w_drift shape mismatch: {w_drift_arr.shape} vs ({n},)")

        # ------------------------------------------------------------------
        # CPLEX Init
        # ------------------------------------------------------------------
        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)

        # ------------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------------

        # 1. w  (indices 0 .. n-1)
        prob.variables.add(
            obj=alpha.tolist(),
            lb=[-cplex.infinity] * n,
            ub=[cplex.infinity] * n,
            names=[f"w_{i}" for i in range(n)]
        )

        # 2. y  (indices n .. n+k-1)
        y_lb = [-cplex.infinity] * k
        y_ub = [cplex.infinity] * k
        if factor_mode == 'hard':
            for j in range(k):
                if factor_bounds_arr[j, 0] > -1e20:
                    y_lb[j] = float(factor_bounds_arr[j, 0])
                if factor_bounds_arr[j, 1] < 1e20:
                    y_ub[j] = float(factor_bounds_arr[j, 1])

        prob.variables.add(
            obj=[0.0] * k,
            lb=y_lb,
            ub=y_ub,
            names=[f"y_{j}" for j in range(k)]
        )
        idx_y_start = n
        var_offset = n + k

        # 3. z  (L1 auxiliary for |w_i|, needed when l1_mode != 'off')
        z_start_idx = None
        if l1_mode != 'off':
            z_start_idx = var_offset
            z_obj = ([-l1_coeff] * n) if l1_mode == 'soft' else ([0.0] * n)
            prob.variables.add(
                obj=z_obj,
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"z_{i}" for i in range(n)]
            )
            var_offset += n

        # 4. d  (drift auxiliary for |w_i - w_drift_i|, needed when drift_mode != 'off')
        d_start_idx = None
        if drift_mode != 'off':
            d_start_idx = var_offset
            if drift_mode == 'soft':
                d_obj = (-0.5 * gamma_arr).tolist()
            else:
                d_obj = [0.0] * n
            prob.variables.add(
                obj=d_obj,
                lb=[0.0] * n,
                ub=[cplex.infinity] * n,
                names=[f"d_{i}" for i in range(n)]
            )
            var_offset += n

        # 5. e_pos / e_neg  (soft factor-bound slacks)
        e_pos_idx = None
        e_neg_idx = None
        if factor_mode == 'soft':
            e_pos_idx = var_offset
            prob.variables.add(
                obj=[0.0] * k, lb=[0.0] * k, ub=[cplex.infinity] * k,
                names=[f"e_pos_{j}" for j in range(k)]
            )
            var_offset += k

            e_neg_idx = var_offset
            prob.variables.add(
                obj=[0.0] * k, lb=[0.0] * k, ub=[cplex.infinity] * k,
                names=[f"e_neg_{j}" for j in range(k)]
            )
            var_offset += k

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # 1. Factor definition:  B^T w - y = 0
        for j in range(k):
            indices = list(range(n)) + [idx_y_start + j]
            values = B[:, j].tolist() + [-1.0]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["E"],
                rhs=[0.0]
            )

        # 2a. z >= |w|  (L1 modeling)
        if z_start_idx is not None:
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i, z_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"], rhs=[0.0]
                )
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i, z_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"], rhs=[0.0]
                )

        # 2b. Hard L1:  sum(z_i) <= l1_limit
        if l1_mode == 'hard':
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=list(range(z_start_idx, z_start_idx + n)),
                    val=[1.0] * n
                )],
                senses=["L"],
                rhs=[float(l1_limit)]
            )

        # 3a. d >= |w - w_drift|  (turnover modeling)
        if d_start_idx is not None:
            for i in range(n):
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i, d_start_idx + i], val=[1.0, -1.0])],
                    senses=["L"], rhs=[float(w_drift_arr[i])]
                )
                prob.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i, d_start_idx + i], val=[-1.0, -1.0])],
                    senses=["L"], rhs=[float(-w_drift_arr[i])]
                )

        # 3b. Hard turnover:  sum(d_i) <= turnover_limit
        if drift_mode == 'hard':
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(
                    ind=list(range(d_start_idx, d_start_idx + n)),
                    val=[1.0] * n
                )],
                senses=["L"],
                rhs=[float(turnover_limit)]
            )

        # 4. Soft factor-bound constraints  (e_pos / e_neg)
        if factor_mode == 'soft':
            for j in range(k):
                ub_val = factor_bounds_arr[j, 1]
                if ub_val < 1e20:
                    prob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[idx_y_start + j, e_pos_idx + j],
                            val=[1.0, -1.0])],
                        senses=["L"], rhs=[float(ub_val)]
                    )
                lb_val = factor_bounds_arr[j, 0]
                if lb_val > -1e20:
                    prob.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[idx_y_start + j, e_neg_idx + j],
                            val=[-1.0, -1.0])],
                        senses=["L"], rhs=[float(-lb_val)]
                    )

        # 5. Hard L2:  ||w||_2^2 <= l2_limit  (quadratic constraint -> QCQP)
        if l2_mode == 'hard':
            prob.quadratic_constraints.add(
                quad_expr=cplex.SparseTriple(
                    ind1=list(range(n)),
                    ind2=list(range(n)),
                    val=[1.0] * n
                ),
                sense="L",
                rhs=float(l2_limit),
                name="l2_hard"
            )

        # ------------------------------------------------------------------
        # Quadratic Objective
        # ------------------------------------------------------------------
        quad_coeffs = []

        # 1. -lambda_risk * w^T D w   (and soft L2 penalty if active)
        D_diag = D if D.ndim == 1 else np.diag(D)
        for i in range(n):
            coeff = -2.0 * lambda_risk * D_diag[i]
            if l2_mode == 'soft':
                coeff -= 2.0 * l2_coeff
            if abs(coeff) > 1e-10:
                quad_coeffs.append((i, i, coeff))

        # 2. -lambda_risk * y^T F y
        is_F_diag = (F.ndim == 1 or
                     (F.ndim == 2 and np.allclose(F, np.diag(np.diag(F)))))
        if is_F_diag:
            F_diag = F if F.ndim == 1 else np.diag(F)

        for i in range(k):
            for j in range(k):
                coeff = 0.0
                if is_F_diag:
                    if i == j:
                        coeff = -2.0 * lambda_risk * F_diag[i]
                else:
                    coeff = -2.0 * lambda_risk * F[i, j]
                if abs(coeff) > 1e-10:
                    quad_coeffs.append(
                        (idx_y_start + i, idx_y_start + j, coeff))

        # 3. Soft factor-bound quadratic penalty:  -penalty * (e_pos^2 + e_neg^2)
        if factor_mode == 'soft':
            for j in range(k):
                q_val = -2.0 * factor_coeff
                quad_coeffs.append((e_pos_idx + j, e_pos_idx + j, q_val))
                quad_coeffs.append((e_neg_idx + j, e_neg_idx + j, q_val))

        prob.objective.set_quadratic_coefficients(quad_coeffs)

        # ------------------------------------------------------------------
        # Solver Parameters
        # ------------------------------------------------------------------
        prob.parameters.threads.set(1)
        prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
        prob.parameters.barrier.convergetol.set(1e-6)
        prob.parameters.preprocessing.presolve.set(1)
        prob.parameters.preprocessing.reduce.set(3)

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        prob.solve()

        # ------------------------------------------------------------------
        # Results
        # ------------------------------------------------------------------
        sol_values = prob.solution.get_values()
        weights = np.array(sol_values[:n])
        factor_exposures = np.array(sol_values[n:n + k])

        obj_val = prob.solution.get_objective_value()
        status = prob.solution.get_status_string()

        l1_norm = np.sum(np.abs(weights))
        l2_norm_sq = np.sum(weights ** 2)

        factor_diff_norm_sq = 0.0
        if factor_bounds_arr is not None:
            dist_sq = 0.0
            for j in range(k):
                v_y = factor_exposures[j]
                lb, ub = factor_bounds_arr[j]
                over = max(0.0, v_y - ub)
                under = max(0.0, lb - v_y)
                dist_sq += over ** 2 + under ** 2
            factor_diff_norm_sq = dist_sq

        # Transaction cost (soft penalty value actually subtracted)
        trans_cost = 0.0
        if drift_mode == 'soft':
            drift_dev = np.abs(weights - w_drift_arr)
            trans_cost = np.sum(gamma_arr * 0.5 * drift_dev)

        return {
            "weights": weights,
            "factor_exposures": factor_exposures,
            "objective": obj_val,
            "status": status,
            "L1_norm": l1_norm,
            "L2_norm_sq": l2_norm_sq,
            "factor_diff_sq": factor_diff_norm_sq,
            "transaction_cost": trans_cost,
        }

    except cplex.CplexError as exc:
        print(f"CPLEX Error: {exc}")
        return None
