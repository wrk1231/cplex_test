import cplex
import numpy as np


def _validate_quadratic_input(matrix, size, name):
    arr = np.asarray(matrix, dtype=float)

    if arr.ndim == 1:
        if arr.shape != (size,):
            raise ValueError(f"{name} shape mismatch: {arr.shape} vs ({size},)")
        return arr

    if arr.ndim != 2 or arr.shape != (size, size):
        raise ValueError(f"{name} shape mismatch: {arr.shape} vs ({size}, {size})")

    if not np.allclose(arr, arr.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")

    return arr


def _append_quadratic_triplets(triplets, matrix, start_idx):
    if matrix.ndim == 1:
        for i, value in enumerate(matrix):
            if abs(value) > 1e-12:
                triplets.append((start_idx + i, start_idx + i, float(value)))
        return

    size = matrix.shape[0]
    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            if abs(value) > 1e-12:
                triplets.append((start_idx + i, start_idx + j, float(value)))


def solve_mvo_target_variance(alpha, B, F, D, target_variance, time_limit=60):
    """
    Solve the target-variance MVO problem:

        maximize    alpha^T w
        subject to  w^T (B F B^T + D) w <= target_variance

    using the factor-auxiliary formulation:

        y = B^T w
        y^T F y + w^T D w <= target_variance

    Parameters
    ----------
    alpha : ndarray (n,)
        Expected returns / alpha signal.
    B : ndarray (n, k)
        Factor loading matrix.
    F : ndarray (k, k) or (k,)
        Factor covariance matrix. A 1-d input is treated as diagonal.
    D : ndarray (n, n) or (n,)
        Idiosyncratic covariance / variance term. A 1-d input is treated as diagonal.
    target_variance : float
        Upper bound on total portfolio variance.
    time_limit : float
        Solver time limit in seconds.

    Returns
    -------
    dict with keys:
        weights, factor_exposures, objective, expected_return, total_variance,
        factor_variance, idiosyncratic_variance, target_variance, variance_slack,
        status
    """
    try:
        alpha_arr = np.asarray(alpha, dtype=float)
        B_arr = np.asarray(B, dtype=float)

        if B_arr.ndim != 2:
            raise ValueError("B must be a 2-d array")

        n, k = B_arr.shape

        if alpha_arr.shape != (n,):
            raise ValueError(f"alpha shape mismatch: {alpha_arr.shape} vs ({n},)")

        F_arr = _validate_quadratic_input(F, k, "F")
        D_arr = _validate_quadratic_input(D, n, "D")

        target_variance_value = float(target_variance)
        if not np.isfinite(target_variance_value):
            raise ValueError("target_variance must be finite")
        if target_variance_value < 0:
            raise ValueError("target_variance must be non-negative")

        prob = cplex.Cplex()
        prob.set_log_stream(None)
        prob.set_results_stream(None)
        prob.set_warning_stream(None)
        prob.objective.set_sense(prob.objective.sense.maximize)

        prob.variables.add(
            obj=alpha_arr.tolist(),
            lb=[-cplex.infinity] * n,
            ub=[cplex.infinity] * n,
            names=[f"w_{i}" for i in range(n)],
        )

        prob.variables.add(
            obj=[0.0] * k,
            lb=[-cplex.infinity] * k,
            ub=[cplex.infinity] * k,
            names=[f"y_{j}" for j in range(k)],
        )

        for j in range(k):
            indices = list(range(n)) + [n + j]
            values = B_arr[:, j].tolist() + [-1.0]
            prob.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=values)],
                senses=["E"],
                rhs=[0.0],
                names=[f"factor_def_{j}"],
            )

        variance_triplets = []
        _append_quadratic_triplets(variance_triplets, D_arr, 0)
        _append_quadratic_triplets(variance_triplets, F_arr, n)

        prob.quadratic_constraints.add(
            quad_expr=cplex.SparseTriple(
                ind1=[item[0] for item in variance_triplets],
                ind2=[item[1] for item in variance_triplets],
                val=[item[2] for item in variance_triplets],
            ),
            sense="L",
            rhs=target_variance_value,
            name="target_variance_cap",
        )

        prob.parameters.threads.set(1)
        prob.parameters.timelimit.set(float(time_limit))
        prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
        prob.parameters.preprocessing.presolve.set(1)
        prob.parameters.preprocessing.reduce.set(3)

        prob.solve()

        status = prob.solution.get_status_string()
        solution_values = np.array(prob.solution.get_values())
        weights = solution_values[:n]
        factor_exposures = solution_values[n:n + k]

        if F_arr.ndim == 1:
            factor_variance = float(np.dot(F_arr, factor_exposures ** 2))
        else:
            factor_variance = float(factor_exposures.T @ F_arr @ factor_exposures)

        if D_arr.ndim == 1:
            idiosyncratic_variance = float(np.dot(D_arr, weights ** 2))
        else:
            idiosyncratic_variance = float(weights.T @ D_arr @ weights)

        total_variance = factor_variance + idiosyncratic_variance
        expected_return = float(alpha_arr @ weights)

        return {
            "weights": weights,
            "factor_exposures": factor_exposures,
            "objective": expected_return,
            "expected_return": expected_return,
            "total_variance": total_variance,
            "factor_variance": factor_variance,
            "idiosyncratic_variance": idiosyncratic_variance,
            "target_variance": target_variance_value,
            "variance_slack": target_variance_value - total_variance,
            "status": status,
        }

    except cplex.CplexError as exc:
        print(f"CPLEX Error: {exc}")
        return None
