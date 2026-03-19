"""
Test & demo script for optimizer_flexible.py — solve_mvo_flexible().

Covers every penalty term in both soft and hard mode, plus combined scenarios.
Run:  python test_optimizer_flexible.py

Each section is self-contained so you can read it as a usage example.
Assertions verify solver feasibility, constraint satisfaction, and expected
penalty behaviour.

Input data is constructed inline so you can see the exact shape, dtype, and
content that the solver expects.
"""
import numpy as np
import sys
sys.stdout.reconfigure(line_buffering=True)

from optimizer_flexible import solve_mvo_flexible


def header(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def calc_turnover(result, w_ref):
    if w_ref is None:
        return 0.0
    return float(np.sum(np.abs(result["weights"] - w_ref)))


# ════════════════════════════════════════════════════════════════════
# Input data construction
# ════════════════════════════════════════════════════════════════════
#
# All tests share this data so that results are comparable across
# sections.  Each array's shape and meaning is documented below.
#
# Dimensions:
#   n = 20   number of assets
#   k = 3    number of risk factors

np.random.seed(42)

n = 20   # assets
k = 3    # factors

# alpha: (n,) float64 — expected return (alpha signal) per asset.
#   Typical magnitude: a few percent annualised.
alpha = np.random.randn(n) * 0.05

# B: (n, k) float64 — factor loading matrix.
#   B[i, j] = exposure of asset i to factor j.
B = np.random.randn(n, k)

# F: (k, k) float64 — factor covariance matrix  (symmetric PSD).
#   Diagonal here for simplicity; full dense matrix also works.
F = np.diag(np.random.rand(k) * 0.1)

# D: (n,) float64 — idiosyncratic (asset-specific) variance.
#   Can also be (n, n) diagonal matrix; 1-d vector is preferred.
D = np.random.rand(n) * 0.05

# w_drift: (n,) float64 — reference / current portfolio weights.
#   Used as the "drift" point for turnover / transaction cost.
#   Sums to 1.0 here (Dirichlet draw), but solver does not require it.
w_drift = np.random.dirichlet(np.ones(n))

# factor_bounds: (k, 2) float64 — per-factor [lower_bound, upper_bound].
#   Used by the factor-bound constraint (soft or hard).
#   Use -np.inf / np.inf for one-sided bounds.
factor_bounds_default = np.array([
    [-0.2,  0.2],   # factor 0
    [-0.3,  0.3],   # factor 1
    [-0.1,  0.1],   # factor 2
])

# Tight factor bounds variant (for penalty-sweep tests)
factor_bounds_tight = np.array([
    [-0.01, 0.01],
    [-0.01, 0.01],
    [-0.01, 0.01],
])

# One-sided / infinite factor bounds variant
factor_bounds_onesided = np.array([
    [-np.inf,  0.1],      # factor 0: upper-only
    [-0.2,     np.inf],   # factor 1: lower-only
    [-np.inf,  np.inf],   # factor 2: fully free
])

LEGACY_RESULT_KEYS = {
    "weights",
    "factor_exposures",
    "objective",
    "status",
    "L1_norm",
    "L2_norm_sq",
    "factor_diff_sq",
    "transaction_cost",
}


# ════════════════════════════════════════════════════════════════════
# 1. Baseline — no penalties at all
# ════════════════════════════════════════════════════════════════════

def test_baseline():
    """
    Minimal call: only alpha-risk trade-off, no regularization.
    Every optional parameter is passed explicitly at its default value.
    """
    header("1. Baseline (no penalties)")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None, "Solver returned None"
    assert "optimal" in result["status"], f"Non-optimal: {result['status']}"
    print(f"  Status     : {result['status']}")
    print(f"  Objective  : {result['objective']:.6f}")
    print(f"  L1 norm    : {result['L1_norm']:.4f}")
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}")
    print("  PASS")
    return result


def test_legacy_result_schema_baseline():
    """
    solve_mvo_flexible should return the same result-key schema used by
    optimizer_soft_constraint_regular_to.py.
    """
    header("1b. Legacy result schema compatibility")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None, "Solver returned None"
    assert "optimal" in result["status"], f"Non-optimal: {result['status']}"
    assert set(result.keys()) == LEGACY_RESULT_KEYS, \
        f"Unexpected legacy keys: {sorted(result.keys())}"
    print(f"  Status     : {result['status']}")
    print(f"  Keys       : {sorted(result.keys())}")
    print("  PASS: solve_mvo_flexible matches old return schema")
    return result


# ════════════════════════════════════════════════════════════════════
# 2. L1 penalty — soft vs hard
# ════════════════════════════════════════════════════════════════════

def test_l1_soft(baseline):
    """
    Soft L1 penalty encourages sparser weights by subtracting
    rho_1 * ||w||_1 from the objective.
    """
    header("2a. L1 Penalty — SOFT (penalty_l1_w=0.1)")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.1,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    print(f"  L1 norm    : {result['L1_norm']:.4f}  (baseline: {baseline['L1_norm']:.4f})")
    assert result["L1_norm"] < baseline["L1_norm"], "L1 norm should decrease"
    print("  PASS: L1 norm decreased with soft penalty")
    return result


def test_l1_hard():
    """
    Hard L1 constraint: ||w||_1 <= l1_limit.  The solver must not
    exceed the budget.
    """
    header("2b. L1 Penalty — HARD (l1_limit=2.0)")
    LIMIT = 2.0

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w='h',
        l1_limit=LIMIT,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    print(f"  L1 norm    : {result['L1_norm']:.4f}  (limit: {LIMIT})")
    assert result["L1_norm"] <= LIMIT + 1e-5, \
        f"L1 norm {result['L1_norm']} exceeds limit {LIMIT}"
    print("  PASS: L1 norm within hard limit")
    return result


def test_l1_hard_missing_limit():
    """
    Passing penalty_l1_w='h' without l1_limit must raise ValueError.
    """
    header("2c. L1 Hard — missing l1_limit (expect ValueError)")
    try:
        solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w='h',
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=None,
            gamma=0.0,
            turnover_limit=None,
            factor_bounds=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASS")


# ════════════════════════════════════════════════════════════════════
# 3. L2 penalty — soft vs hard
# ════════════════════════════════════════════════════════════════════

def test_l2_soft(baseline):
    """
    Soft L2 penalty shrinks weights toward zero by subtracting
    rho_2 * ||w||_2^2 from the objective.
    """
    header("3a. L2 Penalty — SOFT (penalty_l2_w=5.0)")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=5.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}  (baseline: {baseline['L2_norm_sq']:.4f})")
    assert result["L2_norm_sq"] < baseline["L2_norm_sq"], "L2² should decrease"
    print("  PASS: L2² norm decreased with soft penalty")
    return result


def test_l2_hard():
    """
    Hard L2 constraint: ||w||_2^2 <= l2_limit.
    This turns the QP into a QCQP (quadratic constraint).
    """
    header("3b. L2 Penalty — HARD (l2_limit=1.0)")
    LIMIT = 1.0

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w='h',
        l2_limit=LIMIT,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}  (limit: {LIMIT})")
    assert result["L2_norm_sq"] <= LIMIT + 1e-5, \
        f"L2² {result['L2_norm_sq']} exceeds limit {LIMIT}"
    print("  PASS: L2² norm within hard limit")
    return result


def test_l2_hard_missing_limit():
    """
    Passing penalty_l2_w='h' without l2_limit must raise ValueError.
    """
    header("3c. L2 Hard — missing l2_limit (expect ValueError)")
    try:
        solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w='h',
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=None,
            gamma=0.0,
            turnover_limit=None,
            factor_bounds=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASS")


# ════════════════════════════════════════════════════════════════════
# 4. Turnover / transaction cost — soft vs hard
# ════════════════════════════════════════════════════════════════════

def test_turnover_soft():
    """
    Soft turnover penalises deviation from the reference portfolio:
      subtract gamma * 0.5 * sum|w_i - w_drift_i| from objective.
    """
    header("4a. Turnover — SOFT (gamma=0.3)")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=w_drift,
        gamma=0.3,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Turnover   : {turnover:.4f}")
    print(f"  Trans cost : {result['transaction_cost']:.6f}")

    # Verify TC formula:  sum( gamma * 0.5 * |w - w_drift| )
    w = result["weights"]
    expected_tc = np.sum(0.3 * 0.5 * np.abs(w - w_drift))
    assert np.isclose(result["transaction_cost"], expected_tc, atol=1e-5), \
        f"TC mismatch: got {result['transaction_cost']}, expected {expected_tc}"
    print("  PASS: TC matches formula")
    return result


def test_turnover_soft_per_asset():
    """
    Per-asset gamma: pass an array (n,) to penalise some assets more
    heavily than others.  First half gets gamma=0.5, second half 0.01.
    """
    header("4b. Turnover — SOFT per-asset gamma")

    # gamma_vec: (n,) float64 — per-asset turnover penalty
    gamma_vec = np.concatenate([np.full(n // 2, 0.5),
                                np.full(n - n // 2, 0.01)])

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=w_drift,
        gamma=gamma_vec,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Turnover   : {turnover:.4f}")
    print(f"  Trans cost : {result['transaction_cost']:.6f}")

    # Verify TC formula with per-asset gamma:
    #   sum_i gamma_i * 0.5 * |w_i - w_drift_i|
    w = result["weights"]
    expected_tc = np.sum(gamma_vec * 0.5 * np.abs(w - w_drift))
    assert np.isclose(result["transaction_cost"], expected_tc, atol=1e-5), \
        f"TC mismatch: got {result['transaction_cost']}, expected {expected_tc}"
    print("  PASS: TC matches per-asset formula")
    return result


def test_turnover_hard():
    """
    Hard turnover constraint: sum|w_i - w_drift_i| <= turnover_limit.
    """
    header("4c. Turnover — HARD (limit=0.5)")
    LIMIT = 0.5

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=w_drift,
        gamma='h',
        turnover_limit=LIMIT,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Turnover   : {turnover:.4f}  (limit: {LIMIT})")
    assert turnover <= LIMIT + 1e-5, \
        f"Turnover {turnover} exceeds limit {LIMIT}"
    print("  PASS: Turnover within hard limit")
    return result


def test_turnover_hard_missing_args():
    """
    gamma='h' without turnover_limit or w_drift must raise ValueError.
    """
    header("4d. Turnover Hard — missing args (expect ValueError)")

    # Missing turnover_limit
    try:
        solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=w_drift,
            gamma='h',
            turnover_limit=None,
            factor_bounds=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught (no turnover_limit): {e}")

    # Missing w_drift
    try:
        solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=None,
            gamma='h',
            turnover_limit=0.5,
            factor_bounds=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught (no w_drift)       : {e}")

    print("  PASS")


def test_turnover_zero_gamma_no_drift():
    """
    When gamma=0 and w_drift=None, turnover is disabled.
    Also test gamma=None for the same no-op behaviour.
    """
    header("4e. Turnover — gamma=0 / None, no w_drift (no-op)")

    for g in [None, 0, 0.0]:
        result = solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=None,
            gamma=g,
            turnover_limit=None,
            factor_bounds=None,
        )
        assert result is not None
        assert "optimal" in result["status"]
        assert calc_turnover(result, None) == 0.0
        assert result["transaction_cost"] == 0.0
    print("  PASS: All zero-gamma variants work without w_drift")


def test_turnover_invalid_gamma_type():
    """
    Invalid string gamma (anything other than 'h') must raise ValueError.
    """
    header("4f. Turnover — invalid gamma type (expect ValueError)")
    try:
        solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=w_drift,
            gamma='x',
            turnover_limit=None,
            factor_bounds=None,
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  PASS")


# ════════════════════════════════════════════════════════════════════
# 5. Factor bounds — soft vs hard
# ════════════════════════════════════════════════════════════════════

def test_factor_bounds_hard():
    """
    Hard factor bounds: lb_j <= y_j <= ub_j enforced as variable bounds.
    """
    header("5a. Factor Bounds — HARD")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor='h',
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=factor_bounds_default,
    )

    assert result is not None
    assert "optimal" in result["status"]
    y = result["factor_exposures"]
    print(f"  Factor exposures: {y.round(5)}")
    for j in range(k):
        assert y[j] >= factor_bounds_default[j, 0] - 1e-5, \
            f"Factor {j}: {y[j]:.6f} < lb {factor_bounds_default[j, 0]}"
        assert y[j] <= factor_bounds_default[j, 1] + 1e-5, \
            f"Factor {j}: {y[j]:.6f} > ub {factor_bounds_default[j, 1]}"
    print("  PASS: All factor exposures within hard bounds")
    return result


def test_legacy_result_schema_transaction_cost():
    """
    solve_mvo_flexible should keep the same transaction_cost field and
    formula used by optimizer_soft_constraint_regular_to.py.
    """
    header("5aa. Legacy transaction_cost field compatibility")

    gamma_val = 0.3
    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=w_drift,
        gamma=gamma_val,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    assert set(result.keys()) == LEGACY_RESULT_KEYS
    expected_tc = np.sum(gamma_val * 0.5 * np.abs(result["weights"] - w_drift))
    assert np.isclose(result["transaction_cost"], expected_tc, atol=1e-5), \
        f"TC mismatch: got {result['transaction_cost']}, expected {expected_tc}"
    print(f"  Trans cost : {result['transaction_cost']:.6f}")
    print("  PASS: solve_mvo_flexible transaction_cost matches old contract")
    return result


def test_factor_bounds_soft():
    """
    Soft factor bounds: quadratic deadzone penalty for violations.
    Factors may exceed bounds slightly, but are pulled toward them.
    """
    header("5b. Factor Bounds — SOFT (penalty_l2_factor=5.0)")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=5.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=factor_bounds_default,
    )

    assert result is not None
    assert "optimal" in result["status"]
    y = result["factor_exposures"]
    print(f"  Factor exposures: {y.round(5)}")
    print(f"  Bound violation² : {result['factor_diff_sq']:.6f}")
    print("  PASS: Solver converged (violations penalised, not forbidden)")
    return result


def test_factor_bounds_soft_increasing_penalty():
    """
    Increasing penalty_l2_factor should drive factor exposures
    closer to the feasible region (tighter compliance).
    """
    header("5c. Factor Bounds — SOFT, increasing penalty")

    prev_viol = None
    for coeff in [1.0, 10.0, 100.0, 1000.0]:
        r = solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=1.0,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=coeff,
            w_drift=None,
            gamma=0.0,
            turnover_limit=None,
            factor_bounds=factor_bounds_tight,
        )
        assert r is not None
        viol = r["factor_diff_sq"]
        direction = ""
        if prev_viol is not None:
            direction = " (↓)" if viol < prev_viol else " (↑)"
        print(f"  penalty={coeff:7.1f}  violation²={viol:.8f}{direction}")
        if prev_viol is not None:
            assert viol <= prev_viol + 1e-8, \
                "Violation should decrease (or stay) as penalty grows"
        prev_viol = viol
    print("  PASS: Violations monotonically decrease with higher penalty")


def test_factor_bounds_one_sided():
    """
    One-sided factor bounds using ±inf.
    Factor 0: upper-only  (y_0 <= 0.1)
    Factor 1: lower-only  (y_1 >= -0.2)
    Factor 2: fully free
    """
    header("5d. Factor Bounds — HARD, one-sided / inf")

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor='h',
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=factor_bounds_onesided,
    )

    assert result is not None
    assert "optimal" in result["status"]
    y = result["factor_exposures"]
    print(f"  Factor exposures: {y.round(5)}")
    assert y[0] <= 0.1  + 1e-5, f"Factor 0 above upper bound: {y[0]}"
    assert y[1] >= -0.2 - 1e-5, f"Factor 1 below lower bound: {y[1]}"
    print("  PASS: One-sided bounds respected; unbounded factor is free")
    return result


def test_factor_bounds_missing():
    """
    Passing penalty_l2_factor > 0 or 'h' without factor_bounds
    must raise ValueError.
    """
    header("5e. Factor Bounds — missing factor_bounds (expect ValueError)")
    for val in ['h', 5.0]:
        try:
            solve_mvo_flexible(
                alpha=alpha,
                B=B,
                F=F,
                D=D,
                lambda_risk=1.0,
                penalty_l1_w=0.0,
                l1_limit=None,
                penalty_l2_w=0.0,
                l2_limit=None,
                penalty_l2_factor=val,
                w_drift=None,
                gamma=0.0,
                turnover_limit=None,
                factor_bounds=None,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"  Caught (penalty_l2_factor={val!r}): {e}")
    print("  PASS")


# ════════════════════════════════════════════════════════════════════
# 6. Combined scenarios
# ════════════════════════════════════════════════════════════════════

def test_all_soft():
    """
    Every term enabled in soft mode simultaneously.
    """
    header("6a. Combined — all SOFT")

    fb = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]])

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.05,
        l1_limit=None,
        penalty_l2_w=1.0,
        l2_limit=None,
        penalty_l2_factor=10.0,
        w_drift=w_drift,
        gamma=0.2,
        turnover_limit=None,
        factor_bounds=fb,
    )

    assert result is not None
    assert "optimal" in result["status"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Status     : {result['status']}")
    print(f"  Objective  : {result['objective']:.6f}")
    print(f"  L1 norm    : {result['L1_norm']:.4f}")
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}")
    print(f"  Turnover   : {turnover:.4f}")
    print(f"  Trans cost : {result['transaction_cost']:.6f}")
    print(f"  Bound viol²: {result['factor_diff_sq']:.6f}")
    print("  PASS")
    return result


def test_all_hard():
    """
    Every term enabled in hard mode simultaneously.
    """
    header("6b. Combined — all HARD")

    fb = np.array([[-0.3, 0.3], [-0.3, 0.3], [-0.3, 0.3]])
    L1_LIM = 3.0
    L2_LIM = 2.0
    TO_LIM = 1.0

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w='h',
        l1_limit=L1_LIM,
        penalty_l2_w='h',
        l2_limit=L2_LIM,
        penalty_l2_factor='h',
        w_drift=w_drift,
        gamma='h',
        turnover_limit=TO_LIM,
        factor_bounds=fb,
    )

    assert result is not None
    assert "optimal" in result["status"]
    y = result["factor_exposures"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Status     : {result['status']}")
    print(f"  L1 norm    : {result['L1_norm']:.4f}  (limit {L1_LIM})")
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}  (limit {L2_LIM})")
    print(f"  Turnover   : {turnover:.4f}  (limit {TO_LIM})")
    print(f"  Factors    : {y.round(5)}")

    assert result["L1_norm"]   <= L1_LIM + 1e-5
    assert result["L2_norm_sq"] <= L2_LIM + 1e-5
    assert turnover <= TO_LIM + 1e-5
    for j in range(len(fb)):
        assert y[j] >= fb[j, 0] - 1e-5
        assert y[j] <= fb[j, 1] + 1e-5
    print("  PASS: All hard limits satisfied")
    return result


def test_mixed_hard_soft():
    """
    Mix of hard and soft:
      - hard L1     (penalty_l1_w='h', l1_limit=2.5)
      - soft L2     (penalty_l2_w=2.0)
      - soft turnover (gamma=0.2)
      - hard factor bounds (penalty_l2_factor='h')
    """
    header("6c. Combined — MIXED hard + soft")

    fb = np.array([[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]])
    L1_LIM = 2.5

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w='h',
        l1_limit=L1_LIM,
        penalty_l2_w=2.0,
        l2_limit=None,
        penalty_l2_factor='h',
        w_drift=w_drift,
        gamma=0.2,
        turnover_limit=None,
        factor_bounds=fb,
    )

    assert result is not None
    assert "optimal" in result["status"]
    y = result["factor_exposures"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Status     : {result['status']}")
    print(f"  L1 norm    : {result['L1_norm']:.4f}  (hard limit {L1_LIM})")
    print(f"  L2² norm   : {result['L2_norm_sq']:.4f}  (soft)")
    print(f"  Turnover   : {turnover:.4f}  (soft)")
    print(f"  Factors    : {y.round(5)}  (hard bounds)")

    assert result["L1_norm"] <= L1_LIM + 1e-5
    for j in range(len(fb)):
        assert y[j] >= fb[j, 0] - 1e-5
        assert y[j] <= fb[j, 1] + 1e-5
    print("  PASS: Hard limits satisfied, soft terms active")
    return result


# ════════════════════════════════════════════════════════════════════
# 7. Edge cases
# ════════════════════════════════════════════════════════════════════

def test_tight_hard_l1():
    """
    Very tight L1 limit (smaller than baseline) should shrink L1 down
    to the budget.
    """
    header("7a. Edge — very tight hard L1 (limit=0.5)")
    LIMIT = 0.5

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w='h',
        l1_limit=LIMIT,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=None,
        gamma=0.0,
        turnover_limit=None,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    print(f"  L1 norm    : {result['L1_norm']:.4f}  (limit {LIMIT})")
    assert result["L1_norm"] <= LIMIT + 1e-5
    print("  PASS")


def test_tight_hard_turnover():
    """
    Very tight turnover limit forces weights close to w_drift.
    """
    header("7b. Edge — very tight hard turnover (limit=0.1)")
    LIMIT = 0.1

    result = solve_mvo_flexible(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        lambda_risk=1.0,
        penalty_l1_w=0.0,
        l1_limit=None,
        penalty_l2_w=0.0,
        l2_limit=None,
        penalty_l2_factor=0.0,
        w_drift=w_drift,
        gamma='h',
        turnover_limit=LIMIT,
        factor_bounds=None,
    )

    assert result is not None
    assert "optimal" in result["status"]
    turnover = calc_turnover(result, w_drift)
    print(f"  Turnover   : {turnover:.4f}  (limit {LIMIT})")
    diff = np.max(np.abs(result["weights"] - w_drift))
    print(f"  Max |w-w0| : {diff:.6f}")
    assert turnover <= LIMIT + 1e-5
    print("  PASS: Weights very close to w_drift")


def test_lambda_risk_sweep():
    """
    Sweeping lambda_risk from low → high should reduce portfolio variance
    and L2² norm (weights become more conservative).
    """
    header("7c. Edge — lambda_risk sweep")

    prev_l2 = None
    for lam in [0.1, 1.0, 5.0, 20.0]:
        r = solve_mvo_flexible(
            alpha=alpha,
            B=B,
            F=F,
            D=D,
            lambda_risk=lam,
            penalty_l1_w=0.0,
            l1_limit=None,
            penalty_l2_w=0.0,
            l2_limit=None,
            penalty_l2_factor=0.0,
            w_drift=None,
            gamma=0.0,
            turnover_limit=None,
            factor_bounds=None,
        )
        assert r is not None
        direction = ""
        if prev_l2 is not None:
            direction = " (↓)" if r["L2_norm_sq"] < prev_l2 else ""
        print(f"  lambda_risk={lam:5.1f}  L2²={r['L2_norm_sq']:.4f}{direction}")
        if prev_l2 is not None:
            assert r["L2_norm_sq"] <= prev_l2 + 1e-6, \
                "Higher risk aversion should shrink weights"
        prev_l2 = r["L2_norm_sq"]
    print("  PASS: L2² decreases with higher risk aversion")


# ════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════

def run_all():
    print("=" * 60)
    print("  test_optimizer_flexible.py — solve_mvo_flexible()")
    print("=" * 60)

    # Print input data summary so the reader knows what's going in
    print(f"\n  Input data:  n={n} assets, k={k} factors, seed=42")
    print(f"  alpha            : shape={alpha.shape}  dtype={alpha.dtype}")
    print(f"  B (loadings)     : shape={B.shape}  dtype={B.dtype}")
    print(f"  F (factor cov)   : shape={F.shape}  dtype={F.dtype}")
    print(f"  D (idio var)     : shape={D.shape}  dtype={D.dtype}")
    print(f"  w_drift (ref wts): shape={w_drift.shape}  dtype={w_drift.dtype}  sum={w_drift.sum():.4f}")
    print(f"  factor_bounds    : shape={factor_bounds_default.shape}  (default)")
    print(f"  factor_bounds    : shape={factor_bounds_tight.shape}  (tight)")
    print(f"  factor_bounds    : shape={factor_bounds_onesided.shape}  (one-sided)")

    baseline = test_baseline()
    test_legacy_result_schema_baseline()

    # L1
    test_l1_soft(baseline)
    test_l1_hard()
    test_l1_hard_missing_limit()

    # L2
    test_l2_soft(baseline)
    test_l2_hard()
    test_l2_hard_missing_limit()

    # Turnover
    test_turnover_soft()
    test_turnover_soft_per_asset()
    test_turnover_hard()
    test_turnover_hard_missing_args()
    test_turnover_zero_gamma_no_drift()
    test_turnover_invalid_gamma_type()

    # Factor bounds
    test_factor_bounds_hard()
    test_legacy_result_schema_transaction_cost()
    test_factor_bounds_soft()
    test_factor_bounds_soft_increasing_penalty()
    test_factor_bounds_one_sided()
    test_factor_bounds_missing()

    # Combined
    test_all_soft()
    test_all_hard()
    test_mixed_hard_soft()

    # Edge cases
    test_tight_hard_l1()
    test_tight_hard_turnover()
    test_lambda_risk_sweep()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
