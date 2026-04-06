"""
Test and usage example for optimizer_target_variance.py.

Run:
    python test_optimizer_target_variance.py
"""
import numpy as np

from optimizer_target_variance import solve_mvo_target_variance


def header(title):
    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")


def main():
    np.random.seed(42)

    n = 24
    k = 4

    alpha = np.random.randn(n) * 0.03
    B = np.random.randn(n, k)

    factor_cov_root = np.random.randn(k, k)
    F = 0.05 * np.eye(k) + 0.02 * (factor_cov_root @ factor_cov_root.T)
    D = 0.03 + 0.02 * np.random.rand(n)

    low_target = 0.12
    high_target = 0.30
    tol = 1e-6

    header("Low target variance")
    low_result = solve_mvo_target_variance(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        target_variance=low_target,
    )

    assert low_result is not None, "Solver returned None for low target"
    assert "optimal" in low_result["status"], f"Unexpected status: {low_result['status']}"
    assert low_result["total_variance"] <= low_target + tol
    assert np.allclose(low_result["factor_exposures"], B.T @ low_result["weights"], atol=1e-6)

    print(f"Status           : {low_result['status']}")
    print(f"Expected return  : {low_result['expected_return']:.8f}")
    print(f"Total variance   : {low_result['total_variance']:.8f}")
    print(f"Variance slack   : {low_result['variance_slack']:.8f}")

    header("Higher target variance")
    high_result = solve_mvo_target_variance(
        alpha=alpha,
        B=B,
        F=F,
        D=D,
        target_variance=high_target,
    )

    assert high_result is not None, "Solver returned None for high target"
    assert "optimal" in high_result["status"], f"Unexpected status: {high_result['status']}"
    assert high_result["total_variance"] <= high_target + tol
    assert np.allclose(high_result["factor_exposures"], B.T @ high_result["weights"], atol=1e-6)
    assert high_result["expected_return"] >= low_result["expected_return"] - 1e-8

    print(f"Status           : {high_result['status']}")
    print(f"Expected return  : {high_result['expected_return']:.8f}")
    print(f"Total variance   : {high_result['total_variance']:.8f}")
    print(f"Variance slack   : {high_result['variance_slack']:.8f}")

    print("\nPASS: target-variance MVO solver satisfies the variance cap.")


if __name__ == "__main__":
    main()
