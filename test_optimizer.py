"""
Test script for the MVO factor auxiliary optimizer with synthetic data.
"""
import numpy as np
from optimizer import solve_mvo_factor_auxiliary


def generate_psd_matrix_with_condition_number(n, condition_number, scale=1.0, seed=None):
    """
    Generate a positive semi-definite matrix with a specified condition number.
    
    Parameters:
    -----------
    n : int
        Size of the matrix (n x n)
    condition_number : float
        Desired condition number (ratio of largest to smallest eigenvalue)
        Must be >= 1.0
    scale : float
        Scale factor for eigenvalues (controls overall magnitude)
        Default is 1.0
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    numpy.ndarray
        Positive semi-definite matrix of shape (n, n) with the specified condition number
    
    Notes:
    ------
    The condition number is defined as κ(A) = λ_max / λ_min
    A well-conditioned matrix has κ close to 1, while an ill-conditioned matrix has large κ.
    """
    if condition_number < 1.0:
        raise ValueError("Condition number must be >= 1.0")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate eigenvalues logarithmically spaced between λ_min and λ_max
    # where λ_max / λ_min = condition_number
    lambda_min = scale / condition_number
    lambda_max = scale
    
    # Logarithmically space the eigenvalues for better numerical properties
    eigenvalues = np.logspace(np.log10(lambda_min), np.log10(lambda_max), n)
    
    # Generate a random orthogonal matrix via QR decomposition
    random_matrix = np.random.randn(n, n)
    Q, _ = np.linalg.qr(random_matrix)
    
    # Construct PSD matrix: A = Q @ diag(eigenvalues) @ Q.T
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    # Ensure symmetry (to handle numerical errors)
    A = (A + A.T) / 2
    
    return A


def generate_synthetic_data(n=10, k=3, seed=42):
    """
    Generate synthetic data for portfolio optimization.
    
    Parameters:
    -----------
    n : int
        Number of assets
    k : int
        Number of factors
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict with alpha, B, F, D, w_drift
    """
    np.random.seed(seed)
    
    # Alpha: expected returns for each asset
    alpha = np.random.randn(n) * 0.05  # ~5% annual returns
    
    # B: factor loading matrix (n x k)
    # Each asset has exposure to k factors
    B = np.random.randn(n, k) * 0.5
    
    # F and D generation removed as requested
    
    # w_drift: initial portfolio weights (for transaction cost)
    w_drift = np.random.dirichlet(np.ones(n))  # Sums to 1
    
    return {
        'alpha': alpha,
        'B': B,
        'w_drift': w_drift
    }


def main():

    n_assets = 1000
    n_factors = 7
    
    data = generate_synthetic_data(n=n_assets, k=n_factors)
    
    np.random.seed(123)
    w_raw = np.random.randn(n_assets)
    w_raw = w_raw / np.sum(np.abs(w_raw))  # Normalize to L1 norm = 1
    
    
    # Generate Sigma directly with a specific condition number
    target_condition_number = 10000.0
    Sigma = generate_psd_matrix_with_condition_number(n_assets, target_condition_number, seed=123)

    alpha_inverse = Sigma @ w_raw
    
    # Solve with lambda_risk = 0.5 to get w* = w_raw
    # We pass B=Identity, F=Sigma, D=0 to treat the problem as a full covariance optimization
    result = solve_mvo_factor_auxiliary(
        alpha=alpha_inverse,
        B=np.eye(n_assets),
        F=Sigma,
        D=np.zeros(n_assets),
        lambda_risk=0.5  # With this, w* = w_raw
    )
    
    if result:

        for i in range(n_assets):
            diff = result['weights'][i] - w_raw[i]
            print(f"{i:<8} {w_raw[i]:<15.6f} {result['weights'][i]:<15.6f} {diff:<15.6e}")
        
        max_diff = np.max(np.abs(result['weights'] - w_raw))
        print(f"\nMax absolute difference: {max_diff:.6e}")
        print(f"Are weights equal (tol=1e-5)? {np.allclose(result['weights'], w_raw, atol=1e-5)}")


if __name__ == "__main__":
    main()
