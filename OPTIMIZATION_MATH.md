# Mathematical Optimization in Portfolio Construction

This document explains the mathematical concepts and optimization methods implemented in the `optimizer.py` code, which implements a **Mean-Variance Optimization (MVO)** framework with factor risk models and transaction costs.

## Table of Contents
1. [Overview of Mean-Variance Optimization](#overview-of-mean-variance-optimization)
2. [Factor Risk Model Architecture](#factor-risk-model-architecture)
3. [Optimization Problem Formulation](#optimization-problem-formulation)
4. [Transaction Cost Modeling](#transaction-cost-modeling)
5. [L1 Regularization for Sparsity](#l1-regularization-for-sparsity)
6. [CPLEX Implementation Details](#cplex-implementation-details)
7. [Mathematical Transformation to Linear Programming](#mathematical-transformation-to-linear-programming)
8. [Performance Optimization Parameters](#performance-optimization-parameters)

## Overview of Mean-Variance Optimization

The code implements **Markowitz Mean-Variance Optimization (MVO)**, which balances expected return against portfolio risk. The fundamental tradeoff is captured by:

```
Maximize: α'w - λ_risk * w'Σw
Subject to: Constraints on portfolio weights
```

Where:
- **w** = vector of portfolio weights (n assets)
- **α** = vector of expected returns (alpha) for each asset
- **Σ** = covariance matrix of asset returns (n×n)
- **λ_risk** = risk aversion parameter (higher λ means more risk-averse)

## Factor Risk Model Architecture

Instead of using the full covariance matrix Σ directly (which can be unstable for large n), the code uses a **factor risk model** that decomposes risk into systematic (factor) and specific (idiosyncratic) components.

### Mathematical Decomposition:
```
Total Risk = Factor Risk + Specific Risk
w'Σw = (Bw)'F(Bw) + w'Dw
```

Where:
- **B** = n×k factor loading matrix (n assets, k factors)
- **F** = k×k factor covariance matrix
- **D** = n×n diagonal specific risk matrix (or vector of diagonal elements)

This decomposition reduces dimensionality from O(n²) to O(n×k + k²), making the optimization more stable and computationally efficient.

## Optimization Problem Formulation

The complete optimization problem solved by the code is:

### Objective Function:
```
Maximize: α'w - λ_risk * [(Bw)'F(Bw) + w'Dw] - γ * ½ * (|w - w_drift| + |w| - |w_drift|)
```

### Constraints:
1. **Factor exposure constraints**: Bw = y (with optional bounds on y)
2. **L1 constraint**: ∑|w_i| ≤ L1_limit (promotes sparsity)
3. **Drift penalty**: Controls transaction costs relative to previous portfolio w_drift

## Transaction Cost Modeling

The code implements a sophisticated transaction cost model that penalizes deviations from a previous portfolio (`w_drift`):

### Transaction Cost Formula:
```
Transaction Cost = γ * ½ * (|w - w_drift| + |w| - |w_drift|)
```

### Mathematical Explanation:
- **|w - w_drift|** = absolute deviation from previous portfolio
- **|w| - |w_drift|** = change in total exposure magnitude
- **γ** = transaction cost coefficient (can be scalar or asset-specific)

This formulation captures both the **cost of trading** (changing positions) and the **cost of maintaining exposure** (holding positions).

### Implementation Trick:
The absolute value terms |w| and |w - w_drift| are handled using **auxiliary variables** in the linear programming formulation (see Section 7).

## L1 Regularization for Sparsity

The code includes an L1-norm constraint that promotes **sparse portfolios**:

```
∑|w_i| ≤ L1_limit
```

### Benefits of Sparsity:
1. **Reduced transaction costs**: Fewer assets to trade
2. **Improved interpretability**: Easier to understand concentrated portfolios
3. **Reduced estimation error**: Less sensitive to noisy return estimates
4. **Practical implementation**: Fewer positions to manage

### Mathematical Handling:
The absolute value |w_i| is implemented using auxiliary variable z_i with constraints:
- z_i ≥ w_i
- z_i ≥ -w_i
- z_i ≥ 0

This transforms the non-linear absolute value into linear constraints.

## CPLEX Implementation Details

### Variables Structure:
1. **Primary variables**: w_i (n asset weights)
2. **Factor variables**: y_j (k factor exposures, where y = B'w)
3. **Auxiliary variables**:
   - z_i for |w_i| (used in L1 constraint and transaction cost)
   - d_i for |w_i - w_drift_i| (used in transaction cost)

### Quadratic Programming Formulation:
The code solves a **Quadratic Programming (QP)** problem using CPLEX's barrier method:
- **Linear terms**: α'w (expected return) and transaction cost terms
- **Quadratic terms**: -λ_risk * (y'Fy + w'Dw) (risk penalty)

### CPLEX Performance Settings:
```python
prob.parameters.threads.set(0)            # Use all available threads
prob.parameters.qpmethod.set(prob.parameters.qpmethod.values.barrier)
prob.parameters.barrier.convergetol.set(1e-6)
prob.parameters.preprocessing.presolve.set(1)
prob.parameters.preprocessing.reduce.set(3)
```

## Mathematical Transformation to Linear Programming

### Handling Absolute Values:
The key innovation is transforming absolute values into linear constraints using auxiliary variables:

#### For |w_i|:
1. Introduce z_i ≥ 0
2. Add constraints: z_i ≥ w_i and z_i ≥ -w_i
3. Then |w_i| ≤ z_i (and at optimality, z_i = |w_i|)

#### For |w_i - w_drift_i|:
1. Introduce d_i ≥ 0
2. Add constraints: d_i ≥ w_i - w_drift_i and d_i ≥ -(w_i - w_drift_i)
3. Then |w_i - w_drift_i| ≤ d_i (and at optimality, d_i = |w_i - w_drift_i|)

### Objective Function with Auxiliary Variables:
The transaction cost term becomes:
```
-½γ_i * (d_i + z_i - |w_drift_i|)
```
Since |w_drift_i| is constant, it doesn't affect the optimization (only shifts the objective value).

## Performance Optimization Parameters

### Factor Covariance Handling:
The code optimizes for both diagonal and full factor covariance matrices:

```python
if F.ndim == 1 or (F.ndim == 2 and np.allclose(F, np.diag(np.diag(F)))):
    # Diagonal F: O(k) quadratic terms
    F_diag = F if F.ndim == 1 else np.diag(F)
    for j in range(k):
        quad_coeffs.append((n + j, n + j, -2.0 * lambda_risk * F_diag[j]))
else:
    # Full F: O(k²) quadratic terms
    for i in range(k):
        for j in range(k):
            quad_coeffs.append((n + i, n + j, -2.0 * lambda_risk * F[i, j]))
```

### Specific Risk Handling:
Specific risk D is always treated as diagonal:
```python
D_diag = D if D.ndim == 1 else np.diag(D)
for i in range(n):
    quad_coeffs.append((i, i, -2.0 * lambda_risk * D_diag[i]))
```

## Practical Applications

This optimization framework is particularly useful for:

1. **Portfolio construction** with factor-based risk models
2. **Transaction-aware optimization** that accounts for trading costs
3. **Sparse portfolio selection** via L1 regularization
4. **Tactical asset allocation** relative to a strategic benchmark (w_drift)
5. **Risk-budgeting** across factors via factor exposure limits

## Key Mathematical Insights

1. **Factor models reduce dimensionality** while capturing systematic risk
2. **Absolute values can be linearized** using auxiliary variables
3. **Transaction costs are convex** and can be incorporated into optimization
4. **Sparsity via L1 regularization** improves practical implementation
5. **Quadratic programming** efficiently handles mean-variance tradeoffs

## References

1. Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*
2. Grinold, R. C., & Kahn, R. N. (1999). *Active Portfolio Management*
3. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*
4. CPLEX Documentation: Quadratic Programming with Absolute Values