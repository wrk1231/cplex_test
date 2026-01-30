# Mosek Fusion API Translation Plan

This document provides an AI-agent friendly plan to translate the CPLEX optimizer implementation to Mosek Fusion API.

---

## 1. Environment Setup

### 1.1 Create Conda Environment

```bash
# Create conda environment with Python 3.12
conda create -n mk_opt_local python=3.12 -y

# Activate the environment
conda activate mk_opt_local
```

### 1.2 Install Dependencies

```bash
# Install Mosek (requires license file or academic license)
pip install mosek

# Install numpy for numerical operations
pip install numpy

# Install pytest for test cases
pip install pytest
```

### 1.3 Verify Mosek Installation

```python
import mosek
from mosek.fusion import *
print(f"Mosek version: {mosek.getversion()}")
```

> **Note**: Mosek requires a valid license. Academic licenses are free. Set the `MOSEKLM_LICENSE_FILE` environment variable to point to the license file.

---

## 2. CPLEX to Mosek API Translation Reference

### 2.1 Core API Mapping

| CPLEX Construct | Mosek Fusion Equivalent | Notes |
|-----------------|------------------------|-------|
| `cplex.Cplex()` | `Model("name")` | Create optimization model |
| `prob.set_log_stream(None)` | `M.setLogHandler(None)` or `M.setSolverParam("log", 0)` | Suppress logging |
| `prob.objective.set_sense(maximize)` | `ObjectiveSense.Maximize` | Set objective direction |
| `prob.variables.add(obj, lb, ub, names)` | `M.variable("name", n, Domain...)` | Add variables |
| `cplex.infinity` | `float('inf')` or no bound | Unbounded variables |
| `prob.linear_constraints.add()` | `M.constraint("name", Expr.constraint)` | Add linear constraints |
| `prob.objective.set_quadratic_coefficients()` | Conic reformulation (see Section 2.3) | Quadratic terms |
| `prob.solve()` | `M.solve()` | Solve the problem |
| `prob.solution.get_values()` | `var.level()` | Get solution values |
| `prob.solution.get_objective_value()` | `M.primalObjValue()` | Get objective value |
| `prob.solution.get_status_string()` | `M.getProblemStatus()` | Get solution status |

### 2.2 Variable Domain Mappings

| CPLEX Bound | Mosek Domain |
|-------------|--------------|
| `lb=-infinity, ub=infinity` | `Domain.unbounded()` |
| `lb=0, ub=infinity` | `Domain.greaterThan(0.0)` |
| `lb=-infinity, ub=value` | `Domain.lessThan(value)` |
| `lb=value1, ub=value2` | `Domain.inRange(value1, value2)` |

### 2.3 Quadratic Objective Reformulation (Critical)

**CPLEX** uses explicit quadratic objective coefficients. **Mosek Fusion** prefers conic formulations for quadratic terms.

For the quadratic risk term: `w'Σw = y'Fy + w'Dw`

**Option A: Conic Reformulation (Recommended)**

Transform `t ≥ ||Lx||` where `L'L = Σ` (Cholesky) using a rotated quadratic cone:

```python
# For y'Fy where F = L_F @ L_F.T (Cholesky)
# Add auxiliary variable t_F for factor risk
t_F = M.variable("t_F", Domain.greaterThan(0.0))
# Conic constraint: t_F >= ||L_F @ y||_2
# Using rotated cone: 2*t_F*1 >= ||L_F @ y||^2
M.constraint("factor_risk_cone", 
    Expr.vstack(t_F, Expr.constTerm(0.5), Expr.mul(L_F, y)),
    Domain.inRotatedQCone())

# For diagonal D: w'Dw = sum(D_i * w_i^2)
# Add auxiliary variable t_D for specific risk
t_D = M.variable("t_D", Domain.greaterThan(0.0))
# D_sqrt = sqrt(D) (element-wise)
M.constraint("specific_risk_cone",
    Expr.vstack(t_D, Expr.constTerm(0.5), Expr.mulElm(D_sqrt, w)),
    Domain.inRotatedQCone())

# Objective: max alpha'w - lambda_risk * (t_F^2 + t_D^2)
# Simplified to: max alpha'w - lambda_risk * (t_F + t_D) after cone reformulation
```

**Option B: Direct Quadratic Objective (qobj)**

Mosek also supports direct quadratic objectives:

```python
# For quadratic objective: 0.5 * x'Qx
# Q is the quadratic coefficient matrix
M.objective("obj", ObjectiveSense.Maximize,
    Expr.sub(
        Expr.dot(alpha, w),  # Linear term
        Expr.mul(lambda_risk, Expr.quadForm(x, Q))  # Quadratic term
    ))
```

### 2.4 Absolute Value Constraints Translation

The CPLEX implementation uses auxiliary variables for `|w_i|`. Mosek supports this directly:

```python
# For z_i >= |w_i|:
# Mosek has Domain.inQCone() which models ||x|| <= t
# Alternative: two linear constraints
# z >= w  => z - w >= 0
# z >= -w => z + w >= 0

z = M.variable("z", n, Domain.greaterThan(0.0))
M.constraint("abs_pos", Expr.sub(z, w), Domain.greaterThan(0.0))
M.constraint("abs_neg", Expr.add(z, w), Domain.greaterThan(0.0))
```

---

## 3. Implementation Steps

### 3.1 Create `optimizer_mosek.py`

Create a new file `/home/wrk1231/projects/cplex_test/optimizer_mosek.py` with the following structure:

```python
import numpy as np
from mosek.fusion import *

def solve_mvo_factor_auxiliary_mosek(
    alpha,           # Expected returns (n,)
    B,               # Factor loadings (n, k)
    F,               # Factor covariance (k, k) or (k,) if diagonal
    D,               # Idiosyncratic variance (n,) or (n, n) diagonal
    lambda_risk=1,   # Risk aversion parameter
    L1_limit=None,   # L1 constraint on weights
    factor_exposure_limits=None,  # Bounds on factor exposures
    w_drift=None,    # Previous portfolio weights
    gamma=None       # Transaction cost coefficient
):
    """
    Mosek Fusion implementation of MVO with factor model.
    
    Solves: max alpha'w - lambda_risk * (y'Fy + w'Dw) - transaction_cost
    Subject to:
        y = B'w (factor exposures)
        sum(|w|) <= L1_limit (if specified)
        y_lb <= y <= y_ub (if specified)
        transaction cost terms
    """
    pass  # Implementation follows steps below
```

### 3.2 Step-by-Step Translation Logic

#### Step 1: Initialize Model and Parse Parameters

```python
n, k = B.shape

# Parse gamma and w_drift for transaction costs
apply_drift_penalty = False
gamma_arr = None
w_drift_arr = None

if gamma is not None and not (np.isscalar(gamma) and gamma == 0):
    gamma_test = np.asarray(gamma)
    if np.any(gamma_test != 0):
        if w_drift is None:
            raise ValueError("w_drift must be provided when gamma is non-zero")
        apply_drift_penalty = True
        gamma_arr = np.full(n, gamma) if np.isscalar(gamma) else np.asarray(gamma)
        w_drift_arr = np.asarray(w_drift)

# Create Mosek model
with Model("MVO_Factor") as M:
    M.setLogHandler(None)  # Suppress output
```

#### Step 2: Define Variables

```python
# Weight variables (unbounded)
w = M.variable("w", n, Domain.unbounded())

# Factor exposure variables with optional bounds
if factor_exposure_limits is not None:
    if 'bounds' in factor_exposure_limits:
        bounds = factor_exposure_limits['bounds']
        y_lb = [b[0] if b[0] is not None else -float('inf') for b in bounds]
        y_ub = [b[1] if b[1] is not None else float('inf') for b in bounds]
    else:
        y_lb = factor_exposure_limits.get('lower', [-float('inf')] * k)
        y_ub = factor_exposure_limits.get('upper', [float('inf')] * k)
    y = M.variable("y", k, Domain.inRange(y_lb, y_ub))
else:
    y = M.variable("y", k, Domain.unbounded())
```

#### Step 3: Add Auxiliary Variables for Absolute Values

```python
# z_i >= |w_i| for L1 constraint and transaction cost
z = None
if L1_limit is not None or apply_drift_penalty:
    z = M.variable("z", n, Domain.greaterThan(0.0))
    # z >= w
    M.constraint("z_ge_w", Expr.sub(z, w), Domain.greaterThan(0.0))
    # z >= -w
    M.constraint("z_ge_negw", Expr.add(z, w), Domain.greaterThan(0.0))

# d_i >= |w_i - w_drift_i| for transaction cost
d = None
if apply_drift_penalty:
    d = M.variable("d", n, Domain.greaterThan(0.0))
    # d >= w - w_drift
    M.constraint("d_ge_diff", 
        Expr.sub(d, Expr.sub(w, Expr.constTerm(w_drift_arr))),
        Domain.greaterThan(0.0))
    # d >= -(w - w_drift)
    M.constraint("d_ge_negdiff",
        Expr.add(d, Expr.sub(w, Expr.constTerm(w_drift_arr))),
        Domain.greaterThan(0.0))
```

#### Step 4: Add Constraints

```python
# y = B'w constraint
M.constraint("factor_exposure", 
    Expr.sub(y, Expr.mul(B.T, w)),
    Domain.equalsTo(0.0))

# L1 constraint: sum(z) <= L1_limit
if L1_limit is not None:
    M.constraint("L1_limit", 
        Expr.sum(z), 
        Domain.lessThan(L1_limit))
```

#### Step 5: Build Quadratic Objective Using Conic Reformulation

```python
# Approach: Reformulate quadratic terms as conic constraints
# Risk = y'Fy + w'Dw

# For y'Fy: Use Cholesky decomposition F = L_F @ L_F.T
# Then y'Fy = ||L_F.T @ y||^2
# Introduce t_factor such that t_factor >= ||L_F.T @ y||^2
# Using rotated cone: (t_factor, 0.5, L_F.T @ y) in Q_r

t_factor = M.variable("t_factor", Domain.greaterThan(0.0))

if F.ndim == 1 or (F.ndim == 2 and np.allclose(F, np.diag(np.diag(F)))):
    # Diagonal F
    F_diag = F if F.ndim == 1 else np.diag(F)
    F_sqrt = np.sqrt(np.abs(F_diag))
    M.constraint("factor_risk_cone",
        Expr.vstack(t_factor, 
                   Expr.constTerm(0.5),
                   Expr.mulElm(Matrix.dense(F_sqrt.reshape(-1, 1)), y)),
        Domain.inRotatedQCone())
else:
    # Full F matrix - Cholesky decomposition
    L_F = np.linalg.cholesky(F)
    M.constraint("factor_risk_cone",
        Expr.vstack(t_factor,
                   Expr.constTerm(0.5),
                   Expr.mul(Matrix.dense(L_F.T), y)),
        Domain.inRotatedQCone())

# For w'Dw (D is diagonal)
t_specific = M.variable("t_specific", Domain.greaterThan(0.0))
D_diag = D if D.ndim == 1 else np.diag(D)
D_sqrt = np.sqrt(np.abs(D_diag))

M.constraint("specific_risk_cone",
    Expr.vstack(t_specific,
               Expr.constTerm(0.5),
               Expr.mulElm(Matrix.dense(D_sqrt.reshape(-1, 1)), w)),
    Domain.inRotatedQCone())
```

#### Step 6: Build Full Objective

```python
# Objective: max alpha'w - lambda_risk * (t_factor + t_specific) - transaction_cost
obj_expr = Expr.dot(alpha, w)  # alpha'w

# Subtract risk term: -lambda_risk * (t_factor + t_specific)
risk_term = Expr.mul(lambda_risk, Expr.add(t_factor, t_specific))
obj_expr = Expr.sub(obj_expr, risk_term)

# Subtract transaction cost if applicable: -0.5 * gamma * (d + z)
if apply_drift_penalty:
    tc_coeff = 0.5 * gamma_arr
    tc_expr = Expr.add(
        Expr.dot(tc_coeff, d),
        Expr.dot(tc_coeff, z)
    )
    obj_expr = Expr.sub(obj_expr, tc_expr)

M.objective("obj", ObjectiveSense.Maximize, obj_expr)
```

#### Step 7: Solve and Extract Solution

```python
# Set solver parameters
M.setSolverParam("intpntCoTolRelGap", 1e-6)  # Convergence tolerance
M.setSolverParam("numThreads", 0)  # Use all threads

# Solve
M.solve()

# Check status
if M.getProblemStatus() == ProblemStatus.PrimalAndDualFeasible:
    weights = np.array(w.level())
    factor_exposures = np.array(y.level())
    
    # Calculate derived values
    drift_deviation = None
    transaction_cost = None
    if apply_drift_penalty:
        drift_deviation = np.abs(weights - w_drift_arr)
        transaction_cost = np.sum(gamma_arr * 0.5 * (
            drift_deviation + np.abs(weights) - np.abs(w_drift_arr)
        ))
    
    return {
        "weights": weights,
        "objective": M.primalObjValue(),
        "status": str(M.getProblemStatus()),
        "L1_norm": np.sum(np.abs(weights)),
        "factor_exposures": factor_exposures,
        "drift_deviation": drift_deviation,
        "transaction_cost": transaction_cost
    }
else:
    print(f"Mosek Status: {M.getProblemStatus()}")
    return None
```

---

## 4. Test Cases

### 4.1 Create `test_optimizer_mosek.py`

Create file `/home/wrk1231/projects/cplex_test/test_optimizer_mosek.py`:

```python
"""
Test script for Mosek Fusion implementation of MVO factor optimizer.
Validates equivalence with CPLEX implementation.
"""
import numpy as np
import pytest
from optimizer_mosek import solve_mvo_factor_auxiliary_mosek

# Import helper functions from existing test
from test_optimizer import (
    generate_psd_matrix_with_condition_number,
    generate_synthetic_data
)

# Optional: Import CPLEX version for comparison
try:
    from optimizer import solve_mvo_factor_auxiliary
    CPLEX_AVAILABLE = True
except ImportError:
    CPLEX_AVAILABLE = False
```

### 4.2 Test Case 1: Basic Optimization Without Constraints

```python
def test_basic_mvo():
    """Test basic MVO without additional constraints."""
    n, k = 10, 3
    data = generate_synthetic_data(n=n, k=k, seed=42)
    
    # Generate F and D
    F = generate_psd_matrix_with_condition_number(k, 10.0, seed=42)
    D = np.abs(np.random.randn(n)) * 0.01
    
    result = solve_mvo_factor_auxiliary_mosek(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=1.0
    )
    
    assert result is not None, "Mosek solver returned None"
    assert 'weights' in result
    assert len(result['weights']) == n
    assert 'objective' in result
    print(f"Objective: {result['objective']:.6f}")
```

### 4.3 Test Case 2: L1 Constraint

```python
def test_l1_constraint():
    """Test MVO with L1 leverage constraint."""
    n, k = 20, 5
    data = generate_synthetic_data(n=n, k=k, seed=123)
    
    F = generate_psd_matrix_with_condition_number(k, 100.0, seed=123)
    D = np.abs(np.random.randn(n)) * 0.02
    
    L1_limit = 1.5
    
    result = solve_mvo_factor_auxiliary_mosek(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=0.5,
        L1_limit=L1_limit
    )
    
    assert result is not None
    assert result['L1_norm'] <= L1_limit + 1e-6, \
        f"L1 constraint violated: {result['L1_norm']} > {L1_limit}"
    print(f"L1 norm: {result['L1_norm']:.6f} (limit: {L1_limit})")
```

### 4.4 Test Case 3: Factor Exposure Limits

```python
def test_factor_exposure_limits():
    """Test MVO with bounded factor exposures."""
    n, k = 15, 4
    data = generate_synthetic_data(n=n, k=k, seed=456)
    
    F = generate_psd_matrix_with_condition_number(k, 50.0, seed=456)
    D = np.abs(np.random.randn(n)) * 0.015
    
    # Bound first factor between -0.5 and 0.5
    factor_limits = {
        'bounds': [(-0.5, 0.5), (None, None), (-1.0, 1.0), (None, 2.0)]
    }
    
    result = solve_mvo_factor_auxiliary_mosek(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=1.0,
        factor_exposure_limits=factor_limits
    )
    
    assert result is not None
    y = result['factor_exposures']
    assert -0.5 - 1e-6 <= y[0] <= 0.5 + 1e-6, f"Factor 0 out of bounds: {y[0]}"
    assert -1.0 - 1e-6 <= y[2] <= 1.0 + 1e-6, f"Factor 2 out of bounds: {y[2]}"
    assert y[3] <= 2.0 + 1e-6, f"Factor 3 out of bounds: {y[3]}"
```

### 4.5 Test Case 4: Transaction Cost Penalty

```python
def test_transaction_cost():
    """Test MVO with transaction cost relative to drift portfolio."""
    n, k = 25, 6
    data = generate_synthetic_data(n=n, k=k, seed=789)
    
    F = generate_psd_matrix_with_condition_number(k, 100.0, seed=789)
    D = np.abs(np.random.randn(n)) * 0.01
    
    # Random drift portfolio
    w_drift = np.random.dirichlet(np.ones(n))
    gamma = 0.01  # Transaction cost coefficient
    
    result = solve_mvo_factor_auxiliary_mosek(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=0.5,
        w_drift=w_drift,
        gamma=gamma
    )
    
    assert result is not None
    assert result['transaction_cost'] is not None
    assert result['transaction_cost'] >= 0
    print(f"Transaction cost: {result['transaction_cost']:.6f}")
```

### 4.6 Test Case 5: Verify Against Known Solution (Critical)

```python
def test_known_solution():
    """
    Verify optimizer recovers a known optimal solution.
    With alpha = Sigma @ w_raw and lambda_risk = 0.5,
    the optimal w* should equal w_raw.
    """
    n = 100
    np.random.seed(123)
    
    # Generate random target weights
    w_raw = np.random.randn(n)
    w_raw = w_raw / np.sum(np.abs(w_raw))
    
    # Generate covariance matrix
    target_cond = 1000.0
    Sigma = generate_psd_matrix_with_condition_number(n, target_cond, seed=123)
    
    # Compute alpha such that w_raw is optimal at lambda_risk=0.5
    alpha_inverse = Sigma @ w_raw
    
    # Solve with B=I, F=Sigma, D=0
    result = solve_mvo_factor_auxiliary_mosek(
        alpha=alpha_inverse,
        B=np.eye(n),
        F=Sigma,
        D=np.zeros(n),
        lambda_risk=0.5
    )
    
    assert result is not None
    max_diff = np.max(np.abs(result['weights'] - w_raw))
    print(f"Max difference from known solution: {max_diff:.6e}")
    assert np.allclose(result['weights'], w_raw, atol=1e-5), \
        f"Solution differs from expected: max_diff={max_diff}"
```

### 4.7 Test Case 6: Compare with CPLEX (if available)

```python
@pytest.mark.skipif(not CPLEX_AVAILABLE, reason="CPLEX not installed")
def test_compare_cplex_mosek():
    """Compare Mosek and CPLEX solutions for equivalence."""
    n, k = 50, 7
    data = generate_synthetic_data(n=n, k=k, seed=999)
    
    F = generate_psd_matrix_with_condition_number(k, 100.0, seed=999)
    D = np.abs(np.random.randn(n)) * 0.01
    
    # Solve with both solvers
    result_cplex = solve_mvo_factor_auxiliary(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=1.0,
        L1_limit=2.0
    )
    
    result_mosek = solve_mvo_factor_auxiliary_mosek(
        alpha=data['alpha'],
        B=data['B'],
        F=F,
        D=D,
        lambda_risk=1.0,
        L1_limit=2.0
    )
    
    assert result_cplex is not None and result_mosek is not None
    
    # Compare weights
    weight_diff = np.max(np.abs(result_cplex['weights'] - result_mosek['weights']))
    obj_diff = abs(result_cplex['objective'] - result_mosek['objective'])
    
    print(f"Weight max diff: {weight_diff:.6e}")
    print(f"Objective diff: {obj_diff:.6e}")
    
    assert weight_diff < 1e-4, f"Weight solutions differ: {weight_diff}"
    assert obj_diff < 1e-4, f"Objectives differ: {obj_diff}"
```

---

## 5. Running Tests

### 5.1 Activate Environment and Run

```bash
# Activate environment
conda activate mk_opt_local

# Run all Mosek tests
cd /home/wrk1231/projects/cplex_test
pytest test_optimizer_mosek.py -v

# Run specific test
pytest test_optimizer_mosek.py::test_known_solution -v

# Run with comparison to CPLEX (if installed)
pytest test_optimizer_mosek.py::test_compare_cplex_mosek -v
```

### 5.2 Run Main Test Script Interactively

```bash
# Run the main function from test_optimizer_mosek.py
python -c "from test_optimizer_mosek import *; test_known_solution()"
```

---

## 6. Key Differences and Gotchas

| Aspect | CPLEX | Mosek Fusion |
|--------|-------|--------------|
| **Quadratic terms** | `set_quadratic_coefficients()` | Conic reformulation or `Expr.quadForm()` |
| **Context management** | No `with` statement needed | Use `with Model() as M:` pattern |
| **Infinity** | `cplex.infinity` | `float('inf')` or omit bounds |
| **Sparse constraints** | `SparsePair(ind, val)` | `Expr.sub()`, `Expr.add()`, `Expr.mul()` |
| **Variable access** | Index by position | Use variable objects directly |
| **Solution extraction** | `get_values()` | `var.level()` |

### 6.1 Critical: Rotated Quadratic Cone Format

The rotated quadratic cone in Mosek expects: `(t, s, x1, x2, ..., xn)` where `2*t*s >= x1^2 + ... + xn^2`

For our case where we want `t >= 0.5 * ||x||^2`, we set `s = 0.5`:
- Stack: `(t, 0.5, x1, x2, ..., xn)`
- This gives: `2*t*0.5 >= ||x||^2` => `t >= 0.5*||x||^2`

### 6.2 Matrix Operations in Mosek

```python
# Dense matrix multiplication
Matrix.dense(A)  # Convert numpy array to Mosek matrix

# Element-wise multiplication
Expr.mulElm(vec, var)  # vec * var element-wise

# Matrix-vector multiplication  
Expr.mul(A, x)  # A @ x
```

---

## 7. File Summary

| File | Description |
|------|-------------|
| `optimizer_mosek.py` | Mosek Fusion implementation of `solve_mvo_factor_auxiliary_mosek()` |
| `test_optimizer_mosek.py` | Test cases validating Mosek implementation |
| `optimizer.py` | Original CPLEX implementation (reference) |
| `test_optimizer.py` | Original test script with helper functions |

---

## 8. Verification Checklist

- [ ] Conda environment `mk_opt_local` created with Python 3.12
- [ ] Mosek installed and licensed
- [ ] `optimizer_mosek.py` created with full implementation
- [ ] Test case 1: Basic MVO passes
- [ ] Test case 2: L1 constraint passes
- [ ] Test case 3: Factor exposure limits pass
- [ ] Test case 4: Transaction cost passes
- [ ] Test case 5: Known solution verification passes
- [ ] Test case 6: CPLEX comparison passes (if CPLEX available)
- [ ] All edge cases handled (zero gamma, None values, etc.)
