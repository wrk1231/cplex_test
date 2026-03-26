# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project (no `src/` package yet). Core solver modules and test scripts live at the repo root:

- `optimizer.py`: baseline MVO/factor auxiliary solver plus soft L1 variant.
- `optimizer_soft_constraint.py`: soft-constraint solver (legacy transaction-cost formulation).
- `optimizer_soft_constraint_regular_to.py`: updated soft-constraint solver with improved drift-cost behavior and infinite-bound handling.
- `fmp_optimization.py`: factor-mimicking portfolio (FMP) solver and benchmark helper.
- `example_full_usage.py`: end-to-end usage example.
- `test_*.py`: runnable validation scripts.
- `OPTIMIZATION_MATH.md`: math notes and formulation details.

## Build, Test, and Development Commands
There is no build system or packaging config; run modules directly with Python.

- `python example_full_usage.py`: run a full optimization example with all major parameters.
- `python test_no_drift.py`: validate `w_drift=None` behavior across gamma settings.
- `python test_inf_bounds.py`: validate mixed finite/infinite factor bounds.
- `python test_optimizer_soft_constraint.py`: exercise soft-constraint penalty behavior.
- `python test_optimizer_soft_constraint_regular_to.py`: verify regularized transaction-cost variant.
- `python test_optimizer.py`: larger synthetic stress-style run.

Prerequisites: `python` (3.11+), `numpy`, and IBM `cplex` available in the active environment.

## Coding Style & Naming Conventions
- Follow existing Python style: 4-space indentation, `snake_case`, and clear function names.
- Keep interfaces NumPy-first (`np.ndarray` inputs/outputs) and validate dimensions early with explicit `ValueError`s.
- Preserve CPLEX variable naming patterns (`w_i`, `y_j`, `z_i`, `d_i`) for model readability.
- Keep comments focused on optimization intent (constraints/objective terms), not obvious Python mechanics.

## Testing Guidelines
- Tests are currently script-based (not a configured `pytest` suite).
- Add new checks as `test_*.py` files that run via `python <file>`.
- Use deterministic seeds and verify solver status plus key metrics (bounds, penalties, transaction cost).
- For new coverage, prefer `assert` checks so failures are machine-detectable.

## Commit & Pull Request Guidelines
- Mirror current history style: concise, imperative commit subjects (for example, `Add ...`, `Support ...`).
- Keep commits focused: one behavior change plus related tests/docs.
- In PRs, include: purpose, files changed, commands executed, and notable output/solver assumptions.
