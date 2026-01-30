# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This appears to be a new Python project focused on optimization (likely using CPLEX or similar optimization libraries) based on the `optimizer.py` filename. The project is currently in its initial setup phase with minimal structure.

## Current State

The repository currently contains:
- `optimizer.py`: Main Python module (currently empty)

## Development Setup

Since this is a new Python project, typical development setup would include:

1. **Virtual Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Dependency Management**: Consider using `requirements.txt` or `pyproject.toml` for Python dependencies.

3. **Project Structure**: As this grows, typical structure might include:
   - `src/` for source code
   - `tests/` for test files
   - `data/` for input/output data
   - `docs/` for documentation

## Common Tasks

Since this is a new project, common development tasks will emerge as the codebase grows. Based on the filename, this project likely involves:
- Mathematical optimization problems
- Linear/integer programming
- Constraint programming
- CPLEX or similar optimization library integration

## Recommendations for Development

1. **Add a requirements.txt** to track Python dependencies
2. **Create a README.md** with project description and setup instructions
3. **Consider adding tests** using pytest or unittest
4. **Add a .gitignore** file for Python projects
5. **Document the optimization problems** this solver addresses

## Python Environment

When working with this codebase, ensure Python is available and consider using virtual environments to manage dependencies.