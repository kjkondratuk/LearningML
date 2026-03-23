# Module 06: Capstone -- Reusable Math Library

> Package everything from Phase 2A into a clean, tested, reusable library.
> Then demonstrate: solve a linear regression problem using ONLY your mathlib
> functions. No numpy.linalg, no scipy, no sklearn.

## Requirements

1. **`mathlib.py`** must include clean implementations of:
   - Matrix decompositions: LU, QR, SVD, eigendecomposition
   - Gradient computation: numerical gradient, Jacobian, Hessian
   - Probability distributions: Gaussian PDF, MLE, MAP
   - Optimization: gradient descent, Adam, L-BFGS step
   - Information theory: entropy, KL divergence

2. Every function must have:
   - Type hints
   - Docstring with the mathematical formula in LaTeX-style notation
   - Consistent API (NumPy arrays in, NumPy arrays out)

3. Comprehensive test suite that runs in under 30 seconds.

4. Demonstration notebook showing: solve a linear regression problem end-to-end
   using ONLY mathlib functions.

## How to Approach This

1. Copy your best implementations from Modules 01--05.
2. Refactor for a consistent API -- make function signatures uniform.
3. Add any missing edge-case handling you discovered during testing.
4. Write the demo notebook: generate data, form the design matrix, solve
   via SVD pseudoinverse using YOUR mathlib, compare to np.linalg.lstsq.

## Acceptance Criteria

- `pytest spiral_2/2a_math_foundations/06_capstone_math_library/tests/` passes
- All tests complete in under 30 seconds
- The demo notebook runs without errors and produces correct results
