# Module 08: Capstone -- ML Library From Scratch

> Build a sklearn-like library containing clean implementations of every classical
> ML algorithm from Phase 2B. Every class follows `fit(X, y)`, `predict(X)`,
> `score(X, y)`. No scikit-learn in the implementation -- only NumPy and your
> Phase 2A mathlib.

## Requirements

1. **`ml_library.py`** must contain:
   - `LinearRegression` (OLS, GD, and SVD solvers)
   - `RidgeRegression`
   - `LogisticRegression` (binary, Newton and GD solvers)
   - `SoftmaxRegression` (multi-class)
   - `DecisionTree` (classification and regression)
   - `RandomForest`
   - `GradientBoosting`
   - `SVM` (linear)
   - `KernelSVM` (RBF, polynomial)
   - `PCA`

2. Consistent API: `fit(X, y)`, `predict(X)`, `score(X, y)`.

3. Comprehensive test suite comparing each against scikit-learn on 3 datasets.

4. Demo notebook: pick a real dataset, run ALL models, compare accuracy, time,
   and interpretability.

## Acceptance Criteria

- All tests pass
- Each model's accuracy is within 5% of sklearn's on the same data
- Demo notebook runs without errors
