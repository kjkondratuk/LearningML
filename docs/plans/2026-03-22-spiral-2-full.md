The plan file is the one exception I can write to. Let me produce it now -- this will be substantial.

I'll present the complete plan below rather than writing to a file, since the system reminders indicate read-only mode. Here is the full Spiral 2 scaffolding design:

---

# Spiral 2: Depth and Rigor -- Complete Module Scaffolding Plan

## Overarching Design Principles for Spiral 2

**The "Naive then Derive" Pattern:** Spiral 1's "do it the wrong way" challenges become Spiral 2's central pedagogy. Every module follows this arc: (1) implement the naive/brute-force version, (2) identify what is slow or mathematically unsound, (3) derive the efficient/correct version from first principles, (4) verify both produce the same results (within numerical tolerance).

**No scikit-learn.** Every algorithm is implemented from scratch using only NumPy (and PyTorch for Phase 2C). The learner may use scikit-learn ONLY as a reference oracle to verify correctness of their own implementation.

**Paper-and-code derivations.** Each module README includes a "Derive It" section where the learner works through the math on paper (or in LaTeX markdown) before coding. The exercises.py stubs include docstrings that state the mathematical result to derive and implement.

**Testing strategy.** Tests validate against known analytical solutions, hand-computed examples, and (where appropriate) scikit-learn/PyTorch reference implementations with `np.allclose` tolerances.

---

## Dependencies (added to pyproject.toml)

```
# Spiral 2 additions beyond Spiral 1:
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "jupyter",
    "ipykernel",
    "scikit-learn",   # reference oracle only
    "torch",          # Phase 2C onward
    "mlflow",         # Phase 2D
    "optuna",         # Phase 2D (Bayesian hyperparameter tuning)
    "tqdm",           # progress bars
]
```

---

## Phase 2A: Mathematical Foundations (Dedicated Pass)

**Goal:** Build the mathematical toolkit that underpins all of ML. Every concept is implemented in NumPy. The learner should finish this phase able to read matrix calculus notation in papers without flinching.

### Directory Structure

```
spiral_2/2a_math_foundations/
├── README.md
├── 01_linear_algebra/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Image compression via SVD
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 02_calculus_and_optimization/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Gradient descent visualizer
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 03_probability_and_statistics/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Bayesian coin-flip inference
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 04_optimization_methods/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Optimizer shootout on Rosenbrock + saddle functions
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 05_information_theory/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
└── 06_capstone_math_library/
    ├── README.md
    ├── mathlib.py               # Learner builds a reusable math library
    ├── notebook.ipynb           # Demonstration notebook
    └── tests/
        └── test_mathlib.py
```

### Module 01: Linear Algebra

**Math Resources:**
- 3Blue1Brown "Essence of Linear Algebra" full series (chapters 1-15)
- Strang, "Introduction to Linear Algebra" chapters 1-4, 6-7
- Bishop PRML Appendix C (Matrix Identities)

**Exercise Stubs:**
- `matrix_multiply(A, B)` -- Implement matrix multiplication from nested loops, then derive the vectorized version. Compare timing on 500x500 matrices.
- `lu_decomposition(A)` -- Decompose A into lower and upper triangular matrices. Verify A = LU.
- `qr_decomposition(A)` -- Implement Gram-Schmidt orthogonalization to produce Q, R. Verify Q is orthogonal, R is upper triangular, A = QR.
- `eigendecomposition(A)` -- Implement the power iteration method to find the dominant eigenvalue/eigenvector. Then implement QR algorithm for full decomposition.
- `svd(A)` -- Derive SVD from eigendecomposition of A^T A and A A^T. Implement it. Verify against np.linalg.svd.
- `solve_linear_system(A, b)` -- Solve Ax = b using LU decomposition (not np.linalg.solve).
- `compute_pseudoinverse(A)` -- Compute the Moore-Penrose pseudoinverse via SVD. This is the key to least-squares regression.
- `is_positive_definite(A)` -- Check via Cholesky decomposition attempt. Explain why PD matrices matter for ML (covariance matrices, kernel matrices).

**"Naive then Derive" Challenge:** Implement matrix_multiply with three nested for-loops. Time it on 500x500. Then implement the same thing with `np.dot`. The difference demonstrates why NumPy exists.

**Test Scenarios:**
- All decompositions verified by reconstruction (A = LU, A = QR, A = U S V^T) within tolerance 1e-10
- Eigenvalues/vectors verified against np.linalg.eig
- Known 2x2 and 3x3 hand-computed examples
- Edge cases: singular matrices, rectangular matrices for SVD, non-square for QR

**Mini-Project: Image Compression via SVD**
- Load a grayscale image as a matrix
- Compute SVD
- Reconstruct using top-k singular values for k = 1, 5, 10, 20, 50, 100
- Plot reconstruction quality (Frobenius norm error) vs k
- Plot compressed size vs k
- Find the "elbow" -- minimum k for visually acceptable quality

### Module 02: Calculus and Optimization Foundations

**Math Resources:**
- 3Blue1Brown "Essence of Calculus" series
- 3Blue1Brown "Gradient descent, how neural networks learn" (Chapter 2 of neural networks series)
- Boyd & Vandenberghe, "Convex Optimization" chapter 1-2 (free PDF)
- Murphy MLAPP chapter 8.1-8.3 (Optimization)

**Exercise Stubs:**
- `numerical_gradient(f, x, epsilon)` -- Compute gradient of f at x using central differences. Works for scalar and vector inputs.
- `numerical_jacobian(f, x, epsilon)` -- Compute the Jacobian matrix of vector-valued f at x.
- `numerical_hessian(f, x, epsilon)` -- Compute the Hessian matrix (second derivatives) of scalar f at x.
- `taylor_approximate(f, x0, order, points)` -- Compute 1st and 2nd order Taylor approximations of f around x0. Return values at given points.
- `gradient_descent(f, grad_f, x0, lr, n_steps)` -- Basic gradient descent. Return trajectory of all x values for visualization.
- `gradient_descent_with_momentum(f, grad_f, x0, lr, momentum, n_steps)` -- Add momentum term. Return trajectory.
- `check_convexity(f, x_range, n_samples)` -- Numerically verify convexity by checking f(tx + (1-t)y) <= t f(x) + (1-t) f(y) for random pairs.
- `lagrange_dual(f, g, x_range)` -- For a simple constrained problem min f(x) s.t. g(x) = 0, find the solution via Lagrange multipliers. Implement for 2D case.

**"Naive then Derive" Challenge:** Implement gradient descent with a fixed learning rate. Watch it oscillate on a poorly conditioned quadratic. Then derive why momentum helps by analyzing the eigenvalues of the Hessian. Implement momentum and see the difference.

**Test Scenarios:**
- Numerical gradient matches analytical gradient for f(x) = x^2, f(x) = sin(x), f(x,y) = x^2 + y^2
- Hessian is symmetric for all test functions
- Taylor approximation error decreases with order
- Gradient descent converges to known minima (quadratic, Rosenbrock)
- Lagrange multiplier solution matches known analytical solution

**Mini-Project: Gradient Descent Visualizer**
- Create contour plots for 3 functions: quadratic bowl, Rosenbrock banana, saddle point
- Overlay GD trajectories with different learning rates
- Overlay GD with momentum vs without
- Animate the optimization path (matplotlib animation or frame sequence)
- Produce a "learning rate too high" divergence visualization

### Module 03: Probability and Statistics

**Math Resources:**
- 3Blue1Brown "Bayes theorem" and "But what is a probability distribution?"
- Bishop PRML chapters 1.2 (Probability Theory), 2.1-2.4 (Probability Distributions)
- Murphy MLAPP chapters 2 (Probability), 3 (Generative Models for Discrete Data), 4 (Gaussian Models)

**Exercise Stubs:**
- `gaussian_pdf(x, mu, sigma)` -- Implement the Gaussian PDF from the formula. Verify against scipy.stats.norm.pdf.
- `multivariate_gaussian_pdf(x, mu, cov)` -- Implement the multivariate Gaussian PDF. Requires matrix inverse and determinant.
- `mle_gaussian(data)` -- Derive and implement MLE estimates for mu and sigma of a Gaussian. Show that MLE for mu is the sample mean and for sigma^2 is the (biased) sample variance.
- `map_gaussian(data, prior_mu, prior_sigma, likelihood_sigma)` -- Derive and implement MAP estimate for mu with a Gaussian prior. Show how the prior "pulls" the estimate.
- `bayesian_update(prior, likelihood, evidence_grid)` -- Implement Bayes' rule on a discrete grid. Visualize prior, likelihood, and posterior.
- `conjugate_prior_beta_binomial(alpha, beta, n_successes, n_trials)` -- Derive the posterior Beta distribution. Return updated alpha, beta.
- `sample_from_distribution(pdf, x_range, n_samples)` -- Implement rejection sampling. Compare against the true distribution.
- `central_limit_theorem_demo(distribution, sample_sizes, n_experiments)` -- Demonstrate CLT by sampling means from non-Gaussian distributions.

**"Naive then Derive" Challenge:** Compute the MLE of a Gaussian by brute-force grid search over (mu, sigma) pairs, evaluating the log-likelihood at each. Then derive the closed-form solution by taking the derivative, setting it to zero, and solving. Verify both give the same answer. The brute-force version takes seconds; the analytical version is instant.

**Test Scenarios:**
- Gaussian PDF integrates to 1.0 (numerical integration)
- MLE estimates match np.mean and np.var for known data
- MAP estimate is between the prior mean and the MLE (shrinkage)
- Beta-Binomial posterior has correct alpha, beta
- Rejection sampling produces distribution matching the target (KS test)
- CLT demo shows convergence to Gaussian

**Mini-Project: Bayesian Coin Flip Inference**
- Start with a Beta(1,1) prior (uniform)
- Observe coin flips one at a time
- After each flip, compute and plot the posterior
- Animate the posterior evolving from uniform to peaked
- Compare Bayesian posterior mean vs MLE vs MAP
- Show how a strong prior (Beta(100,100)) resists evidence from a biased coin

### Module 04: Optimization Methods

**Math Resources:**
- Ruder, "An overview of gradient descent optimization algorithms" (2016 survey paper -- free arxiv)
- Boyd & Vandenberghe chapters 9-10 (Unconstrained and Constrained Optimization)
- Goodfellow "Deep Learning" textbook chapter 8 (Optimization for Training)

**Exercise Stubs:**
- `sgd(grad_f, x0, lr, n_steps, batch_fn)` -- Stochastic gradient descent with mini-batch gradient function.
- `sgd_momentum(grad_f, x0, lr, momentum, n_steps, batch_fn)` -- SGD with classical momentum.
- `nesterov_momentum(grad_f, x0, lr, momentum, n_steps, batch_fn)` -- Nesterov accelerated gradient. Derive the "look-ahead" step.
- `adagrad(grad_f, x0, lr, n_steps, batch_fn)` -- Adaptive learning rate per parameter. Derive from the idea of "parameters that rarely update should have larger learning rates."
- `rmsprop(grad_f, x0, lr, decay, n_steps, batch_fn)` -- Fix AdaGrad's monotonically decreasing learning rate with exponential moving average.
- `adam(grad_f, x0, lr, beta1, beta2, n_steps, batch_fn)` -- Derive Adam as combination of momentum + RMSProp + bias correction.
- `lbfgs_step(grad_f, x, history, m)` -- Implement the L-BFGS two-loop recursion for one step. Explain why second-order methods are impractical for deep learning but great for convex problems.
- `line_search(f, grad_f, x, direction)` -- Backtracking line search with Armijo condition.

**"Naive then Derive" Challenge:** Train a linear regression on a dataset with features at very different scales (one feature in [0,1], another in [0,100000]). Use vanilla SGD -- watch it zigzag. Then derive why Adam's per-parameter learning rate fixes this. Implement Adam and compare convergence.

**Test Scenarios:**
- All optimizers converge to the minimum of a convex quadratic
- Adam converges faster than vanilla SGD on ill-conditioned problems
- AdaGrad's effective learning rate decreases monotonically (verify numerically)
- Nesterov is faster than classical momentum on a quadratic (count steps to convergence)
- L-BFGS converges in far fewer iterations than first-order methods on a small logistic regression

**Mini-Project: Optimizer Shootout**
- Implement all optimizers with a common interface
- Test on 4 surfaces: convex quadratic, Rosenbrock, Rastrigin (non-convex), logistic regression loss
- For each: plot convergence curves (loss vs iteration), plot trajectories on contour plots
- Produce a comparison table: steps to convergence, final loss, wall-clock time
- Write a 1-paragraph analysis: when would you use each optimizer?

### Module 05: Information Theory

**Math Resources:**
- Cover & Thomas, "Elements of Information Theory" chapters 1-2
- Bishop PRML section 1.6 (Information Theory)
- Murphy MLAPP chapter 2.8 (Information Theory)
- Colah's blog post "Visual Information Theory"

**Exercise Stubs:**
- `entropy(p)` -- Compute Shannon entropy of a discrete distribution. Handle p=0 case.
- `cross_entropy(p, q)` -- Compute cross-entropy. Explain why this is the loss function for classification.
- `kl_divergence(p, q)` -- Compute KL divergence. Show it is not symmetric. Explain connection to cross-entropy loss.
- `mutual_information(joint_p)` -- Compute mutual information from a joint distribution table.
- `differential_entropy_gaussian(sigma)` -- Derive the differential entropy of a Gaussian analytically. Implement and verify.
- `kl_divergence_gaussians(mu1, sigma1, mu2, sigma2)` -- Derive and implement KL divergence between two univariate Gaussians in closed form.

**"Naive then Derive" Challenge:** Estimate entropy by generating samples and using the plug-in estimator (count frequencies, compute entropy). Then compare to the true entropy. Show the bias of the plug-in estimator for small sample sizes.

**Test Scenarios:**
- Entropy of a fair coin is 1.0 bit
- Entropy of a uniform distribution over n items is log2(n)
- KL(p||q) >= 0 always (verify for 100 random pairs)
- KL(p||q) = 0 iff p = q
- Cross-entropy(p, p) = entropy(p)
- Mutual information is symmetric

### Module 06: Capstone -- Reusable Math Library

**Requirements:**
- Package all Phase 2A implementations into a clean `mathlib.py` with a coherent API
- Must include: matrix decompositions (LU, QR, SVD, eigendecomposition), gradient computation (numerical gradient, Jacobian, Hessian), probability distributions (Gaussian PDF, MLE, MAP), optimization (GD, Adam, L-BFGS step), information theory (entropy, KL divergence)
- Each function must have type hints and a docstring with the mathematical formula in LaTeX-style notation
- Comprehensive test suite that runs in under 30 seconds
- A demonstration notebook showing: solve a linear regression problem using ONLY mathlib functions (no numpy.linalg, no scipy, no sklearn)

---

## Phase 2B: Classical ML (Theory Pass)

**Goal:** Derive and implement classical ML algorithms from scratch. The learner should finish this phase able to open Bishop PRML to any chapter on classical methods and follow the derivations.

### Directory Structure

```
spiral_2/2b_classical_ml_theory/
├── README.md
├── 01_linear_regression/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Bayesian linear regression with uncertainty
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 02_logistic_regression/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Multi-class from scratch on MNIST subset
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 03_regularization/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 04_bias_variance/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 05_svms/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Kernel SVM on non-linear data
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 06_decision_trees_and_ensembles/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Random forest from scratch
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 07_dimensionality_reduction/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # PCA + t-SNE on fashion-MNIST
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
└── 08_capstone_ml_from_scratch/
    ├── README.md
    ├── ml_library.py
    ├── notebook.ipynb
    └── tests/
        └── test_ml_library.py
```

### Module 01: Linear Regression -- From MLE to MAP

**Math Resources:**
- Bishop PRML sections 3.1 (Linear Basis Function Models), 3.3 (Bayesian Linear Regression)
- Murphy MLAPP chapter 7 (Linear Regression)
- 3Blue1Brown "Linear Regression" (if available) or StatQuest "Linear Regression" series

**Exercise Stubs:**
- `ols_closed_form(X, y)` -- Derive the normal equations from minimizing RSS. Implement w = (X^T X)^{-1} X^T y. Explain when this breaks (singular X^T X).
- `ols_via_gradient_descent(X, y, lr, n_steps)` -- Derive the gradient of MSE loss. Implement batch GD. Show it converges to the same solution as closed form.
- `ols_via_svd(X, y)` -- Solve using SVD pseudoinverse. Explain why this is numerically more stable than the normal equations.
- `mle_linear_regression(X, y)` -- Derive that MLE under Gaussian noise is equivalent to OLS. Implement and show the equivalence.
- `map_linear_regression(X, y, prior_precision)` -- Derive MAP estimate with Gaussian prior on weights. Show this is ridge regression. Implement.
- `bayesian_linear_regression(X, y, prior_precision, noise_precision)` -- Implement full Bayesian posterior over weights. Return posterior mean, posterior covariance.
- `predictive_distribution(X_test, posterior_mean, posterior_cov, noise_precision)` -- Compute the predictive distribution (mean and variance) at test points. Show uncertainty grows far from training data.
- `polynomial_regression(X, y, degree)` -- Implement polynomial basis expansion + OLS. Show overfitting at high degree.

**"Naive then Derive" Challenge:** Fit a polynomial of degree 15 to 20 data points. Watch it overfit wildly. Then derive why adding a Gaussian prior on weights (MAP = Ridge) fixes the overfitting. Show the regularization path as prior_precision varies.

**Test Scenarios:**
- OLS closed form, GD, and SVD all produce same weights (within tolerance)
- MLE solution matches OLS solution exactly
- MAP solution approaches OLS as prior_precision approaches 0
- Bayesian predictive uncertainty increases away from training data
- Polynomial regression with degree=1 matches OLS

**Mini-Project: Bayesian Linear Regression with Uncertainty**
- Generate synthetic data from a known linear function + noise
- Fit Bayesian linear regression
- Plot: data points, posterior mean prediction, 1-sigma and 2-sigma uncertainty bands
- Show sequential Bayesian updating: fit to 5 points, then 10, then 20, then 100. Show posterior sharpening.
- Compare Bayesian uncertainty to bootstrap uncertainty estimates

### Module 02: Logistic Regression -- The Generalized Linear Model

**Math Resources:**
- Bishop PRML sections 4.1-4.3 (Discriminative Models)
- Murphy MLAPP chapter 8.3-8.4 (Logistic Regression)
- Andrew Ng CS229 lecture notes on logistic regression (free online)

**Exercise Stubs:**
- `sigmoid(z)` -- Implement the sigmoid function. Derive it as the canonical link function for Bernoulli distribution.
- `log_likelihood_binary(X, y, w)` -- Derive and implement the log-likelihood for binary logistic regression.
- `gradient_log_likelihood(X, y, w)` -- Derive the gradient. Show it has the elegant form X^T (y - sigmoid(Xw)).
- `hessian_log_likelihood(X, y, w)` -- Derive the Hessian. Show it is negative semi-definite (log-likelihood is concave).
- `logistic_regression_gd(X, y, lr, n_steps)` -- Fit via gradient ascent on log-likelihood.
- `logistic_regression_newton(X, y, n_steps)` -- Fit via Newton-Raphson (IRLS). Derive the update rule. Show faster convergence than GD.
- `softmax(z)` -- Implement numerically stable softmax. Derive as generalization of sigmoid to K classes.
- `multinomial_logistic_regression(X, y, n_classes, lr, n_steps)` -- Multi-class logistic regression via cross-entropy loss and gradient descent.
- `decision_boundary(w, b, x_range)` -- Compute and return the linear decision boundary for visualization.

**"Naive then Derive" Challenge:** Implement logistic regression with basic gradient descent. Then derive the Newton-Raphson update (IRLS) using the Hessian. Compare: Newton converges in 5-10 iterations; GD takes hundreds. Show this on a 2D classification dataset.

**Test Scenarios:**
- Sigmoid(0) = 0.5, sigmoid maps R to (0,1)
- Log-likelihood is concave (Hessian is NSD at random points)
- GD and Newton converge to same weights
- Newton converges in fewer iterations than GD
- Softmax outputs sum to 1.0
- Multinomial LR achieves >90% on linearly separable synthetic data
- Decision boundary correctly separates linearly separable data

**Mini-Project: Multi-class Classification on MNIST Subset**
- Load MNIST digits 0-4 (5 classes, smaller for from-scratch speed)
- Flatten 28x28 images to 784-dim vectors
- Implement multinomial logistic regression from scratch
- Train with mini-batch SGD
- Plot: training loss curve, confusion matrix, misclassified examples
- Compare accuracy against sklearn's LogisticRegression

### Module 03: Regularization Theory

**Math Resources:**
- Bishop PRML section 3.1.4 (Regularized Least Squares)
- Murphy MLAPP section 7.5 (Ridge Regression), 13.3 (L1 Regularization)
- Hastie, Tibshirani, Friedman "Elements of Statistical Learning" chapter 3.4 (free PDF)

**Exercise Stubs:**
- `ridge_regression(X, y, alpha)` -- Derive as MAP with Gaussian prior. Implement closed form: w = (X^T X + alpha I)^{-1} X^T y.
- `lasso_coordinate_descent(X, y, alpha, n_steps)` -- Implement coordinate descent for L1. Derive the soft-thresholding operator.
- `elastic_net(X, y, alpha, l1_ratio, n_steps)` -- Combine L1 and L2. Implement.
- `regularization_path(X, y, alphas)` -- For each alpha, fit ridge and lasso. Return weight trajectories. Plot weights vs log(alpha).
- `l1_sparsity_demo(X, y, alpha)` -- Generate data with 5 true features and 95 noise features. Show L1 recovers the true features.
- `cross_validate_alpha(X, y, alphas, k_folds, model_fn)` -- Implement k-fold CV to select regularization strength.

**"Naive then Derive" Challenge:** Visualize the L1 vs L2 constraint regions in 2D weight space. Draw the contours of the unregularized loss. Show geometrically why L1 hits the axes (sparsity) and L2 does not. Then implement both and verify on synthetic data with irrelevant features.

**Test Scenarios:**
- Ridge solution approaches OLS as alpha approaches 0
- Lasso produces exact zeros in weight vector for sufficiently large alpha
- Elastic net with l1_ratio=0 matches ridge; l1_ratio=1 matches lasso
- Cross-validation selects a reasonable alpha (not 0, not infinity)
- L1 sparsity demo: lasso zeros out at least 80% of the 95 noise features

### Module 04: Bias-Variance Tradeoff

**Math Resources:**
- Bishop PRML section 3.2 (The Bias-Variance Decomposition)
- Murphy MLAPP section 6.4 (Bias-Variance Tradeoff)
- Hastie ESL chapter 7.2-7.3

**Exercise Stubs:**
- `bias_variance_decomposition(model_fn, X_train_sets, X_test, y_test_true)` -- Given many training sets from the same distribution, compute bias^2, variance, and irreducible error for a model. This is the CORE exercise.
- `polynomial_bias_variance(degree, n_datasets, n_points, noise_std)` -- Generate n_datasets synthetic datasets, fit polynomial of given degree to each, decompose into bias and variance.
- `plot_bias_variance_vs_complexity(degrees, n_datasets, n_points, noise_std)` -- The classic U-shaped test error curve. Show bias decreasing, variance increasing, and their sum.
- `learning_curves(model_fn, X, y, train_sizes)` -- Plot training error and validation error vs training set size. Diagnose high bias vs high variance.
- `double_descent_demo(X, y, model_complexities)` -- For a sufficiently overparameterized model, demonstrate the double descent phenomenon where test error decreases again past the interpolation threshold.

**"Naive then Derive" Challenge:** The learner's intuition from Spiral 1 is "more complex = overfitting." Now derive the bias-variance decomposition formally, proving that E[error] = bias^2 + variance + noise. Then demonstrate double descent, which breaks the simple U-curve intuition.

**Test Scenarios:**
- For degree=1 on cubic data: high bias, low variance
- For degree=20 on cubic data: low bias, high variance
- bias^2 + variance + noise approximately equals mean test error
- Learning curves: high bias model has gap that does not close with more data

### Module 05: Support Vector Machines

**Math Resources:**
- Bishop PRML sections 7.1 (Maximum Margin Classifiers), 7.1.1 (Overlapping Distributions)
- Murphy MLAPP chapter 14 (Kernels)
- Boyd & Vandenberghe chapter 5 (Duality) for the dual formulation

**Exercise Stubs:**
- `hard_margin_svm_primal(X, y)` -- Formulate and solve the primal QP using a simple QP solver (or gradient projection). Derive the margin = 2/||w||.
- `hard_margin_svm_dual(X, y)` -- Derive the dual formulation via Lagrange multipliers. Solve for alphas. Recover w from support vectors.
- `soft_margin_svm(X, y, C)` -- Add slack variables. Derive the modified dual. Implement.
- `kernel_function(x1, x2, kernel_type, **params)` -- Implement linear, polynomial, and RBF kernels.
- `kernel_svm(X, y, C, kernel_fn)` -- Implement kernel SVM using the kernel trick in the dual formulation. The key insight: we never compute the high-dimensional feature map explicitly.
- `smo_simplified(X, y, C, kernel_fn, tol, max_passes)` -- Implement a simplified version of the Sequential Minimal Optimization algorithm for training SVMs.
- `predict_svm(X_test, X_support, y_support, alphas, b, kernel_fn)` -- Prediction using support vectors only.

**"Naive then Derive" Challenge:** Try to classify XOR-pattern data with a linear SVM. Fail. Then derive the kernel trick: show that the RBF kernel implicitly maps to infinite-dimensional space. Implement kernel SVM and classify XOR perfectly. Visualize the decision boundary in the original 2D space.

**Test Scenarios:**
- Hard margin SVM finds the maximum margin hyperplane on linearly separable data
- Primal and dual solutions produce same w and b
- Support vectors are the points closest to the decision boundary
- Kernel SVM with RBF classifies XOR data
- SMO converges and matches sklearn's SVC predictions
- Changing C trades off margin width vs misclassifications

**Mini-Project: Kernel SVM on Non-Linear Data**
- Generate three datasets: linearly separable, concentric circles, two moons
- Train linear SVM, polynomial kernel SVM, and RBF kernel SVM on each
- Visualize decision boundaries for all 9 combinations
- Plot support vectors highlighted on each decision boundary
- Tune C and gamma via cross-validation; show the effect of each on the boundary

### Module 06: Decision Trees and Ensembles

**Math Resources:**
- Bishop PRML section 14.4 (Tree-based models)
- Murphy MLAPP chapter 16 (Adaptive Basis Function Models) sections on trees, bagging, boosting
- Hastie ESL chapters 9 (Additive Models, Trees) and 10 (Boosting)
- Original Random Forest paper by Breiman (2001) -- introduction only

**Exercise Stubs:**
- `gini_impurity(y)` -- Compute Gini impurity. Derive from misclassification probability.
- `entropy_impurity(y)` -- Compute entropy-based impurity. Compare to Gini.
- `information_gain(y, y_left, y_right)` -- Compute IG for a split. Derive from parent entropy minus weighted child entropies.
- `find_best_split(X, y, feature_idx)` -- Find the threshold that maximizes IG for a given feature.
- `build_decision_tree(X, y, max_depth, min_samples)` -- Recursively build a decision tree. Return a tree structure (nested dicts or a Tree class).
- `predict_tree(tree, X)` -- Traverse the tree to predict.
- `prune_tree(tree, X_val, y_val)` -- Implement cost-complexity pruning. Derive the pruning criterion.
- `bagging_ensemble(X, y, n_trees, max_depth, max_features)` -- Implement bagging with random feature subsets (this IS a random forest).
- `gradient_boosting(X, y, n_rounds, lr, max_depth)` -- Implement gradient boosting for regression. Each tree fits the residuals. Derive why this reduces bias.
- `feature_importance(trees, n_features)` -- Compute feature importance from the ensemble.

**"Naive then Derive" Challenge:** Build a single deep decision tree. Watch it overfit (100% training accuracy, poor test accuracy). Then derive why bagging reduces variance: each tree is high-variance low-bias; averaging reduces variance without increasing bias. Implement random forest and see the improvement. Then derive why boosting reduces bias: each tree fits the residual error. Implement gradient boosting and see it outperform on bias-dominated problems.

**Test Scenarios:**
- Gini(uniform distribution) = 1 - 1/K for K classes
- Entropy(fair coin) = 1.0 bit
- Decision tree achieves 100% training accuracy (no depth limit, no min_samples)
- Pruned tree has better test accuracy than unpruned
- Random forest test accuracy > single tree test accuracy
- Gradient boosting test accuracy improves with n_rounds (then saturates)
- Feature importance identifies the true informative features in synthetic data

**Mini-Project: Random Forest From Scratch**
- Implement the full random forest (bagging + random feature subsets + decision trees)
- Train on the Iris dataset and a synthetic dataset with 100 features (10 informative)
- Compare against sklearn's RandomForestClassifier: accuracy, feature importances
- Plot OOB (out-of-bag) error vs number of trees
- Plot feature importance bar chart

### Module 07: Dimensionality Reduction

**Math Resources:**
- Bishop PRML section 12.1 (PCA)
- Murphy MLAPP chapter 12 (Latent Linear Models)
- Van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008) -- the original paper, focus on sections 1-3
- 3Blue1Brown "Eigenvectors and Eigenvalues" (for PCA intuition)

**Exercise Stubs:**
- `pca_via_eigendecomposition(X, n_components)` -- Center data, compute covariance matrix, eigendecompose, project. Return transformed data and explained variance ratios.
- `pca_via_svd(X, n_components)` -- Implement PCA via SVD (more numerically stable). Show equivalence to eigendecomposition approach.
- `reconstruct_from_pca(X_transformed, components, mean)` -- Reconstruct original data from PCA representation. Compute reconstruction error.
- `pca_explained_variance_plot(X, max_components)` -- Scree plot: cumulative explained variance vs number of components. Find the "elbow."
- `kernel_pca(X, n_components, kernel_fn)` -- Implement kernel PCA for non-linear dimensionality reduction. Use the kernel trick.
- `tsne(X, n_components, perplexity, lr, n_steps)` -- Implement t-SNE from scratch. This is the hardest exercise in this module: compute pairwise affinities in high-D (Gaussian), compute pairwise affinities in low-D (Student-t), minimize KL divergence via gradient descent.
- `pca_whitening(X)` -- Transform data to have identity covariance. Derive from PCA + rescaling by eigenvalues.

**"Naive then Derive" Challenge:** Run PCA on a dataset that has a non-linear manifold (e.g., swiss roll). Show that PCA fails miserably because it only finds linear subspaces. Then derive kernel PCA: show that by working in the kernel-induced feature space, PCA can capture non-linear structure.

**Test Scenarios:**
- PCA eigen and SVD approaches give same results
- Explained variance ratios sum to 1.0
- Reconstruction error decreases as n_components increases
- PCA of a 2D dataset along its principal axis gives a 1D representation with maximum variance
- Kernel PCA with RBF separates concentric circles
- t-SNE produces clusters that match known labels (visual check + k-means on output)
- Whitened data has identity covariance matrix

**Mini-Project: PCA + t-SNE on Fashion-MNIST**
- Load Fashion-MNIST (10 classes of clothing)
- Apply PCA: plot explained variance, visualize in 2D with class colors
- Apply t-SNE: visualize in 2D, compare to PCA
- Try PCA as preprocessing for t-SNE (PCA to 50 dims, then t-SNE to 2)
- Reconstruct images from PCA at 10, 50, 100, 200 components. Visualize reconstruction quality.
- Compare wall-clock time: PCA vs t-SNE

### Module 08: Capstone -- ML Library From Scratch

**Requirements:**
- Build `ml_library.py` containing clean implementations of: LinearRegression, RidgeRegression, LogisticRegression, SoftmaxRegression, DecisionTree, RandomForest, GradientBoosting, SVM (linear), KernelSVM, PCA
- Each class must follow a consistent API: `fit(X, y)`, `predict(X)`, `score(X, y)`
- No scikit-learn anywhere in the implementation. Use only NumPy and the Phase 2A mathlib.
- Comprehensive test suite that compares each implementation against scikit-learn on 3 datasets (one synthetic, one small real, one with irrelevant features)
- Demonstration notebook: pick a real dataset (e.g., Wisconsin Breast Cancer), run ALL models, compare accuracy, training time, and interpretability
- Write a 1-page analysis: which model would you choose for this dataset and why?

---

## Phase 2C: Deep Learning (Theory Pass)

**Goal:** Understand deep learning from the math up. Derive backpropagation, implement a neural network framework, understand why modern architectures work. Build a Transformer from scratch.

### Directory Structure

```
spiral_2/2c_deep_learning_theory/
├── README.md
├── 01_backpropagation/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Computational graph engine
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 02_activation_and_initialization/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 03_optimization_for_deep_learning/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 04_regularization_for_deep_learning/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Regularization ablation study
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 05_cnns/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # CNN from scratch in NumPy
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 06_rnns_and_lstms/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Character-level language model
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 07_attention_and_transformers/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Transformer from scratch
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
└── 08_capstone_transformer/
    ├── README.md
    ├── transformer.py
    ├── train.py
    ├── notebook.ipynb
    └── tests/
        └── test_transformer.py
```

### Module 01: Backpropagation Derived and Implemented

**Math Resources:**
- Goodfellow "Deep Learning" chapter 6.5 (Back-Propagation and Other Differentiation Algorithms)
- 3Blue1Brown "Backpropagation" and "Backpropagation calculus" (Neural Networks series chapters 3-4)
- Colah's blog "Calculus on Computational Graphs: Backpropagation"
- Karpathy "micrograd" video lecture (for the computational graph idea)

**Exercise Stubs:**
- `forward_pass(X, weights, biases, activations)` -- Implement a generic forward pass through L layers. Store all intermediate activations (needed for backprop).
- `backward_pass(X, y, weights, biases, activations, cached_activations)` -- Derive and implement backpropagation. For each layer, compute dL/dW and dL/db using the chain rule. Return all gradients.
- `numerical_gradient_check(f, params, epsilon)` -- Verify analytical gradients match numerical gradients. This is the gold standard for debugging backprop.
- `mse_loss_and_grad(y_pred, y_true)` -- Derive the gradient of MSE loss.
- `cross_entropy_loss_and_grad(y_pred, y_true)` -- Derive the gradient of cross-entropy loss. Show the elegant simplification when combined with softmax.
- `softmax_cross_entropy_backward(logits, y_true)` -- Derive the combined softmax + cross-entropy gradient. Show it simplifies to (softmax(logits) - y_one_hot).
- `train_neural_network(X, y, layer_sizes, lr, n_epochs, batch_size)` -- Put it all together: forward, loss, backward, update. Train a multi-layer network from scratch.

**"Naive then Derive" Challenge:** Implement backprop for a 2-layer network by manually writing out all the partial derivatives for that specific architecture. Then derive the general algorithm using the chain rule on a computational graph. Show your general backprop produces the same gradients as the manual version. The manual version is error-prone and does not generalize; the derived version works for any architecture.

**Test Scenarios:**
- Gradient check passes for all layers (relative error < 1e-5)
- 2-layer network learns XOR (loss < 0.01 after training)
- 3-layer network achieves >95% on MNIST digits 0-1
- Softmax + cross-entropy gradient matches numerical gradient
- Training loss monotonically decreases for small enough learning rate

**Mini-Project: Computational Graph Engine**
- Build a simple autograd engine (inspired by Karpathy's micrograd but extended to matrices)
- Implement Value/Tensor class with forward operations: add, mul, matmul, relu, sigmoid, log
- Each operation records itself in a computation graph
- Implement backward() that traverses the graph and computes all gradients
- Demonstrate: define a neural network using the engine, train it, verify gradients match manual backprop

### Module 02: Activation Functions and Weight Initialization

**Math Resources:**
- Goodfellow "Deep Learning" chapter 6.2-6.3 (Hidden Units, Architecture Design)
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) -- the Xavier initialization paper
- He et al., "Delving Deep into Rectifiers" (2015) -- He initialization for ReLU

**Exercise Stubs:**
- `sigmoid_and_gradient(z)` -- Implement sigmoid and derive its gradient: sigma'(z) = sigma(z)(1 - sigma(z)). Show the vanishing gradient problem: gradient is near 0 for |z| > 5.
- `tanh_and_gradient(z)` -- Implement tanh and derive its gradient. Show it is a rescaled sigmoid. Still vanishes.
- `relu_and_gradient(z)` -- Implement ReLU and its subgradient. Show gradient is exactly 0 or 1: no vanishing, but "dying ReLU" problem.
- `leaky_relu_and_gradient(z, alpha)` -- Fix dying ReLU. Derive gradient.
- `gelu_and_gradient(z)` -- Implement GELU (the Transformer activation). Derive as x * Phi(x) where Phi is the Gaussian CDF.
- `xavier_init(fan_in, fan_out)` -- Derive Xavier initialization: Var(w) = 2/(fan_in + fan_out). Implement. Show it maintains activation variance through layers.
- `he_init(fan_in, fan_out)` -- Derive He initialization for ReLU: Var(w) = 2/fan_in. Implement.
- `track_activation_statistics(X, weights_list, activation_fn)` -- Forward-pass X through a deep network, recording mean and variance of activations at each layer. Visualize to show vanishing/exploding.

**"Naive then Derive" Challenge:** Initialize a 20-layer network with random N(0,1) weights and sigmoid activations. Forward-pass data and observe activations collapsing to 0 (vanishing). Then derive Xavier initialization from the condition that Var(output) = Var(input) for each layer. Apply it and show activations stay healthy. Repeat with ReLU and He initialization.

**Test Scenarios:**
- Sigmoid output in (0,1), gradient in (0, 0.25]
- ReLU gradient is 0 for z<0, 1 for z>0
- Xavier-initialized 20-layer sigmoid network: activation variance within [0.5, 2.0] at all layers
- He-initialized 20-layer ReLU network: activation variance within [0.5, 2.0] at all layers
- N(0,1)-initialized 20-layer sigmoid network: activation variance < 0.01 by layer 20 (vanishing)

### Module 03: Optimization for Deep Learning

**Math Resources:**
- Goodfellow "Deep Learning" chapter 8 (Optimization)
- Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014 paper)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (AdamW paper, 2019)
- Smith, "Cyclical Learning Rates for Training Neural Networks" (2017)

**Exercise Stubs:**
- `sgd_with_lr_schedule(model, data, lr_fn, n_epochs)` -- SGD with a learning rate schedule function. Implement step decay, exponential decay, cosine annealing.
- `lr_warmup(base_lr, warmup_steps, current_step)` -- Linear warmup. Derive why Transformers need this: Adam's second moment estimate is inaccurate early on.
- `cosine_annealing_lr(base_lr, current_step, total_steps)` -- Implement cosine schedule. Return lr at current step.
- `adamw(grad_fn, params, lr, beta1, beta2, weight_decay, n_steps)` -- Implement AdamW (decoupled weight decay). Derive the difference from L2-regularized Adam.
- `gradient_clipping(gradients, max_norm)` -- Implement gradient norm clipping. Explain why RNNs need this.
- `learning_rate_finder(model, data, lr_range, n_steps)` -- Implement the LR range test: increase LR exponentially, plot loss vs LR, find the steepest descent point.
- `batch_size_effect(model, data, batch_sizes, n_epochs)` -- Train with different batch sizes, record loss curves. Show the generalization gap between large and small batches.

**"Naive then Derive" Challenge:** Train a neural network with a fixed learning rate. It converges slowly. Then implement cosine annealing and show faster convergence. Derive why: the loss landscape has sharp and flat minima; annealing helps escape sharp minima (which generalize poorly) and settle in flat minima.

**Test Scenarios:**
- Cosine annealing returns base_lr at step 0 and ~0 at total_steps
- Warmup returns 0 at step 0 and base_lr at warmup_steps
- AdamW with weight_decay=0 matches Adam
- Gradient clipping reduces norm to at most max_norm
- LR finder identifies a reasonable learning rate (loss decreasing before divergence)

### Module 04: Regularization for Deep Learning

**Math Resources:**
- Goodfellow "Deep Learning" chapter 7 (Regularization)
- Srivastava et al., "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" (2014)
- Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training" (2015)

**Exercise Stubs:**
- `dropout_forward(X, p, training)` -- Implement dropout: zero out neurons with probability p during training, scale by 1/(1-p). During inference, do nothing.
- `dropout_backward(dout, mask, p)` -- Backprop through dropout.
- `batch_norm_forward(X, gamma, beta, running_mean, running_var, training, momentum)` -- Implement batch normalization. Compute mean/var per feature, normalize, scale/shift.
- `batch_norm_backward(dout, cache)` -- Derive and implement the backward pass for batch norm. This is notoriously tricky -- derive step by step.
- `layer_norm_forward(X, gamma, beta)` -- Implement layer normalization (normalize per sample, not per feature). Explain why Transformers use LayerNorm instead of BatchNorm.
- `layer_norm_backward(dout, cache)` -- Backward pass.
- `weight_decay_vs_l2(params, gradients, weight_decay, l2_lambda)` -- Show that for SGD, weight decay and L2 regularization are equivalent, but for Adam they differ. Implement both and demonstrate.
- `data_augmentation_effect(model, X, y, augment_fn, n_epochs)` -- Train with and without augmentation, compare test accuracy and loss curves.

**"Naive then Derive" Challenge:** Train a network on a small dataset (100 samples) without any regularization. Observe severe overfitting. Then add dropout and batch norm one at a time. Derive why dropout works as approximate ensemble averaging and why batch norm smooths the loss landscape.

**Test Scenarios:**
- Dropout zeros out approximately p fraction of neurons (check over many runs)
- Dropout forward in eval mode is identity
- BatchNorm output has mean approximately 0 and variance approximately 1 per feature (during training)
- BatchNorm running_mean and running_var update correctly
- LayerNorm normalizes per sample
- BatchNorm backward gradient check passes

**Mini-Project: Regularization Ablation Study**
- Take a neural network and a dataset where overfitting is easy (small dataset, large model)
- Train 6 variants: no regularization, dropout only, batch norm only, weight decay only, dropout + batch norm, all three
- For each: plot training and validation loss curves, record final test accuracy
- Produce a summary table comparing all variants
- Write a 1-paragraph analysis of which regularization techniques helped most and why

### Module 05: Convolutional Neural Networks

**Math Resources:**
- Goodfellow "Deep Learning" chapter 9 (Convolutional Networks)
- CS231n lecture notes on ConvNets (Stanford, free online)
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning" (2016, free PDF with excellent diagrams)

**Exercise Stubs:**
- `conv2d_naive(X, kernel, stride, padding)` -- Implement 2D convolution with nested loops. This is the "wrong way."
- `conv2d_im2col(X, kernel, stride, padding)` -- Implement the im2col trick: reshape input patches into columns, then convolution becomes matrix multiplication. Derive the reshaping.
- `conv2d_backward(dout, X, kernel, stride, padding)` -- Derive the backward pass for convolution. Show that the gradient w.r.t. the kernel is also a convolution.
- `max_pool_forward(X, pool_size, stride)` -- Implement max pooling. Store argmax positions for backward.
- `max_pool_backward(dout, cache)` -- Route gradients only to the max positions.
- `receptive_field_calculator(layer_configs)` -- Given a list of (kernel_size, stride, padding) tuples, compute the receptive field at each layer.
- `build_cnn(input_shape, conv_configs, fc_sizes, n_classes)` -- Assemble a CNN from conv layers, pooling, and FC layers. Return a model dict with all weights/biases.
- `forward_cnn(X, model)` -- Forward pass through the full CNN.
- `backward_cnn(loss_grad, model, caches)` -- Full backward pass through the CNN.

**"Naive then Derive" Challenge:** Implement conv2d with 4 nested for-loops. Time it on a 32x32 image with a 3x3 kernel. Then derive the im2col trick: show that convolution can be expressed as matrix multiplication on a reshaped input. Implement im2col and compare timing. The speed difference is dramatic.

**Test Scenarios:**
- Naive and im2col produce identical outputs
- Convolution with identity kernel (center=1, rest=0) returns input
- Convolution with edge-detection kernel detects edges (visual check)
- Backward gradient check passes for conv and pooling layers
- Receptive field calculation matches hand-computed values for a simple 3-layer CNN
- CNN trains and loss decreases on CIFAR-10 subset (even if accuracy is modest -- from-scratch CNN in NumPy will be slow)

**Mini-Project: CNN from Scratch in NumPy**
- Build a 3-layer CNN: Conv(3x3, 8 filters) -> ReLU -> MaxPool(2x2) -> Conv(3x3, 16 filters) -> ReLU -> MaxPool(2x2) -> FC -> Softmax
- Train on MNIST (not CIFAR -- NumPy CNN will be too slow for CIFAR)
- Achieve >95% test accuracy
- Visualize learned filters in the first conv layer
- Visualize feature maps at each layer for a sample image
- Compare training time against the same architecture in PyTorch

### Module 06: RNNs, LSTMs, and GRUs

**Math Resources:**
- Goodfellow "Deep Learning" chapter 10 (Sequence Modeling: Recurrent and Recursive Nets)
- Olah, "Understanding LSTM Networks" blog post
- Karpathy, "The Unreasonable Effectiveness of Recurrent Neural Networks" blog post
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) -- section 1-3

**Exercise Stubs:**
- `rnn_cell_forward(x_t, h_prev, Wxh, Whh, bh)` -- Implement a single RNN cell: h_t = tanh(Wxh * x_t + Whh * h_prev + bh).
- `rnn_forward(X_seq, h0, Wxh, Whh, bh)` -- Unroll the RNN over a sequence. Return all hidden states.
- `rnn_backward(dh_all, cache)` -- Implement backpropagation through time (BPTT). Derive the gradient flow and show the vanishing gradient problem: gradient is multiplied by Whh at each step.
- `gradient_flow_analysis(Whh, T)` -- Compute ||dh_0/dh_T|| for sequence length T. Show it goes to 0 (vanishing) or infinity (exploding) depending on the spectral radius of Whh.
- `lstm_cell_forward(x_t, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo)` -- Implement the LSTM cell with forget, input, cell, and output gates. Derive each gate's purpose.
- `lstm_forward(X_seq, h0, c0, params)` -- Unroll LSTM over a sequence.
- `lstm_backward(dh_all, cache)` -- BPTT for LSTM. Show that the cell state gradient flows through addition (not multiplication), avoiding vanishing.
- `gru_cell_forward(x_t, h_prev, Wz, Wr, Wh, bz, br, bh)` -- Implement GRU cell. Derive as a simplified LSTM with merged forget/input gates.

**"Naive then Derive" Challenge:** Train a vanilla RNN on sequences of length 50. Show it fails to learn long-range dependencies. Analyze the gradient flow: compute the gradient of loss w.r.t. the first hidden state and show it vanishes. Then derive the LSTM: show that the cell state acts as a "gradient highway" because it uses addition instead of multiplication. Implement LSTM and show it learns the same task.

**Test Scenarios:**
- RNN forward matches manual step-by-step computation for 3-step sequence
- BPTT gradient check passes for RNN and LSTM
- Gradient magnitude at step 0 for vanilla RNN decays exponentially with sequence length
- Gradient magnitude at step 0 for LSTM stays bounded
- LSTM learns a simple sequence copy task (copy input to output after a delay)
- GRU matches LSTM quality on the copy task with fewer parameters

**Mini-Project: Character-Level Language Model**
- Implement a character-level LSTM language model
- Train on a text corpus (Shakespeare or similar, ~1MB)
- Generate text by sampling from the model
- Plot training loss over time
- Compare generated text from vanilla RNN vs LSTM after same number of training steps
- Show that RNN-generated text is less coherent at longer generation lengths

### Module 07: Attention and Transformers

**Math Resources:**
- Vaswani et al., "Attention Is All You Need" (2017) -- the original paper
- Jay Alammar, "The Illustrated Transformer" blog post
- Karpathy, "Let's build GPT" video lecture
- Bishop PRML -- no coverage (too old), use Goodfellow chapter 12 for sequence-to-sequence background

**Exercise Stubs:**
- `scaled_dot_product_attention(Q, K, V, mask)` -- Derive the scaling factor 1/sqrt(d_k) from the variance of dot products. Implement attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.
- `multi_head_attention(Q, K, V, n_heads, W_q, W_k, W_v, W_o, mask)` -- Split into heads, apply attention, concat, project. Derive why multiple heads learn different "attention patterns."
- `positional_encoding(seq_len, d_model)` -- Implement sinusoidal positional encoding. Derive why sin/cos at different frequencies encode relative position.
- `feed_forward_block(X, W1, b1, W2, b2)` -- The position-wise FFN: two linear layers with GELU in between.
- `transformer_encoder_layer(X, n_heads, d_model, d_ff, params)` -- Combine: LayerNorm -> Multi-Head Self-Attention -> Residual -> LayerNorm -> FFN -> Residual.
- `transformer_decoder_layer(X, encoder_output, n_heads, d_model, d_ff, params)` -- Add cross-attention. Implement causal masking for autoregressive decoding.
- `causal_mask(seq_len)` -- Generate the upper-triangular mask that prevents attending to future tokens.
- `full_transformer(src, tgt, encoder_layers, decoder_layers, params)` -- The complete encoder-decoder Transformer.

**"Naive then Derive" Challenge:** Implement sequence-to-sequence with an LSTM (from Module 06). Note the bottleneck: the entire source sequence is compressed into a single hidden state vector. Then derive attention as a solution: let the decoder look at ALL encoder hidden states, weighted by relevance. Show that attention removes the bottleneck. Then derive the Transformer: what if we use ONLY attention, no recurrence at all?

**Test Scenarios:**
- Scaled dot-product attention: when Q=K, attention weights are uniform (every token attends equally to itself and others of equal value)
- Multi-head attention output shape is correct
- Positional encoding has correct shape and values match the sin/cos formulas
- Causal mask prevents attending to future positions
- Transformer encoder layer output has same shape as input
- Full Transformer can overfit a tiny dataset (e.g., reverse a sequence of 5 tokens)

**Mini-Project: Transformer From Scratch (in PyTorch)**
- Implement a complete Transformer in PyTorch (not NumPy -- too slow for training)
- Use the from-scratch building blocks (attention, FFN, encoder/decoder layers), not nn.Transformer
- Train on a simple task: sorting sequences of integers (e.g., input: [3,1,4,1,5], output: [1,1,3,4,5])
- Achieve >95% accuracy on sequences of length 10
- Visualize attention patterns: which input tokens does the model attend to when producing each output token?
- Experiment: remove positional encoding and show the model fails on sequence-order-dependent tasks

### Module 08: Capstone -- Transformer from Scratch

**Requirements:**
- Build a complete, well-documented Transformer implementation in PyTorch
- Support both encoder-only (like BERT) and decoder-only (like GPT) configurations
- Train a small GPT-style model on a text dataset (character-level or BPE tokenized)
- Model should generate coherent text after training
- Include: multi-head attention, positional encoding, layer normalization, residual connections, causal masking
- Visualization notebook showing: attention heatmaps for generated text, loss curves, generated samples at different training checkpoints
- Write a 1-page analysis comparing your implementation against PyTorch's nn.Transformer: what differs, what you learned

---

## Phase 2D: Practical ML Engineering

**Goal:** Bridge the gap between "it works on my laptop" and "it works in production." Learn the tools and practices of professional ML engineering.

### Directory Structure

```
spiral_2/2d_ml_engineering/
├── README.md
├── 01_experiment_tracking/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 02_hyperparameter_tuning/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # Bayesian optimization from scratch
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 03_data_pipelines/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   ├── mini_project.py          # PyTorch Dataset + DataLoader pipeline
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 04_model_debugging/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
├── 05_gpu_and_performance/
│   ├── README.md
│   ├── exercises.py
│   ├── notebook.ipynb
│   └── tests/
│       └── test_exercises.py
└── 06_capstone_production_pipeline/
    ├── README.md
    ├── pipeline.py
    ├── config.yaml
    ├── notebook.ipynb
    └── tests/
        └── test_pipeline.py
```

### Module 01: Experiment Tracking

**Resources:**
- MLflow documentation (quickstart + tracking guide)
- Weights & Biases documentation (quickstart)
- "How to organize your ML experiments" -- practical blog posts

**Exercise Stubs:**
- `setup_mlflow_experiment(experiment_name)` -- Create an MLflow experiment. Return experiment ID.
- `log_training_run(model_name, params, metrics, artifacts_dir)` -- Log a complete training run: hyperparameters, metrics at each epoch, model artifacts.
- `compare_runs(experiment_name, metric_name)` -- Query MLflow for all runs in an experiment, return sorted by metric.
- `log_model_with_signature(model, X_sample, y_sample, model_name)` -- Log a model with its input/output signature for reproducibility.
- `create_experiment_report(experiment_name)` -- Generate a summary: best run, hyperparameter distributions, metric correlations.

**"Naive then Derive" Challenge:** Run 10 training experiments with different hyperparameters, saving results in print statements and text files. Try to figure out which configuration was best. Then implement MLflow tracking and show how it makes comparison trivial. The pain of the "naive way" motivates the tooling.

**Test Scenarios:**
- MLflow experiment is created and accessible
- Logged parameters and metrics are retrievable
- Run comparison returns runs in correct sorted order
- Model signature correctly captures input/output shapes

### Module 02: Hyperparameter Tuning

**Math Resources:**
- Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (2012)
- Snoek et al., "Practical Bayesian Optimization of Machine Learning Algorithms" (2012)
- Optuna documentation

**Exercise Stubs:**
- `grid_search(model_fn, param_grid, X, y, k_folds)` -- Implement exhaustive grid search with k-fold CV. Return best params and all results.
- `random_search(model_fn, param_distributions, X, y, k_folds, n_iter)` -- Implement random search. Derive why it is more efficient than grid search in high dimensions.
- `gaussian_process_surrogate(X_observed, y_observed, X_candidates)` -- Implement a simple GP posterior for Bayesian optimization. Return predicted mean and uncertainty.
- `expected_improvement(mu, sigma, y_best)` -- Derive and implement the EI acquisition function.
- `bayesian_optimization(objective_fn, param_space, n_initial, n_iterations)` -- Full Bayesian optimization loop: fit GP surrogate, maximize EI to choose next point, evaluate, repeat.
- `optuna_integration(model_fn, X, y, n_trials)` -- Use Optuna's TPE sampler for hyperparameter tuning. Compare against your from-scratch Bayesian optimization.
- `early_stopping_tuning(model_fn, X, y, patience, n_trials)` -- Implement early stopping within each trial (stop unpromising runs early).

**"Naive then Derive" Challenge:** Do a 5-dimensional grid search with 10 values per dimension (100,000 evaluations). Then show that random search with 100 evaluations finds a comparable result. Derive why: in high dimensions, grid search wastes evaluations on unimportant parameters.

**Test Scenarios:**
- Grid search evaluates all param combinations
- Random search with large n_iter approaches grid search quality
- GP surrogate uncertainty is high far from observed points, low near them
- EI is 0 when predicted mean is below y_best and uncertainty is 0
- Bayesian optimization converges on simple test functions (1D quadratic, 2D Branin)

**Mini-Project: Bayesian Optimization from Scratch**
- Implement a full Bayesian optimization library: GP surrogate, EI acquisition, optimization loop
- Test on: 1D function (visual), 2D Branin function (standard benchmark), hyperparameter tuning for a neural network
- At each iteration, plot: GP posterior mean and uncertainty band, observed points, EI function, next point to evaluate
- Compare sample efficiency against random search: Bayesian optimization should find a good solution in fewer evaluations

### Module 03: Data Pipelines

**Resources:**
- PyTorch DataLoader documentation
- "A Recipe for Training Neural Networks" by Karpathy (blog post)

**Exercise Stubs:**
- `custom_dataset(data_path, transform_fn)` -- Implement a PyTorch-style Dataset class with __len__ and __getitem__. Include lazy loading.
- `data_augmentation_pipeline(image)` -- Implement common augmentations from scratch: random crop, horizontal flip, color jitter, normalization.
- `stratified_split(X, y, test_ratio, val_ratio, seed)` -- Split data maintaining class proportions. Implement from scratch (no sklearn).
- `efficient_dataloader(dataset, batch_size, shuffle, num_workers)` -- Wrap a dataset in a PyTorch DataLoader. Benchmark different num_workers values.
- `handle_imbalanced_data(X, y, strategy)` -- Implement oversampling (SMOTE-lite), undersampling, and class-weighted loss. Compare on an imbalanced dataset.
- `data_validation(X, y)` -- Check for: NaN values, constant features, duplicate rows, class imbalance, feature scale issues. Return a report.

**"Naive then Derive" Challenge:** Load an entire dataset into memory, shuffle with Python's random.shuffle, and batch with list slicing. Time the training loop. Then implement a proper DataLoader with prefetching and multiple workers. Show the GPU utilization difference.

**Test Scenarios:**
- Custom dataset returns correct items by index
- Augmented images have correct shapes and value ranges
- Stratified split maintains class proportions within 5% of original
- Data validation catches planted NaN, constant, and duplicate issues

**Mini-Project: Production Data Pipeline**
- Build a complete training pipeline for CIFAR-10 using PyTorch
- Include: custom Dataset, augmentation, stratified splits, DataLoader with prefetching
- Measure and report: data loading time vs GPU compute time (identify if data-bound)
- Implement a simple caching strategy for preprocessed data
- Compare training speed with num_workers=0,1,2,4,8

### Module 04: Model Debugging

**Resources:**
- Karpathy, "A Recipe for Training Neural Networks" blog post
- "Troubleshooting Deep Neural Networks" by Josh Tobin (presentation slides, free)

**Exercise Stubs:**
- `overfit_single_batch(model, X_batch, y_batch, lr, n_steps)` -- The first debugging step: can the model memorize one batch? If not, something is fundamentally broken.
- `gradient_statistics(model, X, y)` -- Compute gradient mean, std, min, max for each layer. Detect vanishing/exploding gradients.
- `activation_statistics(model, X)` -- Compute activation mean, std, fraction of dead neurons at each layer.
- `learning_rate_sensitivity(model, X, y, lr_values)` -- Train for a few epochs at different LRs. Plot loss vs LR. Find the sweet spot.
- `detect_data_leakage(X_train, X_test, feature_names)` -- Check for features that are suspiciously predictive (likely leaking the target). Check for duplicate rows across train/test.
- `training_diagnostics(train_losses, val_losses, train_accs, val_accs)` -- Analyze training curves and return diagnosis: underfitting, overfitting, good fit, training instability.
- `ablation_study(model_fn, ablation_configs, X, y)` -- Train multiple model variants, each with one component removed. Identify which components contribute to performance.

**"Naive then Derive" Challenge:** Take a neural network with a subtle bug (e.g., wrong loss function, data not shuffled, learning rate 10x too high). Run it and just stare at the loss curve. Now implement the systematic debugging checklist: overfit one batch, check gradients, check activations, try different LRs. Show how each check identifies a different class of bugs.

**Test Scenarios:**
- overfit_single_batch achieves loss < 0.01 on a correctly implemented model
- gradient_statistics detects vanishing gradients in a 50-layer sigmoid network
- activation_statistics detects dead ReLU neurons with bad initialization
- training_diagnostics correctly labels overfitting (train loss << val loss) and underfitting (both loss high)

### Module 05: GPU and Performance

**Resources:**
- PyTorch CUDA documentation
- "Efficient PyTorch" blog posts
- NVIDIA Deep Learning Performance Guide

**Exercise Stubs:**
- `benchmark_cpu_vs_gpu(matrix_sizes, operations)` -- Compare CPU vs GPU timing for matrix multiply, convolution, and element-wise operations at various sizes. Find the crossover point.
- `memory_profiling(model, X, batch_sizes)` -- Estimate memory usage at different batch sizes. Predict the maximum batch size that fits in GPU memory.
- `mixed_precision_training(model, X, y, n_epochs)` -- Implement mixed precision (FP16 forward, FP32 gradients) using PyTorch's autocast. Compare speed and memory.
- `gradient_accumulation(model, X, y, micro_batch_size, accumulation_steps)` -- Simulate a large batch size by accumulating gradients. Verify equivalence to actual large batch.
- `profile_training_loop(model, dataloader, n_steps)` -- Use PyTorch profiler to identify bottlenecks: data loading, forward pass, backward pass, optimizer step.
- `torch_compile_speedup(model, X)` -- Compare model inference time with and without torch.compile. Show the JIT compilation overhead and subsequent speedup.

**"Naive then Derive" Challenge:** Train a model on CPU. Then move to GPU and expect 100x speedup. Find that the speedup is only 2x because data loading is the bottleneck. Profile the training loop, identify the bottleneck, fix the DataLoader, and THEN see the real speedup.

**Test Scenarios:**
- GPU is faster than CPU for matrix multiply at size > 128 (approximate threshold)
- Mixed precision produces results within tolerance of FP32
- Gradient accumulation over 4 micro-batches matches 1 large batch (within numerical tolerance)
- Profiler correctly identifies the slowest phase of the training loop

### Module 06: Capstone -- Production Pipeline

**Requirements:**
- Take the best model from Phase 2B or 2C and build a production-quality training pipeline
- Must include: experiment tracking (MLflow), hyperparameter tuning (Optuna), proper data pipeline (PyTorch DataLoader with augmentation), model checkpointing, early stopping, learning rate scheduling, gradient clipping, mixed precision training
- Configuration via YAML file (no hardcoded hyperparameters)
- Reproducibility: seed everything, log all configs, save the exact commit hash
- Comprehensive logging: loss curves, gradient norms, learning rate schedule, training time per epoch
- A "run training" script that takes a config file and produces a fully tracked experiment
- Final deliverable: a notebook that loads the best model from MLflow, evaluates it, and produces a model card (accuracy, training details, known limitations)

---

## Cross-Phase Dependencies

- Phase 2A Module 06 (mathlib) is used throughout Phase 2B
- Phase 2B Module 08 (ml_library) is used for comparison in Phase 2C
- Phase 2C Module 01 (backprop) builds on Phase 2A Module 02 (calculus)
- Phase 2C Module 07 (Transformers) builds on all previous 2C modules
- Phase 2D uses models from both 2B and 2C for its exercises

## Estimated Timeline (no time pressure, depth over speed)

- Phase 2A: 4-6 weeks (math is the foundation; do not rush)
- Phase 2B: 6-8 weeks (7 algorithm modules + capstone)
- Phase 2C: 6-8 weeks (7 theory modules + Transformer capstone)
- Phase 2D: 3-4 weeks (practical, builds on everything before)

---

### Critical Files for Implementation

- `/Users/kkondratuk/.claude/plans/luminous-swinging-quilt.md` - Scaffolding pattern to follow: self-contained module dirs, stubs-only exercises, TDD, mini-projects, math callouts, style notes
- `/Users/kkondratuk/PycharmProjects/LearningML/docs/plans/2026-03-22-ml-curriculum-design.md` - Master curriculum design defining the spiral structure and Phase 2A-2D scope
- `/Users/kkondratuk/PycharmProjects/LearningML/docs/plans/2026-03-22-phase-1a-scientific-python.md` - Phase 1A implementation plan showing the exact test-first pattern, commit strategy, and exercise style to replicate
- `/Users/kkondratuk/PycharmProjects/LearningML/pyproject.toml` - Dependencies file to update with Spiral 2 requirements (torch, mlflow, optuna, tqdm)
- `/Users/kkondratuk/PycharmProjects/LearningML/spiral_2/` - Top-level directory to create, housing all four phases (2a through 2d)