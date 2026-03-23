# Phase 1B: Classical ML (Intuition Pass)

## Goal

Build working intuition for the core machine learning algorithms. You will implement
key pieces from scratch, then confirm your understanding with scikit-learn. Every
module follows the same loop: **read the concept, build it, break it, test it**.

This is an *intuition* pass, not a theory pass. You do not need to memorize proofs.
You need to be able to explain, in plain language, what each algorithm is doing and
why it works (or fails).

## Prerequisites

- Phase 1A (Python & NumPy foundations) complete
- Python environment with `numpy`, `scikit-learn`, `matplotlib`, and `pytest`

## Modules

| # | Module | Key Ideas | Deliverables |
|---|--------|-----------|-------------|
| 01 | [ML Workflow](01_ml_workflow/) | Train/test split, evaluation metrics, data leakage | exercises |
| 02 | [Linear Regression](02_linear_regression/) | Loss, gradient descent, closed-form vs iterative | exercises + mini-project |
| 03 | [Logistic Regression](03_logistic_regression/) | Sigmoid, log loss, decision boundary | exercises |
| 04 | [Decision Trees & Ensembles](04_decision_trees_and_ensembles/) | Gini, info gain, random forests, boosting | exercises + mini-project |
| 05 | [SVM](05_svm/) | Margins, kernels, the kernel trick | exercises |
| 06 | [Unsupervised Learning](06_unsupervised_learning/) | K-means, PCA, hierarchical clustering | exercises + mini-project |
| 07 | [Model Evaluation](07_model_evaluation/) | Confusion matrix, ROC, cross-validation | exercises |
| 08 | [Feature Engineering](08_feature_engineering/) | Encoding, scaling, polynomial features, selection | exercises |
| 09 | [Capstone: ML Pipeline](09_capstone_ml_pipeline/) | End-to-end pipeline on real data | pipeline project |

## How to Work Through This Phase

1. **Read the module README** to get the concept and the Go parallel.
2. **Run the tests first** (`pytest tests/`). They all fail. Good.
3. **Implement the stubs** in `exercises.py` until the tests go green.
4. **Do the "wrong way" challenges** where marked. Seeing failure builds intuition
   faster than reading about it.
5. **Mini-projects** tie exercises together into a realistic workflow.
6. **Capstone** is your proof that you can build a complete pipeline from scratch.

## Running Tests

```bash
# Run all tests for a module
cd spiral_1/1b_classical_ml/02_linear_regression
pytest tests/ -v

# Run a single test file
pytest tests/test_exercises.py -v

# Run all phase tests
cd spiral_1/1b_classical_ml
pytest -v --tb=short
```

## Estimated Time

- Modules 01-03: ~6-8 hours (foundations)
- Modules 04-06: ~8-10 hours (algorithms)
- Modules 07-08: ~4-6 hours (evaluation and features)
- Module 09: ~6-8 hours (capstone)
- **Total: ~24-32 hours**

## Go Developer Notes

Throughout this phase you will see Go parallels. ML in Python is a different
paradigm from what you are used to: mutable state everywhere, heavy use of
operator overloading via NumPy, and implicit broadcasting. Lean into it rather
than fighting it. The goal is fluency, not purity.
