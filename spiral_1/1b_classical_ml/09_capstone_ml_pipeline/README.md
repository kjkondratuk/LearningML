# Module 09: Capstone — End-to-End ML Pipeline

## Overview

This is where everything comes together. You will build a complete machine learning
pipeline from raw data to final model selection, using the skills from all previous
modules.

**Go parallel:** Think of this as building a complete microservice from scratch. You
have done the individual pieces (HTTP handler, database layer, auth middleware) in
isolation. Now you wire them together, handle the edge cases, and ship it.

## The Task

Build two pipelines:

1. **Regression:** Predict housing prices (Ames Housing or synthetic equivalent).
2. **Classification:** Predict a binary outcome (your choice of dataset).

For each pipeline, you will:
1. Load and explore the data.
2. Engineer at least 5 meaningful features.
3. Train and compare at least 4 models.
4. Select the best model with justification.
5. Produce diagnostic plots (overfitting plot, learning curve).

## Deliverables

Your `pipeline.py` should produce a results dict containing:

- **5 engineered features:** At least 5 features you created (polynomial,
  interaction, binned, encoded, etc.) beyond the raw columns.
- **4 models compared:** At least 4 different models (e.g., linear regression,
  decision tree, random forest, gradient boosting) with train and test scores.
- **1-paragraph winner explanation:** A plain-English string explaining why you
  picked the winning model. Not "it had the best score" but *why* it had the
  best score (bias-variance tradeoff, feature importance, etc.).
- **Overfitting plot data:** Train and test scores across model complexity
  (e.g., tree depth or regularization strength).
- **Learning curve data:** Train and test scores at different training set sizes.

## Pipeline Steps

### 1. load_and_explore

Load the data, compute basic statistics (shape, missing values, feature types).
Return a summary dict.

### 2. build_features

Apply feature engineering: handle missing values, encode categoricals, create
polynomial or interaction features, scale numerics. Return processed train and
test arrays.

### 3. train_and_evaluate_models

Train at least 4 models. For each, record:
- Model name
- Training score
- Test score
- Key hyperparameters

### 4. select_best_model

Pick the model with the best test score. Write a 1-paragraph justification.

### 5. final_evaluation

Produce the diagnostic data: overfitting plot data and learning curve data.

## Resources

- [Ames Housing dataset info](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- All previous module READMEs

## Style Notes

- Each function in `pipeline.py` should be independently testable.
- Use `sklearn.datasets` for synthetic data if you do not want to download
  external datasets.
- The explanation string should be between 50 and 300 words.
- Return all results as a dict so tests can verify structure and basic quality.
- Do not hardcode model scores — the tests check structure, not exact values.
