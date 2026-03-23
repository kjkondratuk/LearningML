# Module 08: Feature Engineering

## Overview

Feature engineering is how you convert raw data into features that make models
work better. It is often the difference between a mediocre model and a good one.
This module covers encoding, scaling, polynomial features, and feature selection.

## Core Concepts

### Encoding Categorical Variables

ML models work with numbers, not strings. You need to convert categorical features:

- **Ordinal encoding:** Assign integers based on order (e.g., low=0, medium=1,
  high=2). Use when there is a natural ordering.
- **One-hot encoding:** Create a binary column for each category. Use when there
  is no natural ordering.

**Go parallel:** Ordinal encoding is like defining a custom `Stringer` interface
where the ordering matters. One-hot encoding is like defining an `enum` with
`iota` — each value gets its own slot.

### Feature Scaling

Many algorithms (linear regression, SVM, K-means) are sensitive to feature scale.
Two common approaches:

- **Standardization (z-score):** Subtract mean, divide by std. Result has mean=0,
  std=1.
- **Min-max scaling:** Scale to [0, 1] range.

Critical rule: fit the scaler on training data only, then transform both train and
test. Otherwise you leak test statistics into your training process.

### Polynomial Features

Create new features by combining existing ones:

    [x1, x2] -> [x1, x2, x1^2, x1*x2, x2^2]

This lets linear models capture nonlinear relationships. The tradeoff: more
features means more risk of overfitting.

### Feature Selection

Not all features help. Some add noise. Feature selection picks the most informative
features:

- **Univariate selection:** Score each feature independently (e.g., mutual
  information, ANOVA F-test) and keep the top K.
- **Model-based selection:** Use a model (e.g., random forest) to rank feature
  importance.

Resources:
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Scikit-learn Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Feature Engineering for ML (Google)](https://developers.google.com/machine-learning/data-prep)

## Exercises

Implement in `exercises.py`:

1. `encode_ordinal` — Map categorical string values to ordered integers.
2. `encode_onehot_manual` — One-hot encode a categorical column from scratch.
3. `scale_features` — Standardize features (z-score normalization).
4. `create_polynomial_features` — Generate degree-2 polynomial features.
5. `select_top_k_features` — Select the K most informative features using
   mutual information.
6. `build_feature_pipeline` — Chain encoding, scaling, and selection into one
   function.

## Style Notes

- `encode_ordinal` should accept a mapping dict (e.g., {"low": 0, "medium": 1,
  "high": 2}).
- `encode_onehot_manual` should work on a 1D array of strings and return a 2D
  binary array. Do not use sklearn's OneHotEncoder.
- `scale_features` should return both the scaled data and the scaler parameters
  (mean, std) so the same transform can be applied to test data.
- `create_polynomial_features` should not use sklearn. Build the combinations
  yourself.
- `select_top_k_features` should use sklearn's `mutual_info_classif` or
  `mutual_info_regression` internally.
