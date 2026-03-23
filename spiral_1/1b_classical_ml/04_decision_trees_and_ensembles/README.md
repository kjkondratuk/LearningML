# Module 04: Decision Trees & Ensembles

## Overview

Decision trees are the first non-linear model in this curriculum. A tree splits
the data into regions by asking yes/no questions about features. Ensembles (random
forests, gradient boosting) combine many trees to get better results than any
single tree.

## Core Concepts

### Decision Trees

A decision tree recursively partitions the feature space. At each node, it picks
the feature and threshold that best separates the data.

**Go parallel:** Think of a decision tree as a chain of `if/else` statements. A
trained tree is literally compiled into a nested conditional:

```
if feature[2] <= 3.5:
    if feature[0] <= 1.2:
        return class_A
    else:
        return class_B
else:
    return class_C
```

The training process figures out which features to branch on and where to put the
thresholds.

### Gini Impurity

Gini impurity measures how "mixed" a set of labels is:

    gini(S) = 1 - sum(p_k^2) for each class k

where p_k is the proportion of class k in set S.

- Pure node (all same class): gini = 0
- Maximum impurity (2 classes, 50/50): gini = 0.5

### Information Gain

Information gain measures how much a split reduces impurity:

    IG = gini(parent) - (n_left/n * gini(left) + n_right/n * gini(right))

The best split maximizes information gain.

### Overfitting and Depth

Deep trees memorize the training data perfectly but fail on new data. This is the
single most important concept in tree-based models. The mini-project will have you
plot the U-shaped curve of test error vs. tree depth.

**Go parallel:** Overfitting is like writing a Go function with a massive switch
statement that has a case for every test input. It passes all tests but handles
nothing new. A shallow tree is like writing a clean, general-purpose function.

### Ensembles: Random Forest

A random forest trains many trees on random subsets of the data and features, then
averages their predictions. The randomness decorrelates the trees, reducing
overfitting.

### Ensembles: Gradient Boosting

Gradient boosting trains trees sequentially. Each new tree tries to correct the
errors of the previous ensemble. It is generally more powerful than random forests
but easier to overfit and harder to tune.

Resources:
- [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
- [StatQuest: Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [StatQuest: Gradient Boost (XGBoost)](https://www.youtube.com/watch?v=3CC4N4z3GJc)
- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)

## Exercises

Implement in `exercises.py`:

1. `gini_impurity` — Compute Gini impurity for a set of labels.
2. `information_gain` — Compute info gain for a candidate split.
3. `find_best_split` — Search all features and thresholds for the best split.
4. `fit_decision_stump` — Train a depth-1 decision tree (single split).
5. `sklearn_random_forest` — Train and evaluate a random forest.
6. `sklearn_gradient_boosting` — Train and evaluate a gradient boosting model.

## Mini-Project

In `mini_project.py`:

`compare_tree_depths` — Train decision trees at depths 1, 2, 3, 5, 10, 20, None
(unlimited). Record train and test accuracy for each. Return the data needed to
plot the overfitting U-curve.

## Style Notes

- For `find_best_split`, iterate over unique values in each feature column as
  candidate thresholds. This is brute force but correct.
- The decision stump returns a dict describing the split, not a sklearn object.
- Use sklearn's `DecisionTreeClassifier`, `RandomForestClassifier`, and
  `GradientBoostingClassifier` for the sklearn exercises.
