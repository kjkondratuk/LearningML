# Module 06: Decision Trees and Ensembles

> A single deep tree overfits. Bagging (random forests) reduces variance without
> increasing bias. Boosting reduces bias by fitting residuals. This module derives
> and implements all three.

## Learning Objectives

- Implement Gini impurity, entropy, and information gain.
- Build a decision tree with recursive splitting.
- Implement bagging with random feature subsets (= random forest).
- Implement gradient boosting for regression.
- Understand why bagging reduces variance and boosting reduces bias.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, section 14.4 | Tree-based models |
| Murphy, *MLAPP*, ch 16 | Trees, bagging, boosting |
| Hastie, *ESL*, ch 9--10 | Trees and boosting (detailed) |
| Breiman, "Random Forests" (2001) | Introduction of the original paper |

## Derive It

1. **Gini impurity.** Derive from the misclassification probability: if you randomly
   label a sample according to the class distribution, Gini = 1 - sum(p_k^2).

2. **Information gain.** IG = H(parent) - weighted average H(children). Show this is
   the reduction in entropy from a split.

3. **Bagging reduces variance.** If you have B independent estimators each with
   variance sigma^2, their average has variance sigma^2/B. Trees are approximately
   independent due to random feature subsets.

4. **Gradient boosting.** Show that fitting the residuals r_t = y - f_{t-1}(x) is
   gradient descent in function space: f_t = f_{t-1} + lr * h_t where h_t fits grad L.

## "Naive then Derive" Challenge

Build a deep tree: 100% training accuracy, terrible test accuracy. Then implement
random forest and gradient boosting. Show each addresses a different problem.

## Exercises

See `exercises.py`.

## Mini-Project: Random Forest From Scratch

Full implementation, compared against sklearn. See `mini_project.py`.
