# Module 05: Support Vector Machines

> SVMs are the crown jewel of classical ML theory: maximum margin, duality theory,
> and the kernel trick all come together. This module derives them from constrained
> optimization.

## Learning Objectives

- Derive the maximum margin classifier from the primal formulation.
- Derive the dual via Lagrange multipliers and understand support vectors.
- Implement soft-margin SVM with slack variables.
- Implement the kernel trick and understand why it avoids explicit feature maps.
- Implement a simplified version of SMO.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, sections 7.1, 7.1.1 | Max margin, soft margin |
| Murphy, *MLAPP*, ch 14 | Kernels |
| Boyd & Vandenberghe, ch 5 | Duality theory |

## Derive It

1. **Margin = 2/||w||.** Show that for a hyperplane w^T x + b = 0, the distance
   from a point x_i to the plane is |w^T x_i + b| / ||w||. The margin is
   the distance to the closest point: 2 / ||w||.

2. **Dual formulation.** Write the Lagrangian, take derivatives w.r.t. w and b,
   substitute back. The dual has only the alpha variables and involves x_i^T x_j.

3. **Kernel trick.** Show that the dual only involves inner products x_i^T x_j.
   Replace with k(x_i, x_j) = phi(x_i)^T phi(x_j). Now you are optimizing in
   the feature space without ever computing phi(x).

## "Naive then Derive" Challenge

Try to classify XOR data with a linear SVM. Fail. Then derive the kernel trick
with RBF and classify it perfectly. Visualize the decision boundary.

## Exercises

See `exercises.py`.

## Mini-Project: Kernel SVM on Non-Linear Data

Decision boundary visualization for 3 datasets x 3 kernels. See `mini_project.py`.
