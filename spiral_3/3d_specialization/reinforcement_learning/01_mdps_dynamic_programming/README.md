# Module 01: MDPs & Dynamic Programming

## Key References
- **Sutton & Barto.** "Reinforcement Learning: An Introduction." 2nd ed. Ch. 3-4.

## Theoretical Background
An MDP is defined by (S, A, P, R, gamma). The Bellman optimality equation is:
V*(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma * V*(s')]

Value iteration applies the Bellman operator repeatedly until convergence.
The Bellman operator is a contraction mapping with factor gamma, guaranteeing
convergence to the unique fixed point V*.

> **Math Callout:** Contraction mappings (Banach fixed point theorem), dynamic
> programming principle. Resource: Sutton & Barto ch. 3-4.

## Exercises
1. `value_iteration` -- Iterate until convergence
2. `policy_iteration` -- Alternate evaluation and improvement
3. `bellman_operator` -- Single Bellman backup
4. `verify_contraction` -- Verify ||TV1 - TV2|| <= gamma * ||V1 - V2||
