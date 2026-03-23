# Module 03: Policy Gradients

## Key Papers
- **Schulman et al. (2015).** "High-Dimensional Continuous Control Using Generalized Advantage Estimation." arXiv:1506.02438
- **Williams (1992).** "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning."

## Theoretical Background
The policy gradient theorem: grad J(theta) = E[sum_t grad log pi(a_t|s_t) * Q(s_t, a_t)].
REINFORCE uses Monte Carlo returns. Actor-critic uses a learned value function.
GAE provides a bias-variance tradeoff for advantage estimation.

> **Math Callout:** Policy gradient theorem proof, variance reduction via baselines, GAE.

## Exercises
1. `reinforce_loss` -- REINFORCE policy gradient loss
2. `compute_returns` -- Discounted returns
3. `generalized_advantage_estimation` -- GAE
4. `actor_critic_loss` -- Combined loss
5. `entropy_bonus` -- Exploration bonus
