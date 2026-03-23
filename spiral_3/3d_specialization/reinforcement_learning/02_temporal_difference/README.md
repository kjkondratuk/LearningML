# Module 02: Temporal Difference Learning

## Key References
- **Sutton & Barto.** Ch. 6-7.

## Theoretical Background
TD learning bootstraps: V(s) += alpha * (r + gamma*V(s') - V(s)).
SARSA is on-policy TD control. Q-learning is off-policy.
TD(lambda) unifies MC and TD via eligibility traces.

> **Math Callout:** Bootstrapping, convergence of stochastic approximation, eligibility traces.

## Exercises
1. `td_zero_update` -- TD(0) value update
2. `sarsa_update` -- On-policy TD control
3. `q_learning_update` -- Off-policy TD control
4. `td_lambda_return` -- Lambda-return
5. `eligibility_trace_update` -- Accumulating traces
