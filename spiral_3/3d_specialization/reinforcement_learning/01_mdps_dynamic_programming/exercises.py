"""
MDP & Dynamic Programming Exercises
Reference: Sutton & Barto, ch. 3-4
"""

import numpy as np


def value_iteration(
    transition_probs: np.ndarray, rewards: np.ndarray, gamma: float, theta: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Value iteration: compute optimal value function and policy.

    Args:
        transition_probs: P(s'|s,a), shape (S, A, S).
        rewards: R(s,a,s'), shape (S, A, S).
        gamma: Discount factor.
        theta: Convergence threshold.

    Returns:
        Tuple of (V_optimal, policy_optimal).
    """
    raise NotImplementedError


def policy_iteration(
    transition_probs: np.ndarray, rewards: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Policy iteration: alternate policy evaluation and improvement.

    Args:
        transition_probs: P(s'|s,a), shape (S, A, S).
        rewards: R(s,a,s'), shape (S, A, S).
        gamma: Discount factor.

    Returns:
        Tuple of (V_optimal, policy_optimal).
    """
    raise NotImplementedError


def bellman_operator(
    V: np.ndarray, transition_probs: np.ndarray, rewards: np.ndarray, gamma: float
) -> np.ndarray:
    """Single application of the Bellman optimality operator.

    (TV)(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma*V(s')]

    Args:
        V: Current value function, shape (S,).
        transition_probs: shape (S, A, S).
        rewards: shape (S, A, S).
        gamma: Discount factor.

    Returns:
        Updated value function, shape (S,).
    """
    raise NotImplementedError


def verify_contraction(
    V1: np.ndarray, V2: np.ndarray, T_V1: np.ndarray, T_V2: np.ndarray, gamma: float
) -> bool:
    """Verify ||T(V1) - T(V2)||_inf <= gamma * ||V1 - V2||_inf.

    Returns:
        True if contraction property holds.
    """
    raise NotImplementedError
