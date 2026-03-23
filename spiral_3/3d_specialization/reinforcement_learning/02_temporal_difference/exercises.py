"""Temporal Difference Exercises. Reference: Sutton & Barto ch. 6-7."""

import numpy as np


def td_zero_update(V: np.ndarray, state: int, next_state: int, reward: float, alpha: float, gamma: float) -> np.ndarray:
    """TD(0) update: V(s) += alpha * (r + gamma*V(s') - V(s))."""
    raise NotImplementedError


def sarsa_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int, a_next: int, alpha: float, gamma: float) -> np.ndarray:
    """SARSA on-policy TD control."""
    raise NotImplementedError


def q_learning_update(Q: np.ndarray, s: int, a: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """Q-learning off-policy update."""
    raise NotImplementedError


def td_lambda_return(rewards: np.ndarray, values: np.ndarray, gamma: float, lambda_param: float) -> np.ndarray:
    """Compute lambda-return for each timestep."""
    raise NotImplementedError


def eligibility_trace_update(trace: np.ndarray, state: int, gamma: float, lambda_param: float) -> np.ndarray:
    """Accumulating eligibility trace update."""
    raise NotImplementedError
