"""Tests for TD exercises."""
import pytest
import numpy as np
from exercises import (
    td_zero_update, sarsa_update, q_learning_update, td_lambda_return,
)

class TestTDZero:
    def test_update_direction(self):
        """If r + gamma*V(s') > V(s), V(s) should increase."""
        V = np.zeros(5)
        V[1] = 1.0
        V_new = td_zero_update(V.copy(), state=0, next_state=1, reward=0.0, alpha=0.1, gamma=0.9)
        assert V_new[0] > V[0]

class TestQLearning:
    def test_max_over_actions(self):
        """Q-learning uses max_a Q(s',a), not the actual next action."""
        Q = np.zeros((3, 2))
        Q[1, 0] = 5.0
        Q[1, 1] = 10.0
        Q_new = q_learning_update(Q.copy(), s=0, a=0, r=1.0, s_next=1, alpha=0.1, gamma=0.9)
        expected = 0.1 * (1.0 + 0.9 * 10.0)
        assert abs(Q_new[0, 0] - expected) < 1e-6

class TestTDLambda:
    def test_lambda_zero_is_td(self):
        """TD(lambda=0) should give one-step TD return."""
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 1.5, 0.0])  # Include terminal
        gamma = 0.9
        returns = td_lambda_return(rewards, values, gamma, lambda_param=0.0)
        # G_0 = r_0 + gamma * V(1) = 1.0 + 0.9 * 1.0 = 1.9
        assert abs(returns[0] - 1.9) < 1e-5
