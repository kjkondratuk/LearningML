"""Tests for MDP exercises."""

import pytest
import numpy as np

from exercises import (
    bellman_operator, policy_iteration, value_iteration, verify_contraction,
)


class TestValueIteration:
    def test_converges(self):
        """Value iteration should converge on a simple gridworld."""
        S, A = 4, 2
        P = np.random.dirichlet(np.ones(S), size=(S, A))
        R = np.random.randn(S, A, S)
        V, pi = value_iteration(P, R, gamma=0.9)
        assert V.shape == (S,)
        assert pi.shape == (S,)

    def test_matches_policy_iteration(self):
        """Both algorithms should find the same optimal policy."""
        S, A = 4, 2
        np.random.seed(42)
        P = np.random.dirichlet(np.ones(S), size=(S, A))
        R = np.random.randn(S, A, S)
        V_vi, pi_vi = value_iteration(P, R, gamma=0.9)
        V_pi, pi_pi = policy_iteration(P, R, gamma=0.9)
        np.testing.assert_allclose(V_vi, V_pi, atol=1e-4)


class TestContraction:
    def test_bellman_is_contraction(self):
        S, A = 4, 2
        np.random.seed(42)
        P = np.random.dirichlet(np.ones(S), size=(S, A))
        R = np.random.randn(S, A, S)
        gamma = 0.9
        V1 = np.random.randn(S)
        V2 = np.random.randn(S)
        T_V1 = bellman_operator(V1, P, R, gamma)
        T_V2 = bellman_operator(V2, P, R, gamma)
        assert verify_contraction(V1, V2, T_V1, T_V2, gamma)
