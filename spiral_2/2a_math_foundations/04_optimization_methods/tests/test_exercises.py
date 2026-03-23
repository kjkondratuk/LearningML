"""
Tests for Optimization Methods exercises.
"""

import numpy as np
import pytest

from exercises import (
    sgd,
    sgd_momentum,
    nesterov_momentum,
    adagrad,
    rmsprop,
    adam,
    lbfgs_step,
    line_search,
)


def quadratic_grad(x):
    """Gradient of f(x) = 0.5 * x^T x."""
    return x.copy()


def ill_conditioned_grad(x):
    """Gradient of f(x) = 0.5 * (x[0]^2 + 100*x[1]^2)."""
    return np.array([x[0], 100 * x[1]])


# ── All optimizers converge on convex quadratic ──────────────────────────────

class TestConvergence:
    """All optimizers should converge to the minimum of a convex quadratic."""

    def test_sgd_converges(self):
        traj = sgd(quadratic_grad, np.array([5.0, 5.0]), lr=0.1, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)

    def test_sgd_momentum_converges(self):
        traj = sgd_momentum(quadratic_grad, np.array([5.0, 5.0]),
                            lr=0.1, momentum=0.9, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)

    def test_nesterov_converges(self):
        traj = nesterov_momentum(quadratic_grad, np.array([5.0, 5.0]),
                                  lr=0.1, momentum=0.9, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)

    def test_adagrad_converges(self):
        traj = adagrad(quadratic_grad, np.array([5.0, 5.0]),
                       lr=1.0, n_steps=200)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=0.1)

    def test_rmsprop_converges(self):
        traj = rmsprop(quadratic_grad, np.array([5.0, 5.0]),
                       lr=0.1, decay=0.9, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)

    def test_adam_converges(self):
        traj = adam(quadratic_grad, np.array([5.0, 5.0]),
                    lr=0.1, beta1=0.9, beta2=0.999, n_steps=200)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)


# ── Adam faster than SGD on ill-conditioned problems ─────────────────────────

class TestAdamVsSGD:
    def test_adam_faster_on_ill_conditioned(self):
        x0 = np.array([5.0, 5.0])
        f = lambda x: 0.5 * (x[0] ** 2 + 100 * x[1] ** 2)

        traj_sgd = sgd(ill_conditioned_grad, x0.copy(), lr=0.01, n_steps=500)
        traj_adam = adam(ill_conditioned_grad, x0.copy(),
                        lr=0.1, beta1=0.9, beta2=0.999, n_steps=500)
        assert f(traj_adam[-1]) < f(traj_sgd[-1])


# ── AdaGrad learning rate decreases ──────────────────────────────────────────

class TestAdaGradDecay:
    def test_effective_lr_decreases(self):
        """AdaGrad's accumulated squared gradient only grows."""
        traj = adagrad(quadratic_grad, np.array([10.0, 10.0]),
                       lr=1.0, n_steps=50)
        # Steps should get smaller over time
        early_step = np.linalg.norm(traj[2] - traj[1])
        late_step = np.linalg.norm(traj[-1] - traj[-2])
        assert late_step < early_step


# ── Nesterov faster than classical momentum ──────────────────────────────────

class TestNesterovVsMomentum:
    def test_nesterov_fewer_steps(self):
        x0 = np.array([5.0, 5.0])
        f = lambda x: 0.5 * x @ x

        traj_mom = sgd_momentum(quadratic_grad, x0.copy(),
                                lr=0.1, momentum=0.9, n_steps=50)
        traj_nes = nesterov_momentum(quadratic_grad, x0.copy(),
                                      lr=0.1, momentum=0.9, n_steps=50)
        assert f(traj_nes[-1]) <= f(traj_mom[-1]) * 1.1  # at least comparable


# ── L-BFGS ────────────────────────────────────────────────────────────────────

class TestLBFGS:
    def test_direction_is_descent(self):
        """L-BFGS direction should be a descent direction."""
        x = np.array([3.0, 4.0])
        g = quadratic_grad(x)
        # Empty history => should fall back to steepest descent
        direction = lbfgs_step(quadratic_grad, x, history=[], m=5)
        # Descent direction: dot(direction, gradient) < 0
        assert np.dot(direction, g) < 0

    def test_with_history(self):
        """With some history, L-BFGS should still produce a descent direction."""
        x0 = np.array([5.0, 5.0])
        x1 = np.array([4.0, 4.0])
        g0 = quadratic_grad(x0)
        g1 = quadratic_grad(x1)
        s = x1 - x0
        y = g1 - g0
        direction = lbfgs_step(quadratic_grad, x1, history=[(s, y)], m=5)
        assert np.dot(direction, g1) < 0


# ── Line Search ───────────────────────────────────────────────────────────────

class TestLineSearch:
    def test_armijo_condition(self):
        f = lambda x: 0.5 * np.dot(x, x)
        grad_f = lambda x: x.copy()
        x = np.array([10.0, 10.0])
        d = -grad_f(x)
        alpha = line_search(f, grad_f, x, d)
        # Armijo: f(x + alpha*d) <= f(x) + c * alpha * grad^T d
        assert f(x + alpha * d) <= f(x) + 1e-4 * alpha * np.dot(grad_f(x), d)

    def test_positive_step_size(self):
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        grad_f = lambda x: 2 * x
        x = np.array([5.0, 5.0])
        d = -grad_f(x)
        alpha = line_search(f, grad_f, x, d)
        assert alpha > 0
