"""
Tests for Module 04: Optimization Intuition

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    is_convex_1d,
    gradient_descent_with_momentum,
    compare_optimizers,
    learning_rate_schedule,
    find_minimum_2d,
)


# ---------------------------------------------------------------------------
# is_convex_1d
# ---------------------------------------------------------------------------

class TestIsConvex1d:
    def test_x_squared_is_convex(self):
        assert is_convex_1d(lambda x: x ** 2, (-10, 10)) is True

    def test_negative_x_squared_is_not_convex(self):
        assert is_convex_1d(lambda x: -(x ** 2), (-10, 10)) is False

    def test_exp_is_convex(self):
        assert is_convex_1d(np.exp, (-5, 5)) is True

    def test_sin_is_not_convex(self):
        assert is_convex_1d(np.sin, (0, 2 * np.pi)) is False

    def test_abs_is_convex(self):
        assert is_convex_1d(abs, (-5, 5)) is True


# ---------------------------------------------------------------------------
# gradient_descent_with_momentum
# ---------------------------------------------------------------------------

class TestGradientDescentWithMomentum:
    def test_quadratic_bowl(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        x_final, _ = gradient_descent_with_momentum(
            f, grad_f, x0=np.array([5.0, -3.0]), lr=0.01, momentum=0.9
        )
        np.testing.assert_allclose(x_final, [0.0, 0.0], atol=1e-3)

    def test_momentum_helps_narrow_valley(self):
        """Momentum should converge faster in elongated valleys."""
        f = lambda v: 100 * v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([200 * v[0], 2 * v[1]])
        x0 = np.array([1.0, 1.0])

        # Without momentum (simulate by setting momentum=0)
        _, hist_no_mom = gradient_descent_with_momentum(
            f, grad_f, x0.copy(), lr=0.005, momentum=0.0, max_iter=200
        )
        # With momentum
        _, hist_mom = gradient_descent_with_momentum(
            f, grad_f, x0.copy(), lr=0.005, momentum=0.9, max_iter=200
        )
        loss_no_mom = f(hist_no_mom[-1])
        loss_mom = f(hist_mom[-1])
        assert loss_mom < loss_no_mom

    def test_history_recorded(self):
        f = lambda v: v[0] ** 2
        grad_f = lambda v: np.array([2 * v[0]])
        _, history = gradient_descent_with_momentum(
            f, grad_f, x0=np.array([5.0]), lr=0.1, max_iter=10
        )
        assert len(history) > 0


# ---------------------------------------------------------------------------
# compare_optimizers
# ---------------------------------------------------------------------------

class TestCompareOptimizers:
    def test_returns_three_keys(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        result = compare_optimizers(f, grad_f, x0=np.array([5.0, 5.0]),
                                    max_iter=50)
        assert "vanilla" in result
        assert "momentum" in result
        assert "adam" in result

    def test_all_losses_decrease(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        result = compare_optimizers(f, grad_f, x0=np.array([5.0, 5.0]),
                                    lr=0.01, max_iter=100)
        for name, losses in result.items():
            assert losses[-1] < losses[0], f"{name} did not decrease loss"

    def test_loss_history_length(self):
        f = lambda v: v[0] ** 2
        grad_f = lambda v: np.array([2 * v[0]])
        result = compare_optimizers(f, grad_f, x0=np.array([3.0]),
                                    max_iter=50)
        for name, losses in result.items():
            assert len(losses) == 50, f"{name} has wrong history length"


# ---------------------------------------------------------------------------
# learning_rate_schedule
# ---------------------------------------------------------------------------

class TestLearningRateSchedule:
    def test_constant(self):
        lrs = learning_rate_schedule("constant", 0.01, 100)
        assert len(lrs) == 100
        assert all(lr == pytest.approx(0.01) for lr in lrs)

    def test_step_starts_at_initial(self):
        lrs = learning_rate_schedule("step", 0.1, 90)
        assert lrs[0] == pytest.approx(0.1)

    def test_step_decays(self):
        lrs = learning_rate_schedule("step", 0.1, 90)
        # After 30 steps (90//3), lr should halve
        assert lrs[30] == pytest.approx(0.05)
        assert lrs[60] == pytest.approx(0.025)

    def test_cosine_starts_at_initial(self):
        lrs = learning_rate_schedule("cosine", 0.1, 100)
        assert lrs[0] == pytest.approx(0.1, abs=1e-6)

    def test_cosine_ends_near_zero(self):
        lrs = learning_rate_schedule("cosine", 0.1, 100)
        assert lrs[-1] < 0.001

    def test_cosine_monotonically_decreasing(self):
        lrs = learning_rate_schedule("cosine", 0.1, 100)
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-10

    def test_length(self):
        for stype in ["constant", "step", "cosine"]:
            lrs = learning_rate_schedule(stype, 0.01, 50)
            assert len(lrs) == 50


# ---------------------------------------------------------------------------
# find_minimum_2d
# ---------------------------------------------------------------------------

class TestFindMinimum2d:
    def test_simple_bowl(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        x_min, f_min, traj = find_minimum_2d(
            f, grad_f, x0=np.array([3.0, -4.0]), method="momentum"
        )
        assert f_min < 1e-4
        np.testing.assert_allclose(x_min, [0.0, 0.0], atol=0.01)

    def test_returns_trajectory(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        _, _, traj = find_minimum_2d(
            f, grad_f, x0=np.array([1.0, 1.0]), max_iter=10
        )
        assert len(traj) > 0
        assert traj[0].shape == (2,)

    def test_adam_method(self):
        f = lambda v: (v[0] - 1) ** 2 + (v[1] + 1) ** 2
        grad_f = lambda v: np.array([2 * (v[0] - 1), 2 * (v[1] + 1)])
        x_min, f_min, _ = find_minimum_2d(
            f, grad_f, x0=np.array([5.0, 5.0]), method="adam", lr=0.1
        )
        assert f_min < 0.01
