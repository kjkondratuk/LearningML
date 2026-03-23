"""
Tests for DL Optimization exercises.
"""

import numpy as np
import pytest

from exercises import (
    lr_warmup,
    cosine_annealing_lr,
    adamw,
    gradient_clipping,
)


class TestLRWarmup:
    def test_zero_at_start(self):
        assert lr_warmup(0.001, 1000, 0) == pytest.approx(0.0, abs=1e-10)

    def test_full_at_end(self):
        assert lr_warmup(0.001, 1000, 1000) == pytest.approx(0.001, abs=1e-10)

    def test_linear_midpoint(self):
        assert lr_warmup(0.001, 1000, 500) == pytest.approx(0.0005, abs=1e-10)


class TestCosineAnnealing:
    def test_start(self):
        assert cosine_annealing_lr(0.1, 0, 100) == pytest.approx(0.1, abs=1e-10)

    def test_end(self):
        lr = cosine_annealing_lr(0.1, 100, 100)
        assert lr < 1e-6  # Should be ~0

    def test_midpoint(self):
        lr = cosine_annealing_lr(0.1, 50, 100)
        assert 0.04 < lr < 0.06  # Should be ~0.05


class TestAdamW:
    def test_zero_weight_decay_matches_adam(self):
        np.random.seed(42)
        params = [np.array([5.0, 5.0])]
        grad_fn = lambda p: [2 * p[0]]  # gradient of x^2 + y^2
        result_wd0 = adamw(grad_fn, [p.copy() for p in params],
                           lr=0.1, beta1=0.9, beta2=0.999,
                           weight_decay=0.0, n_steps=100)
        # Should converge near zero
        np.testing.assert_allclose(result_wd0[0], [0, 0], atol=0.5)


class TestGradientClipping:
    def test_clips_large_norm(self):
        grads = [np.array([3.0, 4.0])]  # norm = 5
        clipped = gradient_clipping(grads, max_norm=2.5)
        norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped))
        np.testing.assert_allclose(norm, 2.5, atol=1e-6)

    def test_no_clip_small_norm(self):
        grads = [np.array([0.1, 0.2])]
        clipped = gradient_clipping(grads, max_norm=10.0)
        np.testing.assert_array_equal(clipped[0], grads[0])
