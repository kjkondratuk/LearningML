"""Tests for BO exercises."""
import pytest
import torch
from exercises import (
    expected_improvement, upper_confidence_bound,
)

class TestEI:
    def test_nonnegative(self):
        mu = torch.tensor([1.0, 2.0, 0.5])
        sigma = torch.tensor([0.5, 0.5, 0.5])
        ei = expected_improvement(mu, sigma, best_so_far=1.5)
        assert (ei >= -1e-6).all()

    def test_zero_when_no_improvement_possible(self):
        mu = torch.tensor([0.0])
        sigma = torch.tensor([0.0])
        ei = expected_improvement(mu, sigma, best_so_far=1.0)
        assert abs(ei.item()) < 1e-6

class TestUCB:
    def test_higher_beta_wider_search(self):
        mu = torch.tensor([1.0])
        sigma = torch.tensor([1.0])
        ucb_low = upper_confidence_bound(mu, sigma, beta=1.0)
        ucb_high = upper_confidence_bound(mu, sigma, beta=3.0)
        assert ucb_high > ucb_low
