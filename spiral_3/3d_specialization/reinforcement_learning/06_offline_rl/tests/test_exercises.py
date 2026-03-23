"""Tests for Offline RL exercises."""
import pytest
import torch
from exercises import (
    iql_value_loss, pessimistic_value_estimate,
)

class TestIQL:
    def test_tau_half_is_mean(self):
        """With tau=0.5, IQL reduces to standard MSE (symmetric loss)."""
        Q = torch.tensor([1.0, 2.0, 3.0])
        V = torch.tensor([1.5, 1.5, 1.5])
        loss = iql_value_loss(Q, V, tau=0.5)
        mse = ((Q - V) ** 2).mean()
        assert torch.allclose(loss, mse * 0.5, atol=0.1)  # Factor may differ

class TestPessimistic:
    def test_less_than_mean(self):
        Q_ens = torch.randn(5, 10)  # 5 ensemble, 10 actions
        actions = torch.arange(10)
        pess = pessimistic_value_estimate(Q_ens, actions, beta=1.0)
        mean = Q_ens.mean(dim=0)
        assert (pess <= mean + 1e-5).all()
