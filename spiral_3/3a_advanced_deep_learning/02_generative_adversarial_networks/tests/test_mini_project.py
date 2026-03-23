"""Tests for WGAN-GP mini-project."""

import pytest
import torch

from mini_project import (
    Critic,
    Generator,
)


class TestGenerator:
    def test_output_shape(self):
        gen = Generator(latent_dim=128, img_channels=3)
        z = torch.randn(4, 128)
        out = gen(z)
        assert out.shape == (4, 3, 32, 32), f"Expected (4,3,32,32), got {out.shape}"

    def test_output_range(self):
        gen = Generator(latent_dim=128, img_channels=3)
        z = torch.randn(4, 128)
        out = gen(z)
        assert out.min() >= -1.0 - 1e-5 and out.max() <= 1.0 + 1e-5


class TestCritic:
    def test_output_shape(self):
        critic = Critic(img_channels=3)
        x = torch.randn(4, 3, 32, 32)
        out = critic(x)
        assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"
