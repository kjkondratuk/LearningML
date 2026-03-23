"""
Tests for the CNN mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    build_mnist_cnn,
)


class TestBuildCNN:
    def test_model_has_correct_keys(self):
        model = build_mnist_cnn()
        assert isinstance(model, dict)
        # Should have weights for conv and fc layers
        assert any("conv" in k or "fc" in k or "W" in k for k in model.keys())
