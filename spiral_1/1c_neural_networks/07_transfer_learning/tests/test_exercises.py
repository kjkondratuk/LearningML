"""
Tests for Module 07: Transfer Learning

Run: pytest tests/test_exercises.py -v
"""

import torch
import torch.nn as nn
import pytest

from exercises import (
    load_pretrained_resnet,
    freeze_backbone,
    count_trainable_params,
    unfreeze_last_n_layers,
    fine_tune,
)


# ---------------------------------------------------------------------------
# load_pretrained_resnet
# ---------------------------------------------------------------------------

class TestLoadPretrainedResnet:
    @pytest.mark.slow
    def test_returns_nn_module(self):
        model = load_pretrained_resnet()
        assert isinstance(model, nn.Module)

    @pytest.mark.slow
    def test_has_fc_layer(self):
        model = load_pretrained_resnet()
        assert hasattr(model, 'fc')

    @pytest.mark.slow
    def test_original_fc_has_1000_classes(self):
        model = load_pretrained_resnet()
        assert model.fc.out_features == 1000


# ---------------------------------------------------------------------------
# freeze_backbone
# ---------------------------------------------------------------------------

class TestFreezeBackbone:
    @pytest.mark.slow
    def test_all_backbone_frozen(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                assert not param.requires_grad, f"{name} should be frozen"

    @pytest.mark.slow
    def test_fc_unfrozen(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        for name, param in model.named_parameters():
            if 'fc' in name:
                assert param.requires_grad, f"{name} should be trainable"

    @pytest.mark.slow
    def test_output_classes(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        assert model.fc.out_features == 10


# ---------------------------------------------------------------------------
# count_trainable_params
# ---------------------------------------------------------------------------

class TestCountTrainableParams:
    def test_all_trainable(self):
        model = nn.Linear(10, 5)  # 55 params
        assert count_trainable_params(model) == 55

    def test_none_trainable(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        assert count_trainable_params(model) == 0

    @pytest.mark.slow
    def test_frozen_resnet(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        trainable = count_trainable_params(model)
        # Only the FC layer: 512 * 10 + 10 = 5130
        assert trainable == 512 * 10 + 10


# ---------------------------------------------------------------------------
# unfreeze_last_n_layers
# ---------------------------------------------------------------------------

class TestUnfreezeLastNLayers:
    @pytest.mark.slow
    def test_more_trainable_after_unfreeze(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        frozen_count = count_trainable_params(model)
        model = unfreeze_last_n_layers(model, n=2)
        unfrozen_count = count_trainable_params(model)
        assert unfrozen_count > frozen_count


# ---------------------------------------------------------------------------
# fine_tune (slow -- requires CIFAR-10 download)
# ---------------------------------------------------------------------------

class TestFineTune:
    @pytest.mark.slow
    def test_returns_expected_keys(self):
        model = load_pretrained_resnet()
        model = freeze_backbone(model, num_classes=10)
        result = fine_tune(model, num_classes=10, epochs=1, batch_size=32)
        assert "train_losses" in result
        assert "test_accuracy" in result
