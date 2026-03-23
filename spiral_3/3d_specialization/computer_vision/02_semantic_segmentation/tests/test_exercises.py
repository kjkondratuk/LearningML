"""Tests for Semantic Segmentation."""
import pytest
import torch
from exercises import (
    unet_encoder_block, unet_decoder_block, dice_loss, mean_iou,
)

class TestDiceLoss:
    def test_perfect_prediction(self):
        pred = torch.ones(4, 1, 32, 32)
        target = torch.ones(4, 1, 32, 32)
        loss = dice_loss(pred, target)
        assert loss.item() < 0.01
    def test_worst_prediction(self):
        pred = torch.zeros(4, 1, 32, 32)
        target = torch.ones(4, 1, 32, 32)
        loss = dice_loss(pred, target)
        assert loss.item() > 0.9
