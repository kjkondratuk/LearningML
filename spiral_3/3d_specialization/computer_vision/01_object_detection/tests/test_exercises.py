"""Tests for Object Detection."""
import pytest
import torch
from exercises import (
    compute_iou, non_max_suppression, anchor_box_generation, yolo_loss, detr_bipartite_matching,
)

class TestIoU:
    def test_identical_boxes(self):
        box = torch.tensor([0.0, 0.0, 1.0, 1.0])
        iou = compute_iou(box, box)
        assert abs(iou.item() - 1.0) < 1e-5
    def test_non_overlapping(self):
        b1 = torch.tensor([0.0, 0.0, 1.0, 1.0])
        b2 = torch.tensor([2.0, 2.0, 3.0, 3.0])
        iou = compute_iou(b1, b2)
        assert abs(iou.item()) < 1e-5
    def test_in_range(self):
        b1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
        b2 = torch.tensor([1.0, 1.0, 3.0, 3.0])
        iou = compute_iou(b1, b2)
        assert 0 <= iou.item() <= 1
