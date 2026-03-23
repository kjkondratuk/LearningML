"""Object Detection Exercises. References: Redmon et al. 2015 YOLO (arXiv:1506.02640), Carion et al. 2020 DETR (arXiv:2005.12872)."""

import torch
import torch.nn as nn

def compute_iou(box1: torch.Tensor, box2: torch.Tensor):
    """Intersection over Union for bounding boxes."""
    raise NotImplementedError

def non_max_suppression(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
    """NMS to remove duplicate detections."""
    raise NotImplementedError

def anchor_box_generation(feature_map_size: int, aspect_ratios: list, scales: list):
    """Generate anchor boxes."""
    raise NotImplementedError

def yolo_loss(predictions: torch.Tensor, targets: torch.Tensor, lambda_coord: float, lambda_noobj: float):
    """YOLO v1 loss function."""
    raise NotImplementedError

def detr_bipartite_matching(predicted: torch.Tensor, targets: torch.Tensor, cost_fn: callable):
    """Hungarian matching for DETR."""
    raise NotImplementedError

