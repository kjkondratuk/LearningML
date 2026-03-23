"""Semantic Segmentation Exercises. References: Ronneberger et al. 2015 U-Net (arXiv:1505.04597)."""

import torch
import torch.nn as nn

def unet_encoder_block(in_channels: int, out_channels: int):
    """Conv-BN-ReLU-Conv-BN-ReLU-MaxPool."""
    raise NotImplementedError

def unet_decoder_block(in_channels: int, out_channels: int):
    """Upsample-Concat-Conv-BN-ReLU-Conv-BN-ReLU."""
    raise NotImplementedError

def dice_loss(predictions: torch.Tensor, targets: torch.Tensor):
    """Dice loss for segmentation."""
    raise NotImplementedError

def mean_iou(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int):
    """Mean IoU across classes."""
    raise NotImplementedError

