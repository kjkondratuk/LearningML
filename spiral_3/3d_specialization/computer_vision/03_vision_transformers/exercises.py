"""Vision Transformers Exercises. References: Dosovitskiy et al. 2020 ViT (arXiv:2010.11929)."""

import torch
import torch.nn as nn

def image_to_patches(image: torch.Tensor, patch_size: int):
    """Split image into patches."""
    raise NotImplementedError

def patch_embedding(patches: torch.Tensor, embedding_dim: int):
    """Linear projection of patches."""
    raise NotImplementedError

def vit_cls_token(patch_embeddings: torch.Tensor):
    """Prepend CLS token and add position embeddings."""
    raise NotImplementedError

def vit_forward(image: torch.Tensor, patch_size: int, num_layers: int, num_heads: int, d_model: int):
    """Full ViT forward pass."""
    raise NotImplementedError

