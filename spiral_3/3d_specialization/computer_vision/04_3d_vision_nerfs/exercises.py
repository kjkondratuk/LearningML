"""3D Vision & NeRFs Exercises. References: Mildenhall et al. 2020 NeRF (arXiv:2003.08934), Kerbl et al. 2023 3DGS (arXiv:2308.14737)."""

import torch
import torch.nn as nn

def ray_generation(camera_intrinsics: torch.Tensor, camera_pose: torch.Tensor, H: int, W: int):
    """Generate rays for all pixels."""
    raise NotImplementedError

def volume_rendering(densities: torch.Tensor, colors: torch.Tensor, t_values: torch.Tensor):
    """NeRF volume rendering equation."""
    raise NotImplementedError

def positional_encoding_nerf(x: torch.Tensor, num_frequencies: int):
    """NeRF positional encoding."""
    raise NotImplementedError

def nerf_mlp_forward(positions: torch.Tensor, directions: torch.Tensor, model):
    """NeRF two-stage MLP."""
    raise NotImplementedError

