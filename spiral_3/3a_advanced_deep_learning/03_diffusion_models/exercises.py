"""
Diffusion Model Exercises

Implement core DDPM/DDIM components from:
    Ho et al. (2020), "Denoising Diffusion Probabilistic Models" (arXiv:2006.11239)
    Song et al. (2020), "Denoising Diffusion Implicit Models" (arXiv:2010.02502)
"""

import torch
import torch.nn as nn


def forward_diffusion_step(
    x_0: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t from q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I).

    Args:
        x_0: Clean data, shape (batch_size, *data_shape).
        t: Timesteps, shape (batch_size,), integers in [0, T-1].
        noise_schedule: Dict with keys 'alpha_bars' of shape (T,).

    Returns:
        Tuple of (x_t, epsilon) where epsilon is the sampled noise.
    """
    raise NotImplementedError


def compute_noise_schedule(
    T: int,
    schedule_type: str = "linear",
) -> dict[str, torch.Tensor]:
    """Compute the noise schedule for T diffusion steps.

    For linear schedule:
        betas linearly from 1e-4 to 0.02
    For cosine schedule:
        alpha_bar_t = f(t)/f(0), where f(t) = cos((t/T + s)/(1+s) * pi/2)^2

    Args:
        T: Number of diffusion steps.
        schedule_type: "linear" or "cosine".

    Returns:
        Dict with keys: 'betas', 'alphas', 'alpha_bars' -- each shape (T,).
    """
    raise NotImplementedError


def reverse_step(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    noise_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Single reverse diffusion step: sample x_{t-1} from p_theta(x_{t-1} | x_t).

    Uses the model to predict noise, then applies the reverse Gaussian transition.

    Args:
        model: Noise prediction model, takes (x_t, t) and returns predicted noise.
        x_t: Noisy data at timestep t, shape (batch_size, *data_shape).
        t: Current timestep, shape (batch_size,).
        noise_schedule: Dict with 'betas', 'alphas', 'alpha_bars'.

    Returns:
        x_{t-1}, shape (batch_size, *data_shape).
    """
    raise NotImplementedError


def ddpm_loss(
    model: nn.Module,
    x_0: torch.Tensor,
    noise_schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Simplified DDPM loss: MSE between predicted and actual noise.

    L_simple = E_{t, x_0, epsilon} [||epsilon - epsilon_theta(x_t, t)||^2]

    Args:
        model: Noise prediction model.
        x_0: Clean data, shape (batch_size, *data_shape).
        noise_schedule: Dict with noise schedule tensors.

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def ddim_sample(
    model: nn.Module,
    x_T: torch.Tensor,
    noise_schedule: dict[str, torch.Tensor],
    steps: int = 50,
) -> torch.Tensor:
    """Deterministic DDIM sampling with fewer steps than DDPM.

    Reference: Song et al. 2020, arXiv:2010.02502

    Args:
        model: Trained noise prediction model.
        x_T: Initial noise, shape (batch_size, *data_shape).
        noise_schedule: Dict with noise schedule tensors.
        steps: Number of denoising steps (can be much less than T).

    Returns:
        Denoised samples x_0, shape (batch_size, *data_shape).
    """
    raise NotImplementedError
