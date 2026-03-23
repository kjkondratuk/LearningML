"""
Variational Autoencoder Exercises

Implement core VAE components from:
    Kingma & Welling (2013), "Auto-Encoding Variational Bayes" (arXiv:1312.6114)

All functions operate on PyTorch tensors.
"""

import torch


def compute_kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between q(z|x) = N(mu, sigma^2 I) and p(z) = N(0, I).

    Uses the closed-form expression:
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean of the encoder distribution, shape (batch_size, latent_dim).
        log_var: Log variance of the encoder distribution, shape (batch_size, latent_dim).

    Returns:
        KL divergence, shape (batch_size,). Per-sample KL summed over latent dimensions.
    """
    raise NotImplementedError


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Sample z using the reparameterization trick: z = mu + sigma * epsilon.

    This enables gradient flow through the sampling operation by separating
    the stochastic component (epsilon ~ N(0, I)) from the learned parameters.

    Args:
        mu: Mean of the encoder distribution, shape (batch_size, latent_dim).
        log_var: Log variance of the encoder distribution, shape (batch_size, latent_dim).

    Returns:
        Sampled z, shape (batch_size, latent_dim).
    """
    raise NotImplementedError


def compute_elbo(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the Evidence Lower Bound (ELBO).

    ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
         = -reconstruction_loss - kl_divergence

    Uses binary cross-entropy for reconstruction (assuming Bernoulli decoder).

    Args:
        x: Original input, shape (batch_size, input_dim).
        x_recon: Reconstructed input (sigmoid outputs), shape (batch_size, input_dim).
        mu: Encoder mean, shape (batch_size, latent_dim).
        log_var: Encoder log variance, shape (batch_size, latent_dim).

    Returns:
        Tuple of (elbo, recon_loss, kl_loss), each shape (batch_size,).
        elbo is the negative loss (higher is better).
        recon_loss and kl_loss are positive values to be minimized.
    """
    raise NotImplementedError


def beta_vae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Beta-VAE objective for disentangled representations.

    loss = recon_loss + beta * kl_loss

    When beta=1, this is the standard VAE. When beta>1, the KL term is upweighted,
    encouraging disentanglement at the cost of reconstruction quality.

    Reference: Higgins et al. 2017, "beta-VAE" (arXiv:1804.03599)

    Args:
        x: Original input, shape (batch_size, input_dim).
        x_recon: Reconstructed input, shape (batch_size, input_dim).
        mu: Encoder mean, shape (batch_size, latent_dim).
        log_var: Encoder log variance, shape (batch_size, latent_dim).
        beta: Weight on the KL term. beta=1 gives standard VAE.

    Returns:
        Scalar loss (mean over batch).
    """
    raise NotImplementedError


def log_importance_weights(
    z: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """Compute log importance weights for Importance-Weighted Autoencoders (IWAE).

    The importance weight is: w = p(z) / q(z|x)

    log w = log p(z) - log q(z|x)
          = log N(z; 0, I) - log N(z; mu, sigma^2 I)

    Reference: Burda et al. 2015, "Importance Weighted Autoencoders" (arXiv:1509.00519)

    Args:
        z: Sampled latent variables, shape (batch_size, latent_dim).
        mu: Encoder mean, shape (batch_size, latent_dim).
        log_var: Encoder log variance, shape (batch_size, latent_dim).

    Returns:
        Log importance weights, shape (batch_size,).
    """
    raise NotImplementedError
