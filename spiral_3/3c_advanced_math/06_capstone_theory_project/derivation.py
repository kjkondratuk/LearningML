"""
Generalization Bound Computation

Compute generalization bounds for trained neural networks.

Reference: Neyshabur et al. 2017 (arXiv:1706.08947)
"""

import torch
import torch.nn as nn


def spectral_norm_bound(model: nn.Module, n_samples: int) -> float:
    """Compute norm-based generalization bound.

    Bound depends on product of spectral norms of weight matrices
    and the Frobenius norm of each layer.

    Args:
        model: Trained neural network.
        n_samples: Number of training samples.

    Returns:
        Generalization bound value.
    """
    raise NotImplementedError


def pac_bayes_bound(model: nn.Module, prior: nn.Module, n_samples: int, delta: float = 0.05) -> float:
    """PAC-Bayes generalization bound.

    Bound = train_error + sqrt((KL(posterior || prior) + log(2*sqrt(n)/delta)) / (2n))

    Args:
        model: Trained (posterior) model.
        prior: Prior model (e.g., initialization).
        n_samples: Number of training samples.
        delta: Confidence parameter.

    Returns:
        Generalization bound value.
    """
    raise NotImplementedError


def compression_bound(model: nn.Module, compressed_model: nn.Module, n_samples: int) -> float:
    """Compression-based generalization bound."""
    raise NotImplementedError
