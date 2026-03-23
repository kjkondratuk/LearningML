"""Policy Gradient Exercises. References: Williams 1992, Schulman et al. 2015 (arXiv:1506.02438)."""

import torch


def reinforce_loss(log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """REINFORCE loss: -sum(log_prob * G_t).

    Args:
        log_probs: Log probabilities of taken actions, shape (T,).
        returns: Discounted returns, shape (T,).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}.

    Args:
        rewards: List of rewards for each timestep.
        gamma: Discount factor.

    Returns:
        Tensor of returns, shape (T,).
    """
    raise NotImplementedError


def generalized_advantage_estimation(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float
) -> torch.Tensor:
    """GAE: A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}.

    where delta_t = r_t + gamma*V(s_{t+1}) - V(s_t).

    Args:
        rewards: shape (T,).
        values: shape (T+1,) including terminal value.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        Advantages, shape (T,).
    """
    raise NotImplementedError


def actor_critic_loss(
    log_probs: torch.Tensor, advantages: torch.Tensor,
    values: torch.Tensor, returns: torch.Tensor, value_coeff: float = 0.5
) -> torch.Tensor:
    """Combined actor-critic loss.

    L = -mean(log_prob * advantage) + value_coeff * mean((V - G)^2)
    """
    raise NotImplementedError


def entropy_bonus(action_probs: torch.Tensor) -> torch.Tensor:
    """Entropy of action distribution: -sum(p * log p)."""
    raise NotImplementedError
