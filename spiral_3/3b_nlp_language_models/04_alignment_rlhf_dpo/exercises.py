"""
Alignment Exercises: RLHF & DPO

Implement alignment components from:
    Ouyang et al. (2022) InstructGPT (arXiv:2203.02155)
    Rafailov et al. (2023) DPO (arXiv:2305.18290)
    Schulman et al. (2017) PPO (arXiv:1707.06347)
"""

import torch


def bradley_terry_probability(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry model: P(chosen > rejected) = sigma(r_chosen - r_rejected).

    Args:
        reward_chosen: Reward for preferred completion, shape (batch_size,).
        reward_rejected: Reward for rejected completion, shape (batch_size,).

    Returns:
        Probability that chosen is preferred, shape (batch_size,).
    """
    raise NotImplementedError


def reward_model_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Reward model training loss: -log sigma(r_chosen - r_rejected).

    Args:
        reward_chosen: Reward for preferred completion, shape (batch_size,).
        reward_rejected: Reward for rejected completion, shape (batch_size,).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def ppo_clipped_objective(
    ratio: torch.Tensor,
    advantage: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """PPO clipped surrogate objective.

    L = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)

    Args:
        ratio: pi_new / pi_old, shape (batch_size,).
        advantage: Advantage estimates, shape (batch_size,).
        epsilon: Clipping parameter.

    Returns:
        Scalar loss (negated for maximization).
    """
    raise NotImplementedError


def dpo_loss(
    pi_logprob_chosen: torch.Tensor,
    pi_logprob_rejected: torch.Tensor,
    ref_logprob_chosen: torch.Tensor,
    ref_logprob_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Direct Preference Optimization loss.

    L = -E[log sigma(beta * (log(pi_chosen/ref_chosen) - log(pi_rejected/ref_rejected)))]

    Args:
        pi_logprob_chosen: Log prob of chosen under policy, shape (batch_size,).
        pi_logprob_rejected: Log prob of rejected under policy, shape (batch_size,).
        ref_logprob_chosen: Log prob of chosen under reference, shape (batch_size,).
        ref_logprob_rejected: Log prob of rejected under reference, shape (batch_size,).
        beta: KL constraint strength.

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def kl_penalty(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """KL divergence penalty: E_pi[log pi - log ref].

    Args:
        pi_logprobs: Log probs under current policy, shape (batch_size, seq_len).
        ref_logprobs: Log probs under reference policy, shape (batch_size, seq_len).

    Returns:
        Scalar KL penalty (mean over batch and sequence).
    """
    raise NotImplementedError
