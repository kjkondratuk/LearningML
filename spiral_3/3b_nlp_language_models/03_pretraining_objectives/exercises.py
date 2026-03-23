"""
Pretraining Objective Exercises

Implement pretraining losses from:
    Devlin et al. (2018) BERT (arXiv:1810.04805)
    Radford et al. (2019) GPT-2
    Clark et al. (2020) ELECTRA (arXiv:2003.10555)
"""

import torch


def masked_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_positions: torch.Tensor,
) -> torch.Tensor:
    """BERT-style masked language model loss.

    Cross-entropy only at masked positions.

    Args:
        logits: Model output logits, shape (batch_size, seq_len, vocab_size).
        labels: True token ids, shape (batch_size, seq_len).
        mask_positions: Boolean mask, shape (batch_size, seq_len). True = masked.

    Returns:
        Scalar loss (mean over masked tokens).
    """
    raise NotImplementedError


def causal_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """GPT-style causal language model loss: next-token prediction.

    L = -mean_t log p(x_t | x_{<t})

    Args:
        logits: Model output, shape (batch_size, seq_len, vocab_size).
                logits[:, t, :] predicts token at position t+1.
        labels: Target token ids, shape (batch_size, seq_len).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def span_corruption_objective(
    input_ids: torch.Tensor,
    sentinel_tokens: dict[int, list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """T5-style span corruption: replace spans with sentinel tokens.

    Given input tokens, replace randomly selected spans with sentinel tokens
    and produce the target sequence (sentinels followed by the replaced spans).

    Args:
        input_ids: Original token ids, shape (seq_len,).
        sentinel_tokens: Dict mapping sentinel_id -> list of replaced token ids.

    Returns:
        Tuple of (corrupted_input, target):
            corrupted_input: Input with spans replaced by sentinels.
            target: Sentinels followed by the original spans.
    """
    raise NotImplementedError


def electra_loss(
    generator_logits: torch.Tensor,
    discriminator_logits: torch.Tensor,
    original_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ELECTRA: generator MLM loss + discriminator replaced token detection.

    Generator loss: standard MLM cross-entropy.
    Discriminator loss: binary cross-entropy per token (replaced or original).

    Args:
        generator_logits: Generator output, shape (batch_size, seq_len, vocab_size).
        discriminator_logits: Discriminator output, shape (batch_size, seq_len, 1).
        original_ids: Original token ids, shape (batch_size, seq_len).
        corrupted_ids: Corrupted token ids (from generator), shape (batch_size, seq_len).

    Returns:
        Tuple of (generator_loss, discriminator_loss).
    """
    raise NotImplementedError


def perplexity(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute perplexity from log probabilities.

    PPL = exp(-1/N * sum log p(x_i))

    Args:
        log_probs: Log probabilities of each token, shape (num_tokens,).

    Returns:
        Scalar perplexity.
    """
    raise NotImplementedError
