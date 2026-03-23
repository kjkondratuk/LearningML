"""
Word Embedding Exercises

Implement embedding algorithms from:
    Mikolov et al. (2013) Word2Vec (arXiv:1301.3781)
    Pennington et al. (2014) GloVe
"""

import torch


def skipgram_forward(
    center_word: torch.Tensor,
    context_words: torch.Tensor,
    embeddings_in: torch.Tensor,
    embeddings_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for Skip-gram: compute scores for positive and negative pairs.

    Args:
        center_word: Center word indices, shape (batch_size,).
        context_words: Context word indices (positive + negative),
                      shape (batch_size, 1 + num_negatives).
        embeddings_in: Input embedding matrix, shape (vocab_size, embed_dim).
        embeddings_out: Output embedding matrix, shape (vocab_size, embed_dim).

    Returns:
        Tuple of (positive_scores, negative_scores):
            positive_scores: shape (batch_size, 1)
            negative_scores: shape (batch_size, num_negatives)
    """
    raise NotImplementedError


def negative_sampling_loss(
    positive_score: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Negative sampling loss for Skip-gram.

    L = -log(sigma(s_+)) - sum_k log(sigma(-s_k^-))

    Args:
        positive_score: Dot product for positive pair, shape (batch_size, 1).
        negative_scores: Dot products for negative pairs, shape (batch_size, K).

    Returns:
        Scalar loss (mean over batch).
    """
    raise NotImplementedError


def cbow_forward(
    context_words: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """Continuous Bag of Words: average context embeddings to predict center word.

    Args:
        context_words: Context word indices, shape (batch_size, context_size).
        embeddings: Embedding matrix, shape (vocab_size, embed_dim).

    Returns:
        Context representation (average of context embeddings),
        shape (batch_size, embed_dim).
    """
    raise NotImplementedError


def glove_loss(
    w_i: torch.Tensor,
    w_j: torch.Tensor,
    b_i: torch.Tensor,
    b_j: torch.Tensor,
    x_ij: torch.Tensor,
    x_max: float = 100.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """GloVe weighted least squares loss.

    L = sum f(X_ij) * (w_i^T w_j + b_i + b_j - log(X_ij))^2

    where f(x) = (x/x_max)^alpha if x < x_max, else 1.

    Args:
        w_i: Word vector for word i, shape (batch_size, embed_dim).
        w_j: Word vector for word j, shape (batch_size, embed_dim).
        b_i: Bias for word i, shape (batch_size,).
        b_j: Bias for word j, shape (batch_size,).
        x_ij: Co-occurrence count, shape (batch_size,).
        x_max: Maximum co-occurrence for weighting function.
        alpha: Exponent for weighting function.

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def analogy_test(
    embeddings: torch.Tensor,
    vocab: dict[str, int],
    a: str,
    b: str,
    c: str,
) -> str:
    """Solve word analogy a:b :: c:? using vector arithmetic.

    result = argmax_{w != a,b,c} cos_sim(w, b - a + c)

    Args:
        embeddings: Embedding matrix, shape (vocab_size, embed_dim).
        vocab: Dictionary mapping words to indices.
        a, b, c: Words forming the analogy.

    Returns:
        The word d that best completes the analogy.
    """
    raise NotImplementedError
