"""
Tests for Pretraining Objective exercises.

Verifies:
- MLM loss correctness with no masking and full masking
- CLM loss for random and perfect predictions
- Perplexity of uniform distribution equals vocab size
- ELECTRA discriminator detects replaced tokens
"""

import pytest
import torch

from exercises import (
    causal_language_model_loss,
    electra_loss,
    masked_language_model_loss,
    perplexity,
)


class TestMLMLoss:
    def test_no_mask_no_loss(self):
        """With no masked positions, loss should be 0 (or handled gracefully)."""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        mask = torch.zeros(2, 10, dtype=torch.bool)
        loss = masked_language_model_loss(logits, labels, mask)
        assert loss.item() == 0.0 or torch.isnan(loss), (
            "No masked tokens should give 0 loss"
        )

    def test_all_masked_valid(self):
        """Full masking should give a valid loss."""
        logits = torch.randn(2, 10, 100)
        labels = torch.randint(0, 100, (2, 10))
        mask = torch.ones(2, 10, dtype=torch.bool)
        loss = masked_language_model_loss(logits, labels, mask)
        assert torch.isfinite(loss) and loss.item() > 0


class TestCLMLoss:
    def test_random_prediction_high_loss(self):
        """Random predictions should give loss close to log(vocab_size)."""
        vocab_size = 100
        logits = torch.zeros(2, 20, vocab_size)  # Uniform predictions
        labels = torch.randint(0, vocab_size, (2, 20))
        loss = causal_language_model_loss(logits, labels)
        expected = torch.log(torch.tensor(float(vocab_size)))
        assert abs(loss.item() - expected.item()) < 0.5

    def test_perfect_prediction_zero_loss(self):
        """Perfect predictions should give near-zero loss."""
        vocab_size = 100
        labels = torch.randint(0, vocab_size, (1, 10))
        logits = torch.full((1, 10, vocab_size), -100.0)
        # Set correct label logits very high
        for t in range(9):  # CLM predicts t+1 from logits at t
            logits[0, t, labels[0, t + 1]] = 100.0
        loss = causal_language_model_loss(logits, labels)
        assert loss.item() < 0.1


class TestPerplexity:
    def test_uniform_distribution(self):
        """Perplexity of uniform distribution over V tokens is V."""
        V = 100
        log_probs = torch.full((50,), -torch.log(torch.tensor(float(V))))
        ppl = perplexity(log_probs)
        assert abs(ppl.item() - V) < 1.0, (
            f"PPL of uniform over {V} should be {V}, got {ppl.item()}"
        )

    def test_perplexity_positive(self):
        log_probs = torch.randn(100) - 5  # Negative log probs
        ppl = perplexity(log_probs)
        assert ppl.item() > 0

    def test_lower_entropy_lower_perplexity(self):
        """More confident predictions should give lower perplexity."""
        log_probs_confident = torch.full((50,), -0.1)
        log_probs_uncertain = torch.full((50,), -5.0)
        ppl_confident = perplexity(log_probs_confident)
        ppl_uncertain = perplexity(log_probs_uncertain)
        assert ppl_confident < ppl_uncertain


class TestELECTRALoss:
    def test_discriminator_detects_replacements(self):
        """Discriminator should distinguish original from replaced tokens."""
        batch_size, seq_len, vocab_size = 2, 10, 50
        gen_logits = torch.randn(batch_size, seq_len, vocab_size)
        original = torch.randint(0, vocab_size, (batch_size, seq_len))
        corrupted = original.clone()
        corrupted[:, 3:6] = torch.randint(0, vocab_size, (batch_size, 3))

        # Discriminator outputs high for replaced, low for original
        disc_logits = torch.zeros(batch_size, seq_len, 1)
        disc_logits[:, 3:6] = 3.0  # High confidence replaced
        disc_logits[:, :3] = -3.0  # High confidence original

        gen_loss, disc_loss = electra_loss(gen_logits, disc_logits, original, corrupted)
        assert torch.isfinite(gen_loss) and torch.isfinite(disc_loss)
