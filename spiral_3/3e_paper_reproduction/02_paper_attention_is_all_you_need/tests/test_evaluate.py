"""Tests for BLEU evaluation."""
import pytest
from evaluate import compute_bleu

class TestBLEU:
    def test_perfect_match(self):
        refs = [["the", "cat", "sat"]]
        hyps = [["the", "cat", "sat"]]
        bleu = compute_bleu(refs, hyps)
        assert bleu > 0.9

    def test_no_overlap(self):
        refs = [["the", "cat"]]
        hyps = [["a", "dog"]]
        bleu = compute_bleu(refs, hyps)
        assert bleu < 0.1
