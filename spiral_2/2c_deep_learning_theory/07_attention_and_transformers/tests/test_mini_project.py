"""
Tests for the Transformer mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    build_sort_dataset,
)


class TestSortDataset:
    def test_shapes(self):
        src, tgt = build_sort_dataset(100, 5, 10)
        assert src.shape == (100, 5)
        assert tgt.shape == (100, 5)

    def test_tgt_is_sorted_src(self):
        src, tgt = build_sort_dataset(50, 8, 20, seed=42)
        for i in range(50):
            np.testing.assert_array_equal(tgt[i], np.sort(src[i]))
