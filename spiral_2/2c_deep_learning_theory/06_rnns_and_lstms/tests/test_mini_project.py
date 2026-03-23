"""
Tests for the character-level language model mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    prepare_char_data,
)


class TestPrepareData:
    def test_encoding(self):
        text = "hello world"
        c2i, i2c, encoded = prepare_char_data(text)
        assert len(c2i) == len(set(text))
        assert encoded.shape == (len(text),)
        # Roundtrip
        decoded = "".join(i2c[i] for i in encoded)
        assert decoded == text
