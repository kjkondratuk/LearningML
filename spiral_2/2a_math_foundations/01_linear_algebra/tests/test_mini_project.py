"""
Tests for the SVD Image Compression mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    svd_compress,
    compression_ratio,
    reconstruction_error,
    find_elbow,
)


@pytest.fixture
def sample_image():
    """Create a synthetic 'image' with known rank structure."""
    np.random.seed(42)
    # Rank-5 image: low-rank structure + small noise
    U = np.random.randn(64, 5)
    V = np.random.randn(5, 48)
    image = U @ V + 0.01 * np.random.randn(64, 48)
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    return image


class TestSVDCompress:
    def test_full_rank_reconstruction(self, sample_image):
        k = min(sample_image.shape)
        recon = svd_compress(sample_image, k)
        np.testing.assert_allclose(recon, sample_image, atol=1e-6)

    def test_rank_1(self, sample_image):
        recon = svd_compress(sample_image, 1)
        assert recon.shape == sample_image.shape
        # Rank-1 matrix has rank 1
        assert np.linalg.matrix_rank(recon) == 1

    def test_error_decreases_with_k(self, sample_image):
        errors = []
        for k in [1, 5, 10, 20]:
            recon = svd_compress(sample_image, k)
            err = np.linalg.norm(sample_image - recon)
            errors.append(err)
        # Errors should be monotonically decreasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1]


class TestCompressionRatio:
    def test_ratio_increases_with_smaller_k(self, sample_image):
        r1 = compression_ratio(sample_image, 1)
        r5 = compression_ratio(sample_image, 5)
        r20 = compression_ratio(sample_image, 20)
        assert r1 > r5 > r20

    def test_known_value(self):
        # 100x80 image, k=10: original = 8000, compressed = 10*(100+80+1) = 1810
        image = np.zeros((100, 80))
        ratio = compression_ratio(image, 10)
        expected = 8000 / (10 * (100 + 80 + 1))
        np.testing.assert_allclose(ratio, expected, atol=1e-10)


class TestReconstructionError:
    def test_zero_error_for_identical(self, sample_image):
        err = reconstruction_error(sample_image, sample_image)
        np.testing.assert_allclose(err, 0.0, atol=1e-10)

    def test_positive_error_for_different(self, sample_image):
        noisy = sample_image + 0.1 * np.random.randn(*sample_image.shape)
        err = reconstruction_error(sample_image, noisy)
        assert err > 0


class TestFindElbow:
    def test_low_rank_image(self, sample_image):
        # Our rank-5 image should need roughly 5 components
        k = find_elbow(sample_image, threshold=0.05)
        assert 1 <= k <= 15  # generous bound

    def test_stricter_threshold_needs_more(self, sample_image):
        k_loose = find_elbow(sample_image, threshold=0.10)
        k_strict = find_elbow(sample_image, threshold=0.01)
        assert k_strict >= k_loose
