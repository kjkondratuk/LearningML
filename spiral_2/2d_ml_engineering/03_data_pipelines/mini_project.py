"""
Mini-Project: Production Data Pipeline
========================================

Spiral 2, Phase 2D, Module 03

Build complete training pipeline for CIFAR-10:
1. Custom Dataset with augmentation
2. Stratified splits
3. DataLoader with prefetching
4. Measure data loading time vs GPU compute time
5. Compare num_workers=0,1,2,4,8
"""

import numpy as np


def build_cifar10_pipeline(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment: bool = True,
):
    """Build complete CIFAR-10 training pipeline.

    Returns:
        train_loader, val_loader, test_loader
    """
    raise NotImplementedError


def benchmark_num_workers(
    dataset, batch_size: int, worker_counts: list[int]
) -> dict[int, float]:
    """Benchmark loading speed with different num_workers.

    Returns:
        dict mapping num_workers -> seconds_per_epoch
    """
    raise NotImplementedError


def measure_bottleneck(
    model, dataloader, n_batches: int = 50
) -> dict[str, float]:
    """Measure data loading time vs compute time.

    Returns:
        dict with 'data_time', 'compute_time', 'ratio'
    """
    raise NotImplementedError
