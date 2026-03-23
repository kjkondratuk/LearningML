# Module 03: Data Pipelines

> A fast model with a slow data pipeline trains at the speed of the pipeline.
> This module builds production-quality data loading with augmentation, stratified
> splits, and prefetching.

## Resources

| Resource | What to Focus On |
|----------|-----------------|
| PyTorch DataLoader documentation | Core API |
| Karpathy, "A Recipe for Training Neural Networks" (blog) | Practical wisdom |

## Exercises

See `exercises.py` for: custom Dataset, augmentation pipeline, stratified splits,
efficient DataLoader, imbalanced data handling, data validation.

## Mini-Project: Production Data Pipeline

Build complete CIFAR-10 pipeline, measure data loading vs GPU compute time.
See `mini_project.py`.
