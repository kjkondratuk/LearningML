"""Distributed Training Exercises."""

import torch
import numpy as np


def all_reduce_ring(gradients: list[torch.Tensor], world_size: int) -> torch.Tensor:
    """Simulate ring all-reduce: result equals sum of all gradients."""
    raise NotImplementedError


def data_parallel_split(batch: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Split batch across workers for data parallelism."""
    raise NotImplementedError


def pipeline_parallel_schedule(num_microbatches: int, num_stages: int) -> list[list[tuple]]:
    """Generate GPipe schedule (forward then backward)."""
    raise NotImplementedError


def tensor_parallel_linear(input: torch.Tensor, weight_shards: list[torch.Tensor], world_size: int) -> torch.Tensor:
    """Column-parallel linear layer."""
    raise NotImplementedError


def communication_cost(message_size: int, bandwidth: float, latency: float, world_size: int, algorithm: str = "ring") -> float:
    """Model communication cost for different algorithms."""
    raise NotImplementedError
