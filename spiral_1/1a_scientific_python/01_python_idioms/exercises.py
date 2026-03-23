"""Module 01: Python Idioms — Exercise Stubs

Implement each function/class below until all tests in tests/test_exercises.py pass.
"""

import functools
from dataclasses import dataclass
from typing import Any, Callable, Iterator


def fibonacci_generator(n: int) -> Iterator[int]:
    """Yield the first n Fibonacci numbers (starting 0, 1, 1, 2, 3, 5, ...).

    Args:
        n: How many Fibonacci numbers to yield. Must be >= 0.

    Yields:
        The next Fibonacci number in the sequence.
    """
    raise NotImplementedError


def retry(max_attempts: int = 3, exceptions: tuple = (Exception,)) -> Callable:
    """Decorator that retries a function up to max_attempts times on specified exceptions.

    If the function still fails after max_attempts, the last exception is raised.

    Args:
        max_attempts: Maximum number of times to call the function.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        A decorator that wraps a function with retry logic.
    """
    raise NotImplementedError


class ManagedResource:
    """A context manager that tracks its lifecycle state.

    Attributes:
        name: Identifier for this resource.
        state: Current lifecycle state — one of "created", "acquired", "released", "error".
        enter_count: Number of times __enter__ has been called.
        exit_count: Number of times __exit__ has been called.

    On enter: set state to "acquired", increment enter_count, return self.
    On exit: set state to "released" (or "error" if an exception occurred),
             increment exit_count. Never suppress exceptions.
    """

    def __init__(self, name: str) -> None:
        raise NotImplementedError

    def __enter__(self) -> "ManagedResource":
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        raise NotImplementedError


def transform_pipeline(data: list, *transforms: Callable) -> list:
    """Apply a sequence of transform functions to each element using comprehensions.

    Each transform is a callable that takes a single value and returns a transformed value.
    Transforms are applied left-to-right (first transform in args is applied first).

    Args:
        data: Input list of values.
        *transforms: Variable number of transform functions.

    Returns:
        A new list with all transforms applied to each element.
    """
    raise NotImplementedError


@dataclass
class DataPoint:
    """An immutable data point with a label, feature vector, and optional weight.

    Fields:
        label: A string category label.
        features: A tuple of float values (use tuple, not list, for hashability).
        weight: A float weighting factor, defaults to 1.0.

    This dataclass should be frozen (immutable) and support equality comparison
    and hashing (so it can be used in sets and as dict keys).
    """

    # TODO: Define the fields (label, features, weight) and configure the dataclass
    #       to be frozen and hashable. Replace this pass with your field definitions.
    pass
