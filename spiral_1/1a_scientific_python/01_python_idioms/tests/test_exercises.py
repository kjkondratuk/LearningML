"""Tests for Module 01: Python Idioms.

Run with: pytest tests/test_exercises.py -v
"""

import types

import pytest

from exercises import (
    DataPoint,
    ManagedResource,
    fibonacci_generator,
    retry,
    transform_pipeline,
)


# ---------------------------------------------------------------------------
# fibonacci_generator
# ---------------------------------------------------------------------------

class TestFibonacciGenerator:
    def test_returns_generator(self):
        result = fibonacci_generator(5)
        assert isinstance(result, types.GeneratorType), (
            "fibonacci_generator should return a generator object, not a list"
        )

    def test_zero_elements(self):
        assert list(fibonacci_generator(0)) == []

    def test_one_element(self):
        assert list(fibonacci_generator(1)) == [0]

    def test_two_elements(self):
        assert list(fibonacci_generator(2)) == [0, 1]

    def test_ten_elements(self):
        expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        assert list(fibonacci_generator(10)) == expected

    def test_laziness(self):
        """Verify the generator yields values one at a time, not all at once."""
        gen = fibonacci_generator(1_000_000)
        # We should be able to pull just the first 3 without computing a million values.
        first = next(gen)
        second = next(gen)
        third = next(gen)
        assert (first, second, third) == (0, 1, 1)
        # The generator should NOT have exhausted — there are more values available.
        fourth = next(gen)
        assert fourth == 2

    def test_negative_n_yields_nothing(self):
        assert list(fibonacci_generator(-1)) == []


# ---------------------------------------------------------------------------
# retry decorator
# ---------------------------------------------------------------------------

class TestRetry:
    def test_succeeds_on_first_try(self):
        call_count = 0

        @retry(max_attempts=3)
        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeeds() == "ok"
        assert call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        @retry(max_attempts=3)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "finally"

        assert fails_twice() == "finally"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @retry(max_attempts=2)
        def always_fails():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            always_fails()

    def test_only_catches_specified_exceptions(self):
        @retry(max_attempts=5, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("wrong type")

        # TypeError is not in the exceptions tuple, so it should NOT be retried.
        with pytest.raises(TypeError, match="wrong type"):
            raises_type_error()

    def test_preserves_function_metadata(self):
        @retry(max_attempts=3)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_passes_arguments_through(self):
        @retry(max_attempts=2)
        def add(a, b, extra=0):
            return a + b + extra

        assert add(1, 2, extra=10) == 13


# ---------------------------------------------------------------------------
# ManagedResource context manager
# ---------------------------------------------------------------------------

class TestManagedResource:
    def test_initial_state(self):
        r = ManagedResource("test")
        assert r.name == "test"
        assert r.state == "created"
        assert r.enter_count == 0
        assert r.exit_count == 0

    def test_enter_acquires(self):
        r = ManagedResource("db")
        with r:
            assert r.state == "acquired"
            assert r.enter_count == 1

    def test_exit_releases(self):
        r = ManagedResource("db")
        with r:
            pass
        assert r.state == "released"
        assert r.exit_count == 1

    def test_returns_self_from_enter(self):
        r = ManagedResource("db")
        with r as ctx:
            assert ctx is r

    def test_error_state_on_exception(self):
        r = ManagedResource("db")
        with pytest.raises(ValueError):
            with r:
                raise ValueError("bad")
        assert r.state == "error"
        assert r.exit_count == 1

    def test_does_not_suppress_exceptions(self):
        r = ManagedResource("db")
        with pytest.raises(RuntimeError):
            with r:
                raise RuntimeError("unhandled")

    def test_multiple_uses(self):
        r = ManagedResource("reusable")
        with r:
            pass
        with r:
            pass
        assert r.enter_count == 2
        assert r.exit_count == 2
        assert r.state == "released"


# ---------------------------------------------------------------------------
# transform_pipeline
# ---------------------------------------------------------------------------

class TestTransformPipeline:
    def test_no_transforms(self):
        assert transform_pipeline([1, 2, 3]) == [1, 2, 3]

    def test_single_transform(self):
        assert transform_pipeline([1, 2, 3], lambda x: x * 2) == [2, 4, 6]

    def test_multiple_transforms(self):
        result = transform_pipeline(
            [1, 2, 3],
            lambda x: x + 10,
            lambda x: x * 2,
        )
        # (1+10)*2=22, (2+10)*2=24, (3+10)*2=26
        assert result == [22, 24, 26]

    def test_order_matters(self):
        add_then_mul = transform_pipeline([5], lambda x: x + 1, lambda x: x * 3)
        mul_then_add = transform_pipeline([5], lambda x: x * 3, lambda x: x + 1)
        assert add_then_mul == [18]  # (5+1)*3
        assert mul_then_add == [16]  # 5*3+1

    def test_empty_list(self):
        assert transform_pipeline([], lambda x: x * 2) == []

    def test_string_transforms(self):
        result = transform_pipeline(
            ["hello", "world"],
            str.upper,
            lambda s: s + "!",
        )
        assert result == ["HELLO!", "WORLD!"]

    def test_returns_new_list(self):
        original = [1, 2, 3]
        result = transform_pipeline(original, lambda x: x)
        assert result is not original


# ---------------------------------------------------------------------------
# DataPoint dataclass
# ---------------------------------------------------------------------------

class TestDataPoint:
    def test_create_with_all_fields(self):
        dp = DataPoint(label="cat", features=(0.1, 0.2, 0.3), weight=2.0)
        assert dp.label == "cat"
        assert dp.features == (0.1, 0.2, 0.3)
        assert dp.weight == 2.0

    def test_default_weight(self):
        dp = DataPoint(label="dog", features=(1.0,))
        assert dp.weight == 1.0

    def test_equality(self):
        dp1 = DataPoint(label="cat", features=(0.1, 0.2))
        dp2 = DataPoint(label="cat", features=(0.1, 0.2))
        assert dp1 == dp2

    def test_inequality(self):
        dp1 = DataPoint(label="cat", features=(0.1,))
        dp2 = DataPoint(label="dog", features=(0.1,))
        assert dp1 != dp2

    def test_frozen(self):
        dp = DataPoint(label="cat", features=(0.1,))
        with pytest.raises(AttributeError):
            dp.label = "dog"

    def test_hashable(self):
        dp1 = DataPoint(label="cat", features=(0.1, 0.2))
        dp2 = DataPoint(label="cat", features=(0.1, 0.2))
        # If hashable and equal, they should work in a set.
        assert len({dp1, dp2}) == 1

    def test_repr_contains_fields(self):
        dp = DataPoint(label="x", features=(1.0,), weight=0.5)
        r = repr(dp)
        assert "x" in r
        assert "1.0" in r
        assert "0.5" in r
