# Module 01: Python Idioms for ML

> **Time estimate:** 3--4 hours
> **Prerequisites:** Basic Python syntax (variables, functions, classes, loops)

## Why This Module Exists

ML codebases are *dense* with Python idioms that don't exist in Go. Open any PyTorch tutorial and you'll see generators feeding data loaders, decorators wrapping training hooks, context managers controlling device placement, and dataclasses holding configuration. If these patterns aren't second nature, you'll be fighting the language instead of learning the math.

This module builds that fluency.

## Concepts

### 1. Generators and `yield`

A generator is a function that produces a sequence of values lazily---it computes each value only when asked.

> **Go parallel:** A generator is like a goroutine writing to an unbuffered channel. The sender (generator) blocks after each `yield` until the receiver (caller) asks for the next value with `next()`. The key difference: Python generators are *pull-based* (caller drives), while Go channels are *push-based* (sender drives).

**Where this shows up in ML:** Data loaders yield batches one at a time so you never load an entire dataset into memory. Think of iterating over 10 million training examples---you want a lazy stream, not a 40GB list.

```python
def batches(data, size):
    for i in range(0, len(data), size):
        yield data[i:i + size]
```

**Key mechanics:**
- `yield` suspends the function and returns a value
- Calling the function returns a generator *object*, not the result
- `next(gen)` resumes execution until the next `yield`
- The generator is *exhausted* after the last yield (raises `StopIteration`)
- You can iterate with `for item in gen:` which handles `StopIteration` for you

### 2. Decorators

A decorator is a function that wraps another function to modify its behavior. The `@decorator` syntax is just sugar for `func = decorator(func)`.

> **Go parallel:** Decorators are middleware. In Go HTTP, you wrap handlers: `logged(authenticated(handler))`. Python decorators do the same thing, but `@logged` above the function definition reads top-to-bottom instead of inside-out.

**Where this shows up in ML:** `@torch.no_grad()` disables gradient computation during inference. `@functools.lru_cache` memoizes expensive feature computations. Custom decorators handle retry logic for flaky data downloads.

**Key mechanics:**
- A decorator takes a function and returns a new function
- Use `functools.wraps` to preserve the original function's name and docstring
- Decorators can accept arguments (requires an extra layer of nesting)

### 3. Context Managers

A context manager defines setup and teardown around a block of code, using `with`.

> **Go parallel:** This is `defer` with structure. In Go, you write `f, err := os.Open(path); defer f.Close()`. Python's `with open(path) as f:` bundles the open and the deferred close into a single construct. The advantage: the teardown is *guaranteed* even if the block raises an exception.

**Where this shows up in ML:** `with torch.cuda.device(0):` sets the active GPU. `with open("model.pt", "wb") as f:` saves model weights. Timer context managers measure training step duration.

**Key mechanics:**
- Implement `__enter__` (setup, returns the resource) and `__exit__` (teardown)
- `__exit__` receives exception info; returning `True` suppresses the exception
- Alternatively, use `@contextlib.contextmanager` with a `yield`

### 4. Comprehensions and Transform Pipelines

List/dict/set comprehensions are concise ways to build collections with inline filtering and transformation.

> **Go parallel:** Go doesn't have comprehensions. The closest thing is building a slice with a for loop and append. Python comprehensions replace that entire pattern with a single expression. Once you're used to them, going back to multi-line loops for simple transforms feels verbose.

**Where this shows up in ML:** Feature engineering pipelines, data filtering, building label maps, extracting columns.

```python
# Filter and transform in one expression
valid_scores = [s.value for s in scores if s.is_valid and s.value > threshold]
```

**Key mechanics:**
- List: `[expr for x in iterable if condition]`
- Dict: `{k: v for k, v in items if condition}`
- Set: `{expr for x in iterable}`
- Generator expression: `(expr for x in iterable)` --- lazy, like a generator
- Nest loops left-to-right: `[f(x, y) for x in xs for y in ys]`

### 5. Dataclasses

Dataclasses auto-generate `__init__`, `__repr__`, `__eq__`, and more from field annotations.

> **Go parallel:** A dataclass is a struct with auto-generated methods. In Go, you define a struct and get zero-value initialization for free. In Python, you'd normally write a boilerplate `__init__` assigning each field. `@dataclass` eliminates that and gives you equality, hashing, and repr for free.

**Where this shows up in ML:** Configuration objects (hyperparameters, training args), data containers for batch metadata, experiment tracking records.

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
```

**Key mechanics:**
- `@dataclass` reads the class annotations and generates methods
- `frozen=True` makes instances immutable (like a Go struct with no exported fields)
- Default values and `field(default_factory=list)` for mutable defaults
- `__post_init__` runs after the generated `__init__` for validation

## Exercises

Implement the stubs in `exercises.py`. Run `pytest tests/` to check your work.

| Function | What It Practices |
|----------|------------------|
| `fibonacci_generator` | Generators, `yield`, lazy evaluation |
| `retry` | Decorators with arguments, `functools.wraps` |
| `ManagedResource` | Context manager protocol (`__enter__`/`__exit__`) |
| `transform_pipeline` | Comprehensions, function composition |
| `DataPoint` | Dataclasses, field defaults, equality |

## Style Notes

- **Generators over lists** when the consumer doesn't need all values at once. Memory matters at ML scale.
- **`functools.wraps`** is not optional. Without it, debugging decorated functions is miserable because `func.__name__` returns the wrapper's name.
- **Dataclasses over dicts** for structured data. You get type hints, IDE autocomplete, and equality for free.
- **Comprehensions** are great for 1-line transforms. If you need `if/elif/else` logic or the expression exceeds ~80 chars, use a regular loop. Readability wins.
- **Context managers** for anything that has a "must cleanup" invariant. If you'd use `defer` in Go, use `with` in Python.
