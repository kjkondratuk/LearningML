# Module 02: NumPy Fundamentals

> **Time estimate:** 4--6 hours
> **Prerequisites:** Module 01 complete, basic algebra refresher
> **Math prep:** Watch [3Blue1Brown: Essence of Linear Algebra, Chapters 1--4](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) before starting.

## Why NumPy Is the Foundation of Everything

Every ML library---PyTorch, TensorFlow, scikit-learn, pandas---is built on top of NumPy arrays or something shaped like them. Understanding NumPy isn't optional; it's *the* prerequisite for everything that follows.

The core insight: NumPy lets you express operations on entire arrays at once instead of looping element-by-element. This isn't just convenient---it's **100x faster** because the actual work happens in compiled C code.

## Concepts

### 1. Arrays and dtypes

A NumPy array is a contiguous block of memory holding elements of a single type. This is fundamentally different from a Python list (which is an array of pointers to objects scattered across the heap).

> **Go parallel:** A NumPy `ndarray` is like a Go slice backed by a typed, contiguous array (`[]float64`). Python lists are more like `[]interface{}`---flexible but slow because of all the pointer chasing and type checks.

**Key properties:**
- `shape`: Tuple of dimensions, e.g., `(3, 4)` for a 3x4 matrix
- `dtype`: Data type of elements, e.g., `float64`, `int32`
- `ndim`: Number of dimensions
- `size`: Total number of elements

### 2. Vectorized Operations and Broadcasting

When you write `a + b` where `a` and `b` are arrays, NumPy adds every element pair in compiled C. No Python loop. This is *vectorization*.

Broadcasting extends this: when arrays have different shapes, NumPy automatically "stretches" the smaller one to match.

> **Go parallel:** Imagine writing a generic `Map[T]([]T, func(T) T) []T` function---but instead of calling a Go function per element, the loop itself runs in C with SIMD instructions. That's what NumPy vectorization gives you.

**Broadcasting rules:**
1. Shapes are compared element-wise from the *right*
2. Dimensions are compatible if they're equal or one of them is 1
3. The array with dimension 1 is stretched to match the other

```python
# Shape (3, 1) + Shape (1, 4) -> Shape (3, 4)
a = np.array([[1], [2], [3]])      # shape (3, 1)
b = np.array([[10, 20, 30, 40]])   # shape (1, 4)
c = a + b                          # shape (3, 4)
```

### 3. Normalization (Standardization)

Normalizing data to have mean=0 and standard deviation=1 is a *universal* preprocessing step in ML. The formula:

$$z = \frac{x - \mu}{\sigma}$$

Where $\mu$ is the mean and $\sigma$ is the standard deviation. After normalization, your features all live on the same scale, which prevents features with large values from dominating the learning.

> **Math resource:** [StatQuest: Standard Deviation](https://www.youtube.com/watch?v=XNgt7F6FqDU)

### 4. Euclidean Distance

The distance between two points in n-dimensional space:

$$d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

This shows up *everywhere*: K-nearest neighbors, clustering, loss functions. It generalizes the Pythagorean theorem to any number of dimensions.

> **Math resource:** [3Blue1Brown: Dot products and duality](https://www.youtube.com/watch?v=LyGKycYT2v0) --- distance and dot products are deeply related.

### 5. Moving Average

A sliding window that smooths noisy data by averaging the last $k$ values. You'll use this to smooth training loss curves so you can see the trend through the noise.

### 6. One-Hot Encoding

Converts categorical labels (0, 1, 2, ...) into binary vectors. Label 2 with 4 classes becomes `[0, 0, 1, 0]`. Neural networks need numeric input, so categories must be encoded this way.

### 7. Dot Products and Batch Operations

The dot product of two vectors is a scalar that measures their similarity:

$$a \cdot b = \sum_{i=1}^{n} a_i b_i$$

In ML, you constantly compute dot products between batches of vectors (e.g., a batch of inputs times a weight matrix). Doing this with a loop is tragic; `np.dot` or `@` does it in one call.

## Exercises

Implement the stubs in `exercises.py`. Run `pytest tests/test_exercises.py -v`.

| Function | What It Practices |
|----------|------------------|
| `normalize_array` | Vectorized arithmetic, mean/std |
| `euclidean_distance` | Broadcasting, sqrt, sum along axes |
| `moving_average` | Slicing, windowed operations |
| `one_hot_encode` | Array creation, fancy indexing |
| `batch_dot_product` | Matrix multiplication, batch operations |

## Mini-Project: K-Nearest Neighbors

In `mini_project.py`, implement `knn_classify`---a K-nearest neighbors classifier using *only* NumPy. This ties together distance computation, sorting, and counting. Run `pytest tests/test_mini_project.py -v`.

## Do It the Wrong Way Challenge

Before you implement `batch_dot_product`, try this:

1. Write a pure Python version using nested for-loops.
2. Write the NumPy vectorized version.
3. Time both on arrays of shape `(1000, 256)`.

```python
import time
import numpy as np

a = np.random.randn(1000, 256)
b = np.random.randn(1000, 256)

# Your slow version
start = time.perf_counter()
result_slow = slow_batch_dot(a, b)
slow_time = time.perf_counter() - start

# Your fast version
start = time.perf_counter()
result_fast = batch_dot_product(a, b)
fast_time = time.perf_counter() - start

print(f"Loop: {slow_time:.4f}s | NumPy: {fast_time:.4f}s | Speedup: {slow_time/fast_time:.0f}x")
```

You should see a **50--200x** speedup. *That* is why we never write element-wise loops in NumPy.

## Style Notes

- **Never loop over array elements** unless there's truly no vectorized alternative. If you find yourself writing `for i in range(len(arr)):`, stop and think about how to express it as an array operation.
- **Use `@` for matrix multiply** instead of `np.dot()` for 2D arrays. It reads more like math: `C = A @ B`.
- **Specify dtype explicitly** when creating arrays: `np.zeros((3, 4), dtype=np.float64)`. Implicit dtype can cause subtle bugs when mixing float32 and float64.
- **Avoid modifying arrays in-place** unless performance demands it. `result = arr * 2` is safer than `arr *= 2` because it doesn't surprise callers holding a reference to the original.
