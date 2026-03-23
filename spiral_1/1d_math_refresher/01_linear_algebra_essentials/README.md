# 01 -- Linear Algebra Essentials

## Learning Objectives

- Compute dot products and matrix multiplications by hand and in code
- Understand linear transforms as matrix-vector products
- Compute and interpret the covariance matrix
- Perform eigendecomposition and use it for PCA

## Why This Matters for ML

Nearly every ML algorithm is a sequence of matrix operations:

- **Linear regression**: `y = X @ w` -- a matrix-vector product.
- **Neural network forward pass**: `h = relu(W @ x + b)` -- also a matmul.
- **PCA**: Find the eigenvectors of the covariance matrix to compress data.

If you are comfortable with these operations, the ML code will read like prose.

### Go Parallel

In Go, a dot product is:

```go
func dot(a, b []float64) float64 {
    sum := 0.0
    for i := range a {
        sum += a[i] * b[i]
    }
    return sum
}
```

Matrix multiply is three nested loops (or a call to gonum). NumPy's
`np.matmul` (or the `@` operator) does the same thing, but optimized with
BLAS libraries that run 100x faster than pure Python loops.

## Key Concepts

### 1. Dot Product

The dot product of two vectors is a scalar:
```
a . b = sum(a_i * b_i)
```

It measures how aligned two vectors are. In ML, it appears in every linear
model and every neuron.

### 2. Matrix Multiplication

`C = A @ B` where `A` is (m, n) and `B` is (n, p), giving `C` of shape (m, p).
Each element `C[i,j]` is the dot product of row `i` of `A` with column `j` of `B`.

> **Math Callout:** Matrix multiplication is not commutative: `A @ B != B @ A`
> in general. But it is associative: `(A @ B) @ C == A @ (B @ C)`. This is
> why you can chain neural network layers.

### 3. Linear Transforms

A matrix-vector product `y = A @ x` is a linear transformation. It can rotate,
scale, shear, or project a vector. Every layer in a neural network (before the
activation) is a linear transform.

### 4. Covariance Matrix

For a data matrix `X` (centered), the covariance matrix is:
```
C = (1/n) * X.T @ X
```

It tells you how features vary together. Diagonal entries are variances;
off-diagonal entries are covariances.

### 5. Eigendecomposition

A square matrix `A` can be decomposed as `A = V @ diag(lambda) @ V^{-1}`,
where `lambda` are eigenvalues and columns of `V` are eigenvectors.

For the covariance matrix, the eigenvectors are the **principal components**,
and the eigenvalues tell you how much variance each component explains.

### 6. PCA (Principal Component Analysis)

Project data onto the top-k eigenvectors of the covariance matrix. This is
the optimal linear dimensionality reduction.

## "Wrong Way" Challenge: Triple-Nested Loop vs np.matmul

Implement matrix multiply with three nested Python `for` loops. Then benchmark
it against `np.matmul` on 500x500 matrices. The difference is staggering --
this is why we use NumPy.

## Style Notes

- Use `@` for matrix multiplication, not `np.dot` (which has confusing
  behavior for high-dimensional arrays).
- Always check shapes explicitly: `assert A.shape[1] == B.shape[0]`.

## Exercises

See [exercises.py](exercises.py) for 6 stubs. Tests in
`tests/test_exercises.py`.
