# 03 -- PyTorch Fundamentals

## Learning Objectives

- Convert between NumPy arrays and PyTorch tensors fluently
- Perform tensor operations (reshape, broadcast, indexing)
- Understand autograd: how PyTorch tracks and computes gradients
- Build a custom `nn.Module` layer
- Wire up a simple model and run a forward pass

## The Big Idea

PyTorch gives you two things that plain NumPy does not:

1. **Automatic differentiation** -- you write the forward pass, PyTorch
   computes the backward pass for you.
2. **GPU acceleration** -- same code, `.to("cuda")`, and it runs on a GPU.

Everything in PyTorch is a **tensor** -- a multi-dimensional array, just like
a NumPy ndarray, but with a gradient-tracking tape attached.

### Go Parallel

If NumPy is like Go's `[]float64` slices with helper functions, PyTorch
tensors are like a wrapper struct:

```go
type Tensor struct {
    Data     []float64
    Grad     []float64
    RequiresGrad bool
    GradFn   func()        // pointer to the operation that created this
}
```

Every operation on tensors with `requires_grad=True` builds a computation
graph. Calling `.backward()` walks that graph in reverse, filling in `.grad`
for every leaf tensor.

## Key Concepts

### 1. Tensor Creation and Conversion

```python
# NumPy -> Torch
t = torch.from_numpy(arr)        # shares memory!
t = torch.tensor(arr)             # copies

# Torch -> NumPy
arr = t.numpy()                   # shares memory (CPU only)
arr = t.detach().cpu().numpy()    # safe version
```

### 2. Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # tensor([7.])  because dy/dx = 2x + 3 = 7
```

> **Math Callout:** Autograd implements reverse-mode automatic differentiation,
> which computes all partial derivatives in one backward pass. This is why
> training a network with millions of parameters is feasible. See 1D-02.

### 3. nn.Module

Every PyTorch model is a subclass of `nn.Module`. You define `__init__` (create
layers) and `forward` (compute output). PyTorch handles parameter registration,
serialization, and device management.

### 4. Custom Linear Layer

Building your own `nn.Module` that does `y = x @ W.T + b` teaches you what
`nn.Linear` does under the hood.

## Style Notes

- Always use `torch.no_grad()` when you do not need gradients (inference,
  evaluation).
- Prefer `torch.tensor(...)` over `torch.Tensor(...)` -- the lowercase version
  infers dtype from the data.
- Detach before converting to NumPy: `tensor.detach().numpy()`.

## Exercises

See [exercises.py](exercises.py) for 5 stubs. Tests in
`tests/test_exercises.py`.
