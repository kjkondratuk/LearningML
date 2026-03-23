# 04 -- The Training Loop

## Learning Objectives

- Create a PyTorch `Dataset` and `DataLoader`
- Build a classifier using `nn.Module`
- Write the canonical training loop: forward, loss, backward, step
- Evaluate on a held-out set
- Put it all together in a full training pipeline

## The Big Idea

Every PyTorch training script follows the same skeleton:

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()          # 1. clear old gradients
        predictions = model(batch_x)   # 2. forward pass
        loss = criterion(predictions, batch_y)  # 3. compute loss
        loss.backward()                # 4. backward pass
        optimizer.step()               # 5. update weights
```

That is it. Everything else is bookkeeping.

### Go Parallel

Think of the training loop as a `for` loop in Go where each iteration calls:

```go
model.ZeroGrad()
pred := model.Forward(batchX)
loss := criterion(pred, batchY)
loss.Backward()
optimizer.Step()
```

The `DataLoader` is like a Go channel that yields batches -- it handles
shuffling, batching, and parallel loading.

## Key Concepts

### 1. Dataset and DataLoader

`Dataset` defines how to access a single sample. `DataLoader` wraps it with
batching, shuffling, and multiprocessing.

### 2. The Classifier

A simple feedforward network: `Linear -> ReLU -> Linear -> ReLU -> Linear`.
The output dimension equals the number of classes.

### 3. Training vs Evaluation

Call `model.train()` before training (enables dropout, batch norm updates).
Call `model.eval()` before evaluation (disables them). Wrap eval in
`torch.no_grad()` to save memory.

### 4. Loss Functions

- **CrossEntropyLoss** for classification (combines log-softmax + NLL)
- **MSELoss** for regression

> **Math Callout:** Cross-entropy loss is `- sum(y_true * log(y_pred))`. It
> penalizes confident wrong predictions heavily. See 1D-03 for the
> information-theoretic background.

## "Wrong Way" Challenge: Forgetting `zero_grad()`

Comment out `optimizer.zero_grad()` and watch what happens. Gradients
accumulate across batches, so the effective gradient is the *sum* of all
previous batches' gradients. The model will diverge or oscillate wildly.

This is the single most common PyTorch bug for beginners.

## Mini-Project: MNIST Digit Classifier

Train a feedforward network on MNIST (28x28 grayscale digits, 10 classes).
Target: >95% test accuracy within 5 epochs.

## Style Notes

- Keep the training loop in a function, not at module level. This makes it
  testable.
- Log loss every N batches, not every batch (printing is slow).
- Use `tqdm` for progress bars if you like, but do not add it as a hard
  dependency.

## Exercises

See [exercises.py](exercises.py) for 6 stubs and
[mini_project.py](mini_project.py) for the MNIST classifier. Tests in `tests/`.
