# 06 -- RNNs for Sequences

## Learning Objectives

- Understand recurrence: how a hidden state carries information across time steps
- Implement a vanilla RNN cell and unroll it manually
- Build an LSTM-based text classifier in PyTorch
- Tokenize text into integer sequences
- Train a sentiment analysis model

## The Big Idea

A **recurrent neural network** processes sequences one element at a time,
maintaining a hidden state that summarizes everything seen so far:

```
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)
```

At each time step, the new hidden state depends on the previous hidden state
and the current input. This gives the network a form of memory.

### Go Parallel

An RNN is a `for` loop with state:

```go
h := make([]float64, hiddenSize)  // initial hidden state
for _, token := range sequence {
    x := embedding[token]
    h = tanh(add(matmul(Whh, h), matmul(Wxh, x), b))
}
// h now summarizes the entire sequence
```

If you have written a lexer or parser in Go that maintains state as it scans
tokens, you already have the right mental model.

## Key Concepts

### 1. Vanilla RNN Cell

```
h_new = tanh(W_hh @ h_old + W_xh @ x + b_h)
```

Simple but suffers from vanishing gradients on long sequences.

### 2. LSTM (Long Short-Term Memory)

Adds **gates** (forget, input, output) that control information flow. The cell
state acts as a conveyor belt that can carry information unchanged across many
time steps.

> **Math Callout:** The forget gate `f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)`
> decides what fraction of each cell state dimension to keep. Values near 0
> mean "forget", near 1 mean "remember". This is how LSTMs solve the vanishing
> gradient problem. See 1D-02 for chain rule intuition.

### 3. Tokenization

Convert raw text to integer indices. A simple approach: split on whitespace,
build a vocabulary mapping `word -> int`, pad sequences to equal length.

### 4. Sentiment Classification

Use the **final hidden state** (or pool over all hidden states) as input to a
linear classifier. Binary output: positive or negative.

## Style Notes

- Always sort or pad sequences in a batch to the same length.
- Use `pack_padded_sequence` / `pad_packed_sequence` for variable-length inputs
  in production, but fixed padding is fine for learning.
- LSTMs return `(output, (h_n, c_n))` -- do not forget to unpack.

## Exercises

See [exercises.py](exercises.py) for 5 stubs. Tests in
`tests/test_exercises.py`.
