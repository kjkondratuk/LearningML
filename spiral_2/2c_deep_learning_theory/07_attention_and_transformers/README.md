# Module 07: Attention and Transformers

> The Transformer replaced recurrence with pure attention. This module derives every
> component: scaled dot-product attention, multi-head attention, positional encoding,
> and the full encoder-decoder architecture.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Vaswani et al., "Attention Is All You Need" (2017) | The original paper |
| Jay Alammar, "The Illustrated Transformer" (blog) | Best visual walkthrough |
| Karpathy, "Let's build GPT" (video) | Building from scratch |

## Derive It

1. **Scaling factor.** If Q and K have entries ~ N(0,1), then QK^T entries have
   variance d_k. Dividing by sqrt(d_k) keeps the variance at 1, preventing
   softmax from saturating.

2. **Multi-head attention.** Show that multiple heads learn different "attention
   patterns" in parallel. The output is the concatenation of all heads, projected.

3. **Positional encoding.** sin/cos at different frequencies encode position.
   Derive: PE(pos+k) can be written as a linear function of PE(pos), so the
   model can learn relative position.

4. **Causal masking.** Setting attention weights to -inf before softmax prevents
   attending to future tokens, enabling autoregressive generation.

## "Naive then Derive" Challenge

Build seq2seq with LSTM (Module 06). Note the bottleneck: entire source compressed
to one vector. Add attention to let the decoder look at all encoder states. Then
derive the Transformer: only attention, no recurrence.

## Exercises

See `exercises.py`.

## Mini-Project: Transformer from Scratch

Build in PyTorch, train on sequence sorting, visualize attention. See `mini_project.py`.
