# Module 06: RNNs, LSTMs, and GRUs

> Recurrence processes sequences by maintaining hidden state. But vanilla RNNs
> cannot learn long-range dependencies because gradients vanish through the
> multiplicative chain of Whh. LSTMs fix this with additive cell state updates.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 10 | Sequence modeling with RNNs |
| Olah, "Understanding LSTM Networks" (blog) | Best visual explanation of LSTM gates |
| Karpathy, "The Unreasonable Effectiveness of RNNs" (blog) | Motivation and demos |
| Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) | Sections 1--3 |

## Derive It

1. **RNN gradient flow.** Show that dh_0/dh_T = product_{t=1}^{T} diag(sigma'(z_t)) Whh.
   If spectral_radius(Whh) < 1, this product -> 0 (vanishing).
   If > 1, it -> infinity (exploding).

2. **LSTM cell state.** Show c_t = f_t * c_{t-1} + i_t * c_tilde_t. The gradient
   flows through ADDITION (f_t * dc_{t+1}), not multiplication by Whh. The forget
   gate f_t controls how much gradient flows through.

3. **GRU as simplified LSTM.** Show GRU merges forget and input gates (z and 1-z).

## "Naive then Derive" Challenge

Train a vanilla RNN on length-50 sequences. Show it fails on long-range dependencies.
Analyze gradient flow. Then implement LSTM and show it succeeds.

## Exercises

See `exercises.py`.

## Mini-Project: Character-Level Language Model

Train an LSTM to generate text character by character. See `mini_project.py`.
