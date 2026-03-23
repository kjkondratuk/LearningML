# Module 03: Optimization for Deep Learning

> The loss landscape of a neural network is non-convex, high-dimensional, and full
> of saddle points. This module builds the practical optimization toolkit: LR
> schedules, warmup, AdamW, and gradient clipping.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 8 | Optimization for training |
| Kingma & Ba, "Adam" (2014) | Adam derivation |
| Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019) | AdamW paper |
| Smith, "Cyclical Learning Rates" (2017) | LR range test |

## Exercises

See `exercises.py` for: LR schedules (step, exponential, cosine), warmup, AdamW,
gradient clipping, LR finder, batch size effects.
