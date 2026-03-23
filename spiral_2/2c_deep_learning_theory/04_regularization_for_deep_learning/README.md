# Module 04: Regularization for Deep Learning

> Deep networks have millions of parameters and can memorize anything. Dropout,
> batch norm, and layer norm are the three techniques that make deep learning work
> in practice. This module derives why each works.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 7 | Regularization strategies |
| Srivastava et al., "Dropout" (2014) | Original dropout paper |
| Ioffe & Szegedy, "Batch Normalization" (2015) | BatchNorm paper |

## Exercises

See `exercises.py` for: dropout forward/backward, batch norm forward/backward,
layer norm, weight decay vs L2 comparison, data augmentation effects.

## Mini-Project: Regularization Ablation Study

Train 6 variants with different regularization combos, compare. See `mini_project.py`.
