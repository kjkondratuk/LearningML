# Module 03: Pretraining Objectives

## Learning Objectives

- Implement masked language modeling (BERT-style) and causal language modeling (GPT-style)
- Understand the tradeoffs between MLM, CLM, and other pretraining objectives
- Implement T5 span corruption and ELECTRA replaced token detection
- Compute and interpret perplexity

## Key Papers

- **Devlin et al. (2018).** "BERT: Pre-training of Deep Bidirectional Transformers." arXiv:1810.04805
- **Radford et al. (2019).** "Language Models are Unsupervised Multitask Learners." (GPT-2)
- **Clark et al. (2020).** "ELECTRA: Pre-training Text Encoders as Discriminators." arXiv:2003.10555

## Theoretical Background

### Masked Language Modeling (MLM)

Randomly mask 15% of tokens and predict them: L_MLM = -E[log p(x_masked | x_unmasked)].
Only masked positions contribute to the loss.

### Causal Language Modeling (CLM)

Next-token prediction with autoregressive factorization:
L_CLM = -sum_t log p(x_t | x_{<t}). Uses causal attention mask.

### Perplexity

PPL = exp(-1/N * sum log p(x_t | context)) = exp(cross_entropy).
A uniform distribution over V tokens gives PPL = V.

> **Math Callout:** Cross-entropy and KL divergence, perplexity as exponentiated
> cross-entropy, bits per character. Resource: Stanford CS224n Lecture 9-11.

## Exercises

1. `masked_language_model_loss` -- BERT-style MLM loss
2. `causal_language_model_loss` -- GPT-style CLM loss
3. `span_corruption_objective` -- T5-style span corruption
4. `electra_loss` -- ELECTRA replaced token detection
5. `perplexity` -- Compute perplexity from log probabilities
