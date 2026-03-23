# Module 02: Transformer Deep Dive

## Learning Objectives

- Implement sinusoidal, rotary (RoPE), and ALiBi positional encodings
- Understand attention scaling and numerical stability
- Build multi-head attention from scratch
- Explore efficient attention variants (FlashAttention concepts)

## Key Papers

- **Vaswani et al. (2017).** "Attention Is All You Need." arXiv:1706.03762
- **Su et al. (2021).** "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864
- **Dao et al. (2022).** "FlashAttention: Fast and Memory-Efficient Exact Attention." arXiv:2205.14135

## Theoretical Background

### Positional Encodings

Since Transformers have no inherent notion of position, we must inject positional
information. Three approaches:

**Sinusoidal (Vaswani et al. 2017):** PE(pos, 2i) = sin(pos / 10000^(2i/d_model)).
Frequencies decrease geometrically with dimension.

**RoPE (Su et al. 2021):** Applies rotation matrices to query and key vectors. The dot
product q^T k depends only on the relative position m-n, not absolute positions.

**ALiBi (Press et al. 2022):** Adds a linear bias to attention scores based on distance:
score_ij = q_i^T k_j - m * |i - j|, where m differs per head.

### Attention Scaling

Scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
The 1/sqrt(d_k) factor prevents the dot products from growing too large, which would
push softmax into saturated regions with vanishing gradients.

> **Math Callout:** Softmax numerical stability (log-sum-exp trick), computational
> complexity of attention O(n^2 d), rotary embeddings as complex number multiplication.
> Resource: Jay Alammar "The Illustrated Transformer."

## Exercises

1. `sinusoidal_positional_encoding` -- Original PE from Vaswani et al.
2. `rotary_position_embedding` -- RoPE rotation matrices
3. `scaled_dot_product_attention` -- Full attention with scaling and masking
4. `multi_head_attention` -- Split, attend, concatenate, project
5. `flash_attention_tiled` -- Simplified tiled attention
6. `alibi_bias` -- ALiBi positional bias matrix

## Style Notes

- Use `torch.einsum` for clarity in attention computations.
- Always apply the log-sum-exp trick for numerical stability in softmax.
