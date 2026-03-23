# Module 02: Reproduce "Attention Is All You Need"

## Paper
**Vaswani et al. (2017).** "Attention Is All You Need." arXiv:1706.03762

## Why This Paper
Foundational to modern ML. By this point you have built Transformer components in
earlier modules; now you reproduce the full system end-to-end for machine translation.

## Reproduction Requirements
1. Full encoder-decoder Transformer for machine translation
2. WMT 2014 English-German (or smaller parallel corpus)
3. Exact training schedule: warmup + inverse square root decay
4. Label smoothing (epsilon = 0.1)
5. BLEU evaluation (paper: 28.4; target: BLEU > 20 with limited compute)
6. Ablation: reproduce at least one row from Table 3
7. Attention heatmap visualization

## Key Architecture Details
- d_model=512, d_ff=2048, h=8, d_k=d_v=64
- 6 encoder layers, 6 decoder layers
- Dropout=0.1, label smoothing epsilon=0.1
- Adam with beta1=0.9, beta2=0.98, epsilon=1e-9
- lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
