# Module 04: Reproduce "Direct Preference Optimization"

## Paper
**Rafailov et al. (2023).** "Direct Preference Optimization." arXiv:2305.18290

## Requirements
1. Derive DPO loss from KL-constrained reward maximization
2. DPO training on GPT-2 medium (or similar small LM)
3. Anthropic HH-RLHF or UltraFeedback for preference pairs
4. Compare against RLHF baseline (reward model + PPO)
5. Win rate evaluation against SFT baseline
6. Ablation: vary beta, plot accuracy/diversity tradeoff
7. Extension: IPO (Azar et al. 2023, arXiv:2310.12036)
