# Module 06: Capstone -- Fine-tune a Language Model

## Project Overview

LoRA/QLoRA fine-tuning of a 7B parameter model on instruction-following, then DPO alignment.

## Requirements

1. **Base Model.** Select Llama-2-7B or Mistral-7B.
2. **Dataset.** Prepare an instruction-following dataset (at least 5k examples, Alpaca-format or custom).
3. **LoRA from Scratch.** Implement LoRA rank decomposition (W = W_0 + BA where B is d x r and A is r x d). Compare against HuggingFace PEFT.
4. **Fine-tuning.** LoRA with rank 16. Track training loss, eval loss, GPU memory.
5. **DPO.** Apply DPO on preference pairs (UltraFeedback or custom).
6. **Evaluation.** Perplexity, MT-Bench style evaluation, human evaluation on 50 samples.
7. **Comparison.** LoRA fine-tuned vs DPO-aligned outputs qualitatively.
8. **Report.** 2-page report: what worked, what did not, what you would change.

## Key Reference

- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
