# Module 06: Capstone -- Generative Model Paper Reproduction

## Project Overview

Reproduce the DDPM paper (Ho et al. 2020, arXiv:2006.11239) on CIFAR-10 or CelebA 64x64.

This is a full end-to-end research reproduction. You will implement the U-Net architecture,
training algorithm, and sampling procedure from the paper, then evaluate with FID scores.

## Paper

**Ho et al. (2020).** "Denoising Diffusion Probabilistic Models." arXiv:2006.11239

## Requirements

1. **Architecture.** Implement the full U-Net with time embeddings (sinusoidal),
   residual blocks, and self-attention at 16x16 resolution. Match the paper's
   architecture choices (Section 4).

2. **Training.** Implement linear noise schedule with T=1000. Training loop samples
   random t, adds noise, and trains the model to predict epsilon. Train for at least
   100k steps. Log loss curves with W&B or MLflow.

3. **Sampling.** Implement both DDPM sampling (Algorithm 2, stochastic) and DDIM
   sampling (Song et al. 2020, arXiv:2010.02502, deterministic).

4. **Evaluation.** Compute FID score between 10k generated samples and the test set.
   Target: FID < 50 (the paper achieves 3.17 with full compute; limited compute is
   expected).

5. **Visualization.** Produce a grid of 64 generated samples. Show the denoising
   process for a few samples (every 100 steps).

6. **Extension (optional).** Implement classifier-free guidance (Ho & Salimans 2022,
   arXiv:2207.12598) for class-conditional generation.

7. **Report.** Write a 1-page report comparing your results to the paper's Table 1.
   Discuss what factors limited your FID score.

## Deliverables

- `reproduce.py` -- U-Net architecture with time embeddings
- `train.py` -- Training loop with logging
- `evaluate.py` -- FID computation and sample quality metrics

## Evaluation Criteria

- Architecture matches paper specifications
- Training converges (loss decreases over time)
- Generated samples are recognizable (not pure noise)
- FID score is computed correctly
- Report honestly discusses results and limitations
