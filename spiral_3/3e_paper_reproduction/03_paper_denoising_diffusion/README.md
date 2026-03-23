# Module 03: Reproduce "Denoising Diffusion Probabilistic Models"

## Paper
**Ho et al. (2020).** "Denoising Diffusion Probabilistic Models." arXiv:2006.11239

## Requirements
1. U-Net architecture with time embeddings and attention blocks
2. Full training algorithm (Algorithm 1)
3. Sampling algorithm (Algorithm 2)
4. Train on CIFAR-10 for 200k+ steps
5. DDIM sampling comparison (50 vs 1000 steps)
6. FID score < 50 (paper: 3.17)
7. Extension: classifier-free guidance (Ho & Salimans 2022, arXiv:2207.12598)
