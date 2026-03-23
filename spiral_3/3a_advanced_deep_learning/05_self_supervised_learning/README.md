# Module 05: Self-Supervised Learning

## Learning Objectives

- Understand contrastive learning as mutual information maximization
- Implement InfoNCE and NT-Xent loss functions
- Build SimCLR augmentation pipeline and training loop
- Implement masked autoencoder loss for vision
- Compare contrastive (SimCLR, BYOL) vs generative (MAE) self-supervised approaches

## Key Papers

- **Chen et al. (2020).** "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR). arXiv:2002.05709
- **He et al. (2021).** "Masked Autoencoders Are Scalable Vision Learners" (MAE). arXiv:2111.06377
- **Grill et al. (2020).** "Bootstrap Your Own Latent" (BYOL). arXiv:2006.07733

## Theoretical Background

### Contrastive Learning and Mutual Information

InfoNCE loss provides a lower bound on mutual information I(X; Y):

    L_InfoNCE = -E[log(exp(sim(x, x+) / tau) / sum_j exp(sim(x, x_j) / tau))]

where x+ is a positive pair and x_j includes both positives and negatives.

### NT-Xent (Normalized Temperature-scaled Cross-Entropy)

SimCLR uses NT-Xent: for a positive pair (i, j) in a batch of 2N samples:

    l(i, j) = -log(exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau))

### Masked Autoencoders

MAE randomly masks a high proportion (75%) of image patches and trains an encoder-decoder
to reconstruct only the masked patches. The loss is MSE on masked positions.

### BYOL: Learning Without Negatives

BYOL uses an online network and a target network (EMA of online). The loss is the cosine
similarity between the online projection and the target projection, with no negative pairs.

> **Math Callout:** Mutual information (Cover & Thomas ch. 2), contrastive estimation
> (Gutmann & Hyvarinen 2010), information-theoretic bounds.
> Resource: Lilian Weng "Self-Supervised Representation Learning."

## Exercises

1. `info_nce_loss` -- InfoNCE contrastive loss
2. `simclr_augmentation_pair` -- Random augmentation for two views
3. `nt_xent_loss` -- NT-Xent loss from SimCLR
4. `masked_autoencoder_loss` -- MAE loss on masked patches
5. `byol_loss` -- BYOL loss without negatives

## Mini-Project

SimCLR on CIFAR-10: train a ResNet-18 encoder with NT-Xent, then evaluate with
linear probing on the frozen representations.

## Style Notes

- Use `torchvision.transforms` for augmentations.
- Temperature tau is critical: start with 0.1 as in the SimCLR paper.
