# Module 01: Variational Autoencoders

## Learning Objectives

- Derive the Evidence Lower Bound (ELBO) from first principles
- Implement the reparameterization trick and understand why it enables gradient-based
  optimization of stochastic objectives
- Build a VAE from scratch and understand each component's role
- Extend to Beta-VAE for disentangled representations

## Key Paper

**Kingma & Welling (2013).** "Auto-Encoding Variational Bayes." arXiv:1312.6114

Read Sections 1-3 carefully. Section 2.4 introduces the reparameterization trick -- this
is the key insight that makes VAEs trainable.

## Theoretical Background

### Latent Variable Models

We model data x as generated from a latent variable z:

    p(x) = integral p(x|z) p(z) dz

This integral is typically intractable, so we cannot compute p(x) or the posterior
p(z|x) = p(x|z)p(z) / p(x) directly.

### The ELBO

We introduce a variational approximation q_phi(z|x) and derive:

    log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || p(z))

The right-hand side is the Evidence Lower Bound (ELBO). Maximizing it with respect to
both the generative model (decoder) and the inference model (encoder) gives us the VAE
objective.

### KL Divergence for Gaussians

When q(z|x) = N(mu, sigma^2 I) and p(z) = N(0, I), the KL divergence has a closed-form
solution:

    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

### The Reparameterization Trick

Instead of sampling z ~ q(z|x) directly (which blocks gradient flow), we write:

    z = mu + sigma * epsilon,  where epsilon ~ N(0, I)

This moves the stochasticity into epsilon, making the sampling operation differentiable
with respect to mu and sigma.

> **Math Callout:** KL divergence properties (Cover & Thomas, "Elements of Information
> Theory" ch. 2), Evidence Lower Bound derivation (Bishop, "Pattern Recognition and Machine
> Learning" ch. 10), Reparameterization trick (Kingma & Welling 2013, Section 2.4).
> Video: Arxiv Insights "Variational Autoencoders."

## Exercises

See `exercises.py` for function stubs. Implement:

1. `compute_kl_divergence` -- Closed-form KL between the encoder distribution and the
   standard normal prior.
2. `reparameterize` -- The reparameterization trick.
3. `compute_elbo` -- Full ELBO with reconstruction and KL terms.
4. `beta_vae_loss` -- Beta-VAE objective (Higgins et al. 2017, arXiv:1804.03599).
5. `log_importance_weights` -- For Importance-Weighted Autoencoders (Burda et al. 2015,
   arXiv:1509.00519).

## Mini-Project

In `mini_project.py`, build a full VAE on MNIST:
- Encoder: 2-layer MLP mapping 784 -> 256 -> (mu, log_var) with latent dim 20
- Decoder: 2-layer MLP mapping 20 -> 256 -> 784
- Train for 50 epochs, log ELBO components separately
- Generate samples by decoding random z ~ N(0, I)
- Perform latent space interpolation between two digits

## Derivation Prompts

1. Starting from log p(x) = KL(q||p) + ELBO, prove that the ELBO is a lower bound.
2. Derive the closed-form KL divergence between two multivariate Gaussians.
3. Show that the reparameterization trick gives an unbiased gradient estimator.

## Style Notes

- Use `torch.distributions` for reference implementations but implement the math yourself.
- Name variables to match the paper: `mu`, `log_var`, `z`, `x_recon`.
- Keep encoder and decoder as separate classes for clarity.
