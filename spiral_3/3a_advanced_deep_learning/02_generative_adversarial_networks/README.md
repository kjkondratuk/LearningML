# Module 02: Generative Adversarial Networks

## Learning Objectives

- Understand the minimax game formulation and its connection to Jensen-Shannon divergence
- Implement original GAN and Wasserstein GAN loss functions
- Understand mode collapse and training instability
- Implement gradient penalty for enforcing the Lipschitz constraint
- Compute FID scores for evaluating generative models

## Key Papers

- **Goodfellow et al. (2014).** "Generative Adversarial Nets." arXiv:1406.2661
- **Arjovsky et al. (2017).** "Wasserstein GAN." arXiv:1701.07875
- **Gulrajani et al. (2017).** "Improved Training of Wasserstein GANs." arXiv:1704.00028

## Theoretical Background

### The Minimax Game

The GAN objective is:

    min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]

At the Nash equilibrium, the generator distribution matches the data distribution, and
the discriminator outputs 0.5 everywhere.

### From JS Divergence to Wasserstein Distance

The original GAN minimizes the Jensen-Shannon divergence between p_data and p_g. This
causes training difficulties when the supports of the two distributions do not overlap
(the JS divergence saturates at log 2).

The Wasserstein distance (Earth Mover's distance) provides a more meaningful gradient
signal even when distributions have disjoint support:

    W(P, Q) = inf_{gamma in Pi(P,Q)} E_{(x,y)~gamma}[||x - y||]

By the Kantorovich-Rubinstein duality, this can be computed using 1-Lipschitz functions:

    W(P, Q) = sup_{||f||_L <= 1}  E_P[f(x)] - E_Q[f(x)]

The gradient penalty (WGAN-GP) enforces the Lipschitz constraint by penalizing the
gradient norm on interpolated samples.

> **Math Callout:** Jensen-Shannon divergence, Wasserstein/Earth Mover's distance
> (Villani, "Optimal Transport"), Nash equilibrium in two-player games.
> Resource: Lilian Weng blog "From GAN to WGAN."

## Exercises

See `exercises.py`. Implement:

1. `generator_loss_original` -- Original GAN generator loss: -log(D(G(z)))
2. `discriminator_loss_original` -- Original discriminator loss
3. `wasserstein_loss` -- Wasserstein distance estimate
4. `gradient_penalty` -- WGAN-GP gradient penalty term
5. `spectral_norm` -- Spectral normalization for weight matrices
6. `frechet_inception_distance` -- FID score between two feature distributions

## Mini-Project

In `mini_project.py`, implement WGAN-GP on CIFAR-10. Track FID over training.

## Derivation Prompts

1. Show that the optimal discriminator for the original GAN is D*(x) = p_data(x) / (p_data(x) + p_g(x)).
2. Derive the connection between the original GAN objective and JS divergence.
3. Prove that the Wasserstein distance is continuous even when JS divergence is not.

## Style Notes

- Use `torch.autograd.grad` for computing the gradient penalty.
- Discriminator in WGAN is called "critic" (no sigmoid at output).
- Log training metrics every 100 iterations to detect mode collapse early.
