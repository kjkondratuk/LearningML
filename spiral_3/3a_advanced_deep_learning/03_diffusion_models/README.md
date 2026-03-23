# Module 03: Diffusion Models

## Learning Objectives

- Understand the forward diffusion process as a Markov chain that gradually adds noise
- Derive the reverse process and its connection to score matching
- Implement DDPM training and sampling algorithms
- Implement DDIM for accelerated deterministic sampling

## Key Papers

- **Ho et al. (2020).** "Denoising Diffusion Probabilistic Models." arXiv:2006.11239
- **Song et al. (2020).** "Score-Based Generative Modeling through Stochastic Differential Equations." arXiv:2011.13456
- **Song et al. (2020).** "Denoising Diffusion Implicit Models." arXiv:2010.02502

## Theoretical Background

### Forward Process

The forward process adds Gaussian noise over T steps:

    q(x_t | x_{t-1}) = N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)

Key property: we can sample x_t directly from x_0:

    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

where alpha_bar_t = prod_{s=1}^t (1 - beta_s).

### Reverse Process

The reverse process p_theta(x_{t-1} | x_t) is also Gaussian when the noise steps are
small. Ho et al. show that training simplifies to predicting the noise:

    L_simple = E_{t, x_0, epsilon} [||epsilon - epsilon_theta(x_t, t)||^2]

### Score Matching Connection

Song et al. show that diffusion models are equivalent to learning the score function
(gradient of log density): s_theta(x, t) ~ nabla_x log p_t(x).

### DDIM Sampling

DDIM provides a deterministic mapping from noise to data using the same trained model,
enabling fewer sampling steps with minimal quality loss.

> **Math Callout:** Stochastic differential equations (Song et al. 2020 tutorial),
> Gaussian transitions and their composition, score functions (Hyvarinen 2005,
> "Estimation of Non-Normalized Statistical Models by Score Matching").
> Resource: Lilian Weng "What are Diffusion Models?"

## Exercises

1. `forward_diffusion_step` -- Sample x_t from x_0 in one step
2. `compute_noise_schedule` -- Linear or cosine beta schedule
3. `reverse_step` -- Single reverse diffusion step
4. `ddpm_loss` -- Simplified noise prediction loss
5. `ddim_sample` -- Deterministic DDIM sampling

## Derivation Prompts

1. Derive q(x_t | x_0) by composing the individual Gaussian transitions.
2. Show that the optimal denoising model predicts E[epsilon | x_t].
3. Derive the DDIM update rule from the DDPM framework.

## Style Notes

- Use notation from Ho et al.: alpha, alpha_bar, beta for noise schedule.
- Always test with T=1000 for the full schedule, but support fewer steps for testing.
