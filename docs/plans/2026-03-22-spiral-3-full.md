Good. I have all the context I need. Now I will produce the complete Spiral 3 scaffolding plan. This is a substantial document, so I will write it directly as my response.

---

# Spiral 3: Advanced Topics & Research Frontier -- Complete Module Scaffolding

## Preamble

This document specifies the directory structure, module contents, exercise stubs, test strategies, paper references, math resources, and capstone projects for all five phases of Spiral 3. It follows the established scaffolding pattern from the `luminous-swinging-quilt.md` plan: self-contained module directories, stubs-only exercises, TDD-driven workflow, mini-projects, math callouts, and style notes.

**Key differences from Spirals 1-2:** Spiral 3 is research-level. Exercises shift from "implement this function" to "implement this paper's algorithm" or "derive this result." Tests verify mathematical properties (convergence rates, bounds, distributional properties) not just input/output correctness. Each module references specific papers by arXiv ID. The learner is expected to read original papers and reproduce key results.

---

## Phase 3A: Advanced Deep Learning

### Directory Structure

```
spiral_3/3a_advanced_deep_learning/
├── README.md                              # Phase overview, module index, prerequisites
├── 01_variational_autoencoders/
│   ├── README.md                          # Lesson: latent variable models, ELBO derivation
│   │                                      #   Math callout: KL divergence, ELBO, reparameterization
│   │                                      #   Paper: Kingma & Welling 2013 (arXiv:1312.6114)
│   │                                      #   Style notes
│   ├── exercises.py                       # Stubs for VAE components
│   ├── notebook.ipynb                     # Interactive: visualize latent space, interpolation
│   ├── mini_project.py                    # Stub: full VAE on MNIST with latent traversals
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 02_generative_adversarial_networks/
│   ├── README.md                          # Lesson: minimax game, Nash equilibrium, training dynamics
│   │                                      #   Math callout: Jensen-Shannon divergence, Wasserstein distance
│   │                                      #   Paper: Goodfellow et al. 2014 (arXiv:1406.2661)
│   │                                      #   Paper: Arjovsky et al. 2017 WGAN (arXiv:1701.07875)
│   │                                      #   Style notes
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize training dynamics, mode collapse
│   ├── mini_project.py                    # Stub: WGAN-GP on CIFAR-10
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 03_diffusion_models/
│   ├── README.md                          # Lesson: forward/reverse diffusion, score matching, DDPM
│   │                                      #   Math callout: stochastic differential equations, score functions
│   │                                      #   Paper: Ho et al. 2020 DDPM (arXiv:2006.11239)
│   │                                      #   Paper: Song et al. 2020 Score-Based (arXiv:2011.13456)
│   │                                      #   Style notes
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize forward process, noise schedules
│   ├── mini_project.py                    # Stub: DDPM on MNIST from scratch
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 04_graph_neural_networks/
│   ├── README.md                          # Lesson: message passing, spectral vs spatial, GCN, GAT
│   │                                      #   Math callout: graph Laplacian, spectral graph theory
│   │                                      #   Paper: Kipf & Welling 2016 GCN (arXiv:1609.02907)
│   │                                      #   Paper: Velickovic et al. 2017 GAT (arXiv:1710.10903)
│   │                                      #   Style notes
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize message passing on small graphs
│   ├── mini_project.py                    # Stub: node classification on Cora dataset
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 05_self_supervised_learning/
│   ├── README.md                          # Lesson: contrastive learning, masked modeling, BYOL, SimCLR
│   │                                      #   Math callout: InfoNCE loss, mutual information
│   │                                      #   Paper: Chen et al. 2020 SimCLR (arXiv:2002.05709)
│   │                                      #   Paper: He et al. 2021 MAE (arXiv:2111.06377)
│   │                                      #   Style notes
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: augmentation strategies, representation quality
│   ├── mini_project.py                    # Stub: SimCLR on CIFAR-10
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
└── 06_capstone_generative_model/
    ├── README.md                          # Full paper reproduction project spec
    ├── reproduce.py                       # Stub: implement paper's core algorithm
    ├── train.py                           # Stub: training loop with logging
    ├── evaluate.py                        # Stub: FID score, sample quality metrics
    └── tests/
        ├── test_reproduce.py
        └── test_evaluate.py
```

### Module Details

#### Module 01: Variational Autoencoders

**Exercises (stubs in `exercises.py`):**
- `compute_kl_divergence(mu, log_var)` -- KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0,1). Closed-form.
- `reparameterize(mu, log_var)` -- Reparameterization trick: sample z = mu + sigma * epsilon.
- `compute_elbo(x, x_recon, mu, log_var)` -- ELBO = reconstruction loss + KL term. Return all three.
- `beta_vae_loss(x, x_recon, mu, log_var, beta)` -- Beta-VAE objective for disentangled representations.
- `log_importance_weights(z, mu, log_var)` -- For importance-weighted autoencoders (IWAE).

**Test scenarios (`test_exercises.py`):**
- KL divergence is non-negative. KL(N(0,1) || N(0,1)) = 0.
- Reparameterized samples have correct mean/variance (statistical test over 10k samples, tolerance 0.05).
- ELBO is a lower bound on log p(x) -- verify on a tractable Gaussian case.
- Beta=1 recovers standard VAE. Beta=0 is pure reconstruction.
- Gradient flows through reparameterized samples (verify with torch.autograd).

**Math callout:** KL divergence (Cover & Thomas, ch. 2), Evidence Lower Bound (Bishop PRML ch. 10), Reparameterization trick (Kingma & Welling 2013 Section 2.4). Video: Arxiv Insights "Variational Autoencoders."

**Paper:** Kingma & Welling, "Auto-Encoding Variational Bayes" (arXiv:1312.6114).

#### Module 02: Generative Adversarial Networks

**Exercises (stubs in `exercises.py`):**
- `generator_loss_original(d_fake)` -- Original GAN generator loss: -log(D(G(z))).
- `discriminator_loss_original(d_real, d_fake)` -- Original discriminator loss.
- `wasserstein_loss(d_real, d_fake)` -- Wasserstein distance (Earth Mover's distance).
- `gradient_penalty(discriminator, real, fake, lambda_gp)` -- WGAN-GP gradient penalty term.
- `spectral_norm(weight_matrix)` -- Compute spectral norm for spectral normalization.
- `frechet_inception_distance(real_features, fake_features)` -- FID score computation.

**Test scenarios:**
- Generator and discriminator losses are adversarial: when one decreases the other increases.
- Wasserstein loss is unbounded (unlike original GAN loss).
- Gradient penalty enforces 1-Lipschitz: ||grad|| close to 1 on interpolated samples.
- Spectral norm equals largest singular value (verify against `numpy.linalg.svd`).
- FID between identical distributions is 0. FID is non-negative.

**Math callout:** Jensen-Shannon divergence, Wasserstein/Earth Mover's distance (Villani, "Optimal Transport"), Nash equilibrium. Resource: Lilian Weng blog "From GAN to WGAN."

**Papers:** Goodfellow et al. 2014 (arXiv:1406.2661), Arjovsky et al. 2017 (arXiv:1701.07875), Gulrajani et al. 2017 WGAN-GP (arXiv:1704.00028).

#### Module 03: Diffusion Models

**Exercises (stubs in `exercises.py`):**
- `forward_diffusion_step(x_0, t, noise_schedule)` -- q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I).
- `compute_noise_schedule(T, schedule_type)` -- Linear or cosine schedule. Return betas, alphas, alpha_bars.
- `reverse_step(model, x_t, t, noise_schedule)` -- Single reverse diffusion step p(x_{t-1} | x_t).
- `ddpm_loss(model, x_0, noise_schedule)` -- Simplified loss: MSE between predicted and actual noise.
- `ddim_sample(model, x_T, noise_schedule, steps)` -- Deterministic DDIM sampling (fewer steps).

**Test scenarios:**
- Forward process: x_T approaches N(0, I) as T grows. Test KS statistic at T=1000.
- Noise schedule: alpha_bar monotonically decreases from near 1 to near 0.
- Loss is non-negative. Loss decreases over training iterations on simple distribution.
- DDIM with full steps produces same distribution as DDPM.
- Reverse process: given a perfect noise predictor (oracle), samples recover x_0 distribution.

**Math callout:** Stochastic differential equations (introductory: Song et al. 2020 tutorial), Gaussian transitions, score functions (Hyvarinen 2005). Resource: Lilian Weng "What are Diffusion Models?"

**Papers:** Ho et al. 2020 DDPM (arXiv:2006.11239), Song et al. 2020 (arXiv:2011.13456), Song et al. 2020 DDIM (arXiv:2010.02502).

#### Module 04: Graph Neural Networks

**Exercises (stubs in `exercises.py`):**
- `graph_laplacian(adjacency_matrix)` -- Compute normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}.
- `gcn_layer(x, adjacency, weights)` -- Single GCN layer: H' = sigma(D_hat^{-1/2} A_hat D_hat^{-1/2} H W).
- `gat_attention(x_i, x_j, a_weights)` -- Compute GAT attention coefficient e_ij = LeakyReLU(a^T [Wh_i || Wh_j]).
- `message_passing_step(node_features, edge_index, message_fn, aggregate_fn)` -- Generic MPNN step.
- `graph_readout(node_embeddings, method)` -- Global pooling: mean, sum, or attention-based.

**Test scenarios:**
- Graph Laplacian eigenvalues are in [0, 2] for normalized version.
- GCN on fully connected graph: all nodes get same representation (permutation invariance).
- GAT attention coefficients sum to 1 per node (softmax normalization).
- Message passing is equivariant to node permutation: permuting inputs permutes outputs.
- Readout is invariant to node permutation for mean/sum pooling.

**Math callout:** Spectral graph theory (Chung, "Spectral Graph Theory" ch. 1-2), message passing framework (Gilmer et al. 2017 arXiv:1704.01212). Resource: Distill.pub "A Gentle Introduction to Graph Neural Networks."

**Papers:** Kipf & Welling 2016 (arXiv:1609.02907), Velickovic et al. 2017 (arXiv:1710.10903).

#### Module 05: Self-Supervised Learning

**Exercises (stubs in `exercises.py`):**
- `info_nce_loss(anchor, positive, negatives, temperature)` -- InfoNCE contrastive loss.
- `simclr_augmentation_pair(image)` -- Random crop + color jitter + Gaussian blur for two views.
- `nt_xent_loss(z_i, z_j, temperature)` -- Normalized temperature-scaled cross-entropy (SimCLR).
- `masked_autoencoder_loss(original, reconstructed, mask)` -- MAE loss: only on masked patches.
- `byol_loss(online_projection, target_projection)` -- BYOL loss without negatives.

**Test scenarios:**
- InfoNCE loss decreases when positive pair similarity increases.
- InfoNCE lower bounds mutual information: verify on known distributions.
- NT-Xent with temperature=0.1 vs temperature=1.0: lower temperature produces sharper distributions.
- MAE loss is 0 when reconstruction is perfect on masked regions.
- BYOL loss is symmetric with respect to its two inputs (up to stop-gradient).

**Math callout:** Mutual information (Cover & Thomas ch. 2), contrastive estimation (Gutmann & Hyvarinen 2010), information-theoretic bounds. Resource: Lilian Weng "Self-Supervised Representation Learning."

**Papers:** Chen et al. 2020 SimCLR (arXiv:2002.05709), He et al. 2021 MAE (arXiv:2111.06377), Grill et al. 2020 BYOL (arXiv:2006.07733).

#### Module 06: Capstone -- Generative Model Paper Reproduction

**Project:** Reproduce the DDPM paper (Ho et al. 2020, arXiv:2006.11239) on a non-trivial dataset (CIFAR-10 or CelebA 64x64).

**Specific requirements:**
1. Implement the full U-Net architecture with time embeddings from the paper.
2. Implement linear noise schedule and training loop.
3. Train for at least 100k steps. Log loss curves with W&B or MLflow.
4. Implement both DDPM and DDIM sampling.
5. Compute FID score. Target: FID < 50 on CIFAR-10 (the paper achieves ~3.17, but limited compute is expected).
6. Produce a grid of 64 generated samples.
7. Implement conditional generation (class-conditioned) as an extension.
8. Write a 1-page report comparing your results to the paper's Table 1.

**Tests:**
- U-Net output shape matches input shape.
- Forward pass produces valid outputs at all timesteps.
- FID computation runs without error on synthetic distributions.
- Sampling produces images in valid pixel range [0, 1].

---

## Phase 3B: NLP & Language Models

### Directory Structure

```
spiral_3/3b_nlp_language_models/
├── README.md
├── 01_word_embeddings/
│   ├── README.md                          # Lesson: distributional hypothesis, Word2Vec, GloVe
│   │                                      #   Math callout: negative sampling, NCE, SVD
│   │                                      #   Paper: Mikolov et al. 2013 (arXiv:1301.3781)
│   │                                      #   Paper: Pennington et al. 2014 GloVe
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize embeddings with t-SNE/UMAP
│   ├── mini_project.py                    # Stub: train Word2Vec from scratch on text corpus
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 02_transformer_deep_dive/
│   ├── README.md                          # Lesson: positional encodings, attention scaling, efficiency
│   │                                      #   Math callout: softmax stability, rotary embeddings
│   │                                      #   Paper: Vaswani et al. 2017 (arXiv:1706.03762)
│   │                                      #   Paper: Su et al. 2021 RoPE (arXiv:2104.09864)
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: attention visualization, scaling behavior
│   ├── mini_project.py                    # Stub: efficient attention variants (flash, linear)
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 03_pretraining_objectives/
│   ├── README.md                          # Lesson: MLM, CLM, T5 span corruption, ELECTRA
│   │                                      #   Math callout: cross-entropy, perplexity
│   │                                      #   Paper: Devlin et al. 2018 BERT (arXiv:1810.04805)
│   │                                      #   Paper: Radford et al. 2019 GPT-2
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: compare MLM vs CLM on same data
│   ├── mini_project.py                    # Stub: pretrain a small transformer on WikiText
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 04_alignment_rlhf_dpo/
│   ├── README.md                          # Lesson: reward modeling, PPO for LMs, DPO
│   │                                      #   Math callout: Bradley-Terry model, policy gradient
│   │                                      #   Paper: Ouyang et al. 2022 InstructGPT (arXiv:2203.02155)
│   │                                      #   Paper: Rafailov et al. 2023 DPO (arXiv:2305.18290)
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize reward model behavior
│   ├── mini_project.py                    # Stub: DPO fine-tuning on preference data
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 05_scaling_laws_icl/
│   ├── README.md                          # Lesson: Chinchilla scaling, emergent abilities, ICL theory
│   │                                      #   Math callout: power laws, Bayesian in-context learning
│   │                                      #   Paper: Hoffmann et al. 2022 Chinchilla (arXiv:2203.15556)
│   │                                      #   Paper: Wei et al. 2022 Emergent Abilities (arXiv:2206.07682)
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: fit scaling laws to training data
│   └── tests/
│       └── test_exercises.py
└── 06_capstone_finetune_lm/
    ├── README.md                          # Full fine-tuning project spec
    ├── data_prep.py                       # Stub: dataset preparation
    ├── finetune.py                        # Stub: LoRA/QLoRA fine-tuning pipeline
    ├── evaluate.py                        # Stub: perplexity, downstream task eval
    └── tests/
        ├── test_data_prep.py
        └── test_evaluate.py
```

### Module Details

#### Module 01: Word Embeddings

**Exercises:**
- `skipgram_forward(center_word, context_words, embeddings_in, embeddings_out)` -- Forward pass for Skip-gram with negative sampling.
- `negative_sampling_loss(positive_score, negative_scores)` -- NEG loss: -log(sigma(s+)) - sum(log(sigma(-s-))).
- `cbow_forward(context_words, embeddings)` -- Continuous Bag of Words forward pass.
- `glove_loss(w_i, w_j, b_i, b_j, x_ij, x_max, alpha)` -- GloVe weighted least squares loss.
- `analogy_test(embeddings, vocab, a, b, c)` -- Solve a:b :: c:? using vector arithmetic.

**Test scenarios:**
- Negative sampling loss is minimized when positive score is high and negative scores are low.
- GloVe loss weight function: f(x) = (x/x_max)^alpha for x < x_max, else 1.
- Analogy: after training on sufficient data, king - man + woman is close to queen (cosine similarity > 0.5).
- Embedding dimensionality: output shapes are consistent throughout forward pass.
- Loss is differentiable (gradients are finite and nonzero).

**Math callout:** Noise Contrastive Estimation (Gutmann & Hyvarinen 2010), SVD relation to GloVe (Levy & Goldberg 2014), distributional semantics. Resource: Stanford CS224n Lecture 1-2.

**Papers:** Mikolov et al. 2013 Word2Vec (arXiv:1301.3781), Pennington et al. 2014 GloVe (EMNLP).

#### Module 02: Transformer Deep Dive

**Exercises:**
- `sinusoidal_positional_encoding(seq_len, d_model)` -- Original sinusoidal PE from Vaswani et al.
- `rotary_position_embedding(x, seq_len)` -- RoPE: apply rotation matrices to query/key.
- `scaled_dot_product_attention(q, k, v, mask, dropout)` -- Full attention with proper scaling.
- `multi_head_attention(q, k, v, num_heads, d_model)` -- Split, attend, concatenate, project.
- `flash_attention_tiled(q, k, v, block_size)` -- Simplified tiled attention (FlashAttention concept).
- `alibi_bias(num_heads, seq_len)` -- ALiBi positional bias matrix.

**Test scenarios:**
- Sinusoidal PE: position 0 always has the same encoding. Frequencies decrease with dimension.
- RoPE: rotating q and k by same angle preserves dot product (relative position invariance).
- Attention with uniform queries: output is mean of values.
- Attention with causal mask: position i only attends to positions <= i.
- Flash attention (tiled) produces same output as naive attention (numerical tolerance 1e-5).
- ALiBi: bias is linear in distance, slope differs per head.

**Math callout:** Softmax numerical stability (log-sum-exp trick), computational complexity of attention O(n^2 d), rotary embeddings as complex number multiplication. Resource: Jay Alammar "The Illustrated Transformer."

**Papers:** Vaswani et al. 2017 (arXiv:1706.03762), Su et al. 2021 RoPE (arXiv:2104.09864), Dao et al. 2022 FlashAttention (arXiv:2205.14135).

#### Module 03: Pretraining Objectives

**Exercises:**
- `masked_language_model_loss(logits, labels, mask_positions)` -- BERT-style MLM: cross-entropy only at masked positions.
- `causal_language_model_loss(logits, labels)` -- GPT-style CLM: next-token prediction with causal mask.
- `span_corruption_objective(input_ids, sentinel_tokens)` -- T5-style: replace spans with sentinels, predict spans.
- `electra_loss(generator_logits, discriminator_logits, original_ids, corrupted_ids)` -- ELECTRA: replaced token detection.
- `perplexity(log_probs)` -- Compute perplexity from log probabilities.

**Test scenarios:**
- MLM loss: masking no tokens gives 0 loss (no signal). Masking all tokens is valid.
- CLM loss: perfectly predicted sequence gives 0 loss. Random predictions give loss close to log(vocab_size).
- Perplexity of uniform distribution over V tokens is V.
- ELECTRA discriminator: binary cross-entropy per token, ground truth is which tokens were replaced.
- Span corruption: sentinels in input correspond to spans in target.

**Math callout:** Cross-entropy and its relation to KL divergence, perplexity as exponentiated cross-entropy, bits per character. Resource: Stanford CS224n Lecture 9-11.

**Papers:** Devlin et al. 2018 BERT (arXiv:1810.04805), Radford et al. 2019 GPT-2 (OpenAI blog), Clark et al. 2020 ELECTRA (arXiv:2003.10555).

#### Module 04: Alignment -- RLHF & DPO

**Exercises:**
- `bradley_terry_probability(reward_chosen, reward_rejected)` -- P(chosen > rejected) = sigma(r_c - r_r).
- `reward_model_loss(reward_chosen, reward_rejected)` -- -log(sigma(r_c - r_r)).
- `ppo_clipped_objective(ratio, advantage, epsilon)` -- Clipped surrogate objective from PPO.
- `dpo_loss(pi_logprob_chosen, pi_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta)` -- Direct Preference Optimization loss.
- `kl_penalty(pi_logprobs, ref_logprobs)` -- KL divergence penalty for RLHF.

**Test scenarios:**
- Bradley-Terry: equal rewards give probability 0.5. Higher chosen reward gives probability > 0.5.
- Reward model loss is non-negative. Gradient pushes r_chosen up and r_rejected down.
- PPO clipped objective: when ratio is within [1-eps, 1+eps], unclipped. Outside: clipped.
- DPO loss: at convergence, implicit reward margin equals beta * log(pi/ref) margin.
- DPO with beta -> infinity approaches supervised learning on chosen completions.
- KL penalty is 0 when pi = ref.

**Math callout:** Bradley-Terry preference model, policy gradient theorem (Sutton & Barto ch. 13), KL-constrained optimization. Resource: Anthropic "Constitutional AI" paper, Hugging Face RLHF blog.

**Papers:** Ouyang et al. 2022 InstructGPT (arXiv:2203.02155), Rafailov et al. 2023 DPO (arXiv:2305.18290), Schulman et al. 2017 PPO (arXiv:1707.06347).

#### Module 05: Scaling Laws & In-Context Learning

**Exercises:**
- `fit_power_law(compute, loss_values)` -- Fit L(C) = a * C^(-alpha) + L_inf.
- `chinchilla_optimal_ratio(compute_budget)` -- Given FLOP budget, compute optimal N (params) and D (tokens).
- `compute_flops_estimate(num_params, num_tokens)` -- Approximate FLOPs as 6 * N * D.
- `bayesian_icl_posterior(prior, likelihood_fn, examples)` -- In-context learning as implicit Bayesian inference.
- `emergent_ability_metric(model_sizes, task_accuracy)` -- Detect phase transitions in scaling curves.

**Test scenarios:**
- Power law fit: R^2 > 0.95 on synthetic power law data with noise.
- Chinchilla ratio: params and tokens scale equally with compute (approximately N proportional to D).
- FLOPs estimate: correct order of magnitude for known model/data pairs (GPT-3: 175B params, 300B tokens).
- Bayesian ICL: with more examples, posterior concentrates (lower entropy).
- Emergent ability detection: correctly identifies step-function behavior vs gradual improvement.

**Math callout:** Power laws and log-log plots, Bayesian inference (posterior = prior * likelihood / evidence), phase transitions. Resource: Kaplan et al. 2020 "Scaling Laws for Neural Language Models" (arXiv:2001.08361).

**Papers:** Hoffmann et al. 2022 Chinchilla (arXiv:2203.15556), Wei et al. 2022 (arXiv:2206.07682), Xie et al. 2021 "An Explanation of In-context Learning as Implicit Bayesian Inference" (arXiv:2111.02080).

#### Module 06: Capstone -- Fine-tune a Language Model

**Project:** LoRA/QLoRA fine-tuning of a 7B parameter model on a custom instruction-following task, then evaluate with DPO.

**Specific requirements:**
1. Select a base model (Llama-2-7B or Mistral-7B).
2. Prepare an instruction-following dataset (at least 5k examples). Can use Alpaca-format or custom.
3. Implement LoRA from scratch (rank-decomposition of weight updates). Compare against HuggingFace PEFT.
4. Fine-tune with LoRA rank 16. Track training loss, eval loss, GPU memory.
5. Implement DPO on preference pairs (use UltraFeedback or custom preference data).
6. Evaluate: perplexity, MT-Bench style evaluation, human evaluation on 50 samples.
7. Compare LoRA fine-tuned vs DPO-aligned outputs qualitatively.
8. Write a 2-page report: what worked, what did not, what you would change.

**Tests:**
- LoRA decomposition: rank-r update has correct shape, reconstructed weight has correct dimensions.
- Fine-tuned model perplexity is lower than base model on target domain.
- DPO training: loss decreases over epochs.
- Generated outputs are valid (not degenerate, not empty).

---

## Phase 3C: Advanced Math & Theory

### Directory Structure

```
spiral_3/3c_advanced_math/
├── README.md
├── 01_learning_theory/
│   ├── README.md                          # Lesson: PAC learning, VC dimension, Rademacher complexity
│   │                                      #   Math callout: concentration inequalities, union bounds
│   │                                      #   Book: Shalev-Shwartz & Ben-David "Understanding ML" ch. 3-6
│   ├── exercises.py                       # Derive and compute bounds
│   ├── notebook.ipynb                     # Interactive: empirical vs theoretical bounds
│   └── tests/
│       └── test_exercises.py
├── 02_information_geometry/
│   ├── README.md                          # Lesson: Fisher information, natural gradient, manifolds
│   │                                      #   Math callout: Riemannian geometry basics, Fisher metric
│   │                                      #   Paper: Amari 1998, Martens 2014 (arXiv:1412.1193)
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize Fisher information for Gaussians
│   └── tests/
│       └── test_exercises.py
├── 03_kernel_methods/
│   ├── README.md                          # Lesson: RKHS, Mercer's theorem, kernel trick, GP connection
│   │                                      #   Math callout: Hilbert spaces, positive definite functions
│   │                                      #   Book: Scholkopf & Smola "Learning with Kernels" ch. 2
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: kernel PCA, kernel ridge regression
│   ├── mini_project.py                    # Stub: kernel method benchmarking suite
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 04_convex_optimization/
│   ├── README.md                          # Lesson: duality, KKT, proximal methods, ADMM
│   │                                      #   Math callout: Fenchel conjugate, strong duality
│   │                                      #   Book: Boyd & Vandenberghe "Convex Optimization" ch. 5, 9
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: visualize convergence of different methods
│   ├── mini_project.py                    # Stub: implement proximal gradient descent for LASSO
│   └── tests/
│       ├── test_exercises.py
│       └── test_mini_project.py
├── 05_measure_theory_for_ml/
│   ├── README.md                          # Lesson: sigma-algebras, measure, Lebesgue integral, Radon-Nikodym
│   │                                      #   Math callout: when and why discrete intuitions break
│   │                                      #   Book: Pollard "A User's Guide to Measure Theoretic Probability" ch. 1-3
│   ├── exercises.py
│   ├── notebook.ipynb                     # Interactive: Monte Carlo integration, importance sampling
│   └── tests/
│       └── test_exercises.py
└── 06_capstone_theory_project/
    ├── README.md                          # Derive generalization bound for a specific architecture
    ├── derivation.py                      # Stub: computational verification of theoretical results
    ├── experiments.py                     # Stub: empirical validation of bounds
    └── tests/
        ├── test_derivation.py
        └── test_experiments.py
```

### Module Details

#### Module 01: Statistical Learning Theory

**Exercises:**
- `vc_dimension_linear_classifier(d)` -- VC dimension of linear classifiers in R^d is d+1. Compute and verify.
- `pac_sample_complexity(epsilon, delta, vc_dim)` -- Compute m >= (1/eps) * (vc_dim * log(1/eps) + log(1/delta)).
- `rademacher_complexity_linear(X, num_samples)` -- Empirical Rademacher complexity: E_sigma[sup_w (1/n) sum sigma_i w^T x_i].
- `hoeffding_bound(n, t)` -- P(|S_n - E[S_n]| >= t) <= 2*exp(-2n*t^2).
- `generalization_bound(train_error, n, vc_dim, delta)` -- Compute the VC generalization bound.
- `mcdiarmid_bound(f_values, perturbation_bound, n, t)` -- McDiarmid's inequality.

**Test scenarios:**
- VC dimension of R^d linear classifier is d+1: shattering d+1 points and failing on d+2.
- PAC sample complexity increases as epsilon decreases (tighter accuracy demands more data).
- Rademacher complexity is non-negative.
- Hoeffding bound: tighter than Markov/Chebyshev on bounded random variables.
- Generalization bound: train error 0.05 with n=10000 and VC dim=50 gives bound < 0.15.
- All bounds are valid probabilities (in [0, 1]).

**Math callout:** Concentration inequalities (Boucheron et al. "Concentration Inequalities"), growth functions, Sauer's lemma. Resource: Shalev-Shwartz & Ben-David "Understanding Machine Learning: From Theory to Algorithms" ch. 3-6 (free PDF online).

#### Module 02: Information Geometry

**Exercises:**
- `fisher_information_gaussian(mu, sigma)` -- Fisher information matrix for N(mu, sigma^2): [[1/sigma^2, 0], [0, 2/sigma^2]].
- `fisher_information_bernoulli(p)` -- Fisher information for Bernoulli(p): 1/(p(1-p)).
- `natural_gradient_step(params, grad, fisher_matrix, lr)` -- Natural gradient: params -= lr * F^{-1} * grad.
- `kl_divergence_second_order(fisher, delta_theta)` -- KL(theta || theta + delta) approx (1/2) delta^T F delta.
- `geodesic_distance_gaussians(mu1, sigma1, mu2, sigma2)` -- Fisher-Rao distance between two univariate Gaussians.

**Test scenarios:**
- Fisher information for Gaussian: verify numerically by computing E[(d/dtheta log p)^2] via Monte Carlo.
- Natural gradient step: on a Gaussian family, natural gradient converges faster than vanilla gradient.
- KL second-order approximation: error is O(delta^3) -- verify by comparing to exact KL.
- Fisher-Rao distance is a true metric: triangle inequality holds, d(x,x)=0, symmetric.
- Fisher information is positive semi-definite (all eigenvalues >= 0).

**Math callout:** Riemannian manifolds (do Carmo "Riemannian Geometry" ch. 1 for basics), Fisher metric, Cramer-Rao bound. Resource: Amari "Information Geometry and Its Applications" ch. 1-3.

**Paper:** Martens 2014 "New Insights and Perspectives on the Natural Gradient Method" (arXiv:1412.1193).

#### Module 03: Kernel Methods & RKHS

**Exercises:**
- `rbf_kernel(x1, x2, sigma)` -- Gaussian RBF kernel: k(x,y) = exp(-||x-y||^2 / (2*sigma^2)).
- `polynomial_kernel(x1, x2, degree, c)` -- Polynomial kernel: k(x,y) = (x^T y + c)^d.
- `verify_mercer_conditions(kernel_matrix)` -- Check that kernel matrix is PSD (all eigenvalues >= 0).
- `kernel_ridge_regression(K, y, lambda_reg)` -- alpha = (K + lambda*I)^{-1} y.
- `kernel_pca(K, n_components)` -- PCA in feature space via eigendecomposition of centered kernel matrix.
- `nystrom_approximation(X, X_landmark, kernel_fn, rank)` -- Nystrom low-rank kernel approximation.

**Test scenarios:**
- RBF kernel: k(x,x) = 1 always. k(x,y) -> 0 as ||x-y|| -> infinity.
- Polynomial kernel: degree 1 is linear kernel (recovers linear regression in RKHS).
- Mercer: gram matrix of valid kernel is PSD. Random matrix is not.
- Kernel ridge regression: with lambda -> 0, interpolates training data. With lambda -> infinity, predicts mean.
- Kernel PCA: on concentric circles, separates classes that linear PCA cannot.
- Nystrom approximation error decreases with more landmarks.

**Math callout:** Hilbert spaces, positive definite functions, Mercer's theorem (Scholkopf & Smola ch. 2), representer theorem. Resource: Scholkopf & Smola "Learning with Kernels" (MIT Press).

#### Module 04: Convex Optimization

**Exercises:**
- `verify_convexity(f, domain_points)` -- Check f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y).
- `lagrangian(f, constraints, multipliers)` -- Form L(x, lambda) = f(x) + sum lambda_i g_i(x).
- `kkt_conditions_check(x_star, lambda_star, f_grad, g, g_grad)` -- Verify all four KKT conditions.
- `proximal_operator_l1(x, threshold)` -- Soft-thresholding: prox_{t*||.||_1}(x) = sign(x) * max(|x|-t, 0).
- `proximal_gradient_descent(f_grad, prox_g, x0, lr, iterations)` -- ISTA for composite objectives f+g.
- `admm_step(x, z, u, f_prox, g_prox, rho)` -- One step of ADMM: x-update, z-update, u-update.
- `fenchel_conjugate_quadratic(A, b)` -- Compute f*(y) for f(x) = (1/2)x^T A x + b^T x.

**Test scenarios:**
- Convexity: quadratic with positive definite Hessian is convex. Negative definite is not.
- KKT: for a simple QP, verify that the known solution satisfies all KKT conditions.
- Proximal L1: produces exact zeros for small values (sparsity). Shrinks large values by threshold.
- Proximal gradient descent on LASSO converges to sklearn's Lasso solution (tolerance 1e-4).
- ADMM: converges to same solution as proximal gradient but potentially faster.
- Fenchel conjugate of quadratic: f*(y) = (1/2)(y-b)^T A^{-1} (y-b).

**Math callout:** Convex conjugates (Boyd & Vandenberghe ch. 3), duality gap, strong duality via Slater's condition (ch. 5), proximal operators (Parikh & Boyd 2014). Resource: Boyd & Vandenberghe "Convex Optimization" (free PDF), Parikh & Boyd "Proximal Algorithms" (Foundations and Trends in Optimization).

#### Module 05: Measure-Theoretic Probability for ML

**Exercises:**
- `lebesgue_integral_mc(f, domain, n_samples)` -- Approximate integral via Monte Carlo (serves as computational entry point).
- `importance_sampling(f, target_density, proposal_density, proposal_sampler, n_samples)` -- IS estimate with variance computation.
- `radon_nikodym_discrete(p, q)` -- Compute density ratio dP/dQ for discrete distributions.
- `change_of_variables(f, jacobian, x)` -- p_y(y) = p_x(f^{-1}(y)) * |det(J_{f^{-1}}(y))|. For normalizing flows.
- `kl_divergence_mc(log_p, log_q, samples_from_p)` -- Monte Carlo estimate of KL(P || Q).

**Test scenarios:**
- MC integral of known functions: |estimate - exact| < 3*sigma/sqrt(n) with high probability.
- Importance sampling: lower variance than naive MC when proposal matches target.
- Radon-Nikodym: density ratio is non-negative. Integrates to 1 under Q.
- Change of variables: transformed density integrates to 1 (verify numerically).
- KL divergence MC: non-negative, 0 iff P=Q, consistent with exact value on known distributions.

**Math callout:** Sigma-algebras (Pollard ch. 1), Lebesgue integration (Pollard ch. 2), Radon-Nikodym theorem and its role in density estimation, change of variables for normalizing flows. Resource: Pollard "A User's Guide to Measure Theoretic Probability." Lighter alternative: Chapter 1 of Durrett "Probability: Theory and Examples."

#### Module 06: Capstone -- Theory Project

**Project:** Derive and empirically validate a generalization bound for a specific neural network architecture.

**Specific requirements:**
1. Choose one: PAC-Bayes bound for a 2-layer ReLU network, OR norm-based generalization bound following Neyshabur et al. 2017 (arXiv:1706.08947).
2. Write out the complete derivation (in LaTeX in the README or a separate PDF).
3. Implement the bound computation: given a trained network, compute the numerical value of the bound.
4. Train networks of varying width on MNIST. Plot bound vs actual test error.
5. Show that the bound is vacuous (greater than 1) for large networks -- this is the current state of theory.
6. Implement a tighter bound (compression-based or PAC-Bayes with learned prior) and compare.
7. Write a 2-page report discussing the gap between theory and practice in deep learning generalization.

**Tests:**
- Bound computation returns a valid probability (in [0, 1] or a meaningful value).
- Bound increases with model complexity (wider networks have larger bounds).
- Bound is non-negative.
- Empirical test error is always below the bound (bound is valid).

---

## Phase 3D: Specialization Tracks

### Directory Structure

```
spiral_3/3d_specialization/
├── README.md                              # Overview of all tracks, how to navigate
├── reinforcement_learning/
│   ├── README.md
│   ├── 01_mdps_dynamic_programming/
│   │   ├── README.md                      # Lesson: MDPs, Bellman equations, value/policy iteration
│   │   │                                  #   Math callout: contraction mappings, fixed point theorems
│   │   │                                  #   Book: Sutton & Barto ch. 3-4
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 02_temporal_difference/
│   │   ├── README.md                      # Lesson: TD(0), SARSA, Q-learning, eligibility traces
│   │   │                                  #   Math callout: bootstrapping, TD error, convergence
│   │   │                                  #   Book: Sutton & Barto ch. 6-7
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 03_policy_gradients/
│   │   ├── README.md                      # Lesson: REINFORCE, actor-critic, advantage estimation
│   │   │                                  #   Math callout: policy gradient theorem proof, GAE
│   │   │                                  #   Paper: Schulman et al. 2015 GAE (arXiv:1506.02438)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 04_ppo_and_modern_pg/
│   │   ├── README.md                      # Lesson: TRPO, PPO, implementation details
│   │   │                                  #   Paper: Schulman et al. 2017 PPO (arXiv:1707.06347)
│   │   │                                  #   Paper: Engstrom et al. 2020 "Implementation Matters"
│   │   ├── exercises.py
│   │   ├── mini_project.py               # Stub: PPO on CartPole/LunarLander
│   │   └── tests/
│   │       ├── test_exercises.py
│   │       └── test_mini_project.py
│   ├── 05_model_based_rl/
│   │   ├── README.md                      # Lesson: world models, Dreamer, MBPO, planning
│   │   │                                  #   Paper: Hafner et al. 2019 Dreamer (arXiv:1912.01603)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 06_offline_rl/
│   │   ├── README.md                      # Lesson: CQL, IQL, distributional shift, pessimism
│   │   │                                  #   Paper: Kumar et al. 2020 CQL (arXiv:2006.04779)
│   │   ├── exercises.py
│   │   └── tests/
│   │       └── test_exercises.py
│   └── 07_capstone_rl_agent/
│       ├── README.md                      # Full RL agent project
│       ├── environment.py
│       ├── agent.py
│       ├── train.py
│       └── tests/
│           └── test_agent.py
├── computer_vision/
│   ├── README.md
│   ├── 01_object_detection/
│   │   ├── README.md                      # Lesson: anchor boxes, YOLO, Faster R-CNN, DETR
│   │   │                                  #   Paper: Redmon et al. 2015 YOLO (arXiv:1506.02640)
│   │   │                                  #   Paper: Carion et al. 2020 DETR (arXiv:2005.12872)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 02_semantic_segmentation/
│   │   ├── README.md                      # Lesson: U-Net, FCN, DeepLab, mask prediction
│   │   │                                  #   Paper: Ronneberger et al. 2015 U-Net (arXiv:1505.04597)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 03_vision_transformers/
│   │   ├── README.md                      # Lesson: ViT, DeiT, Swin Transformer
│   │   │                                  #   Paper: Dosovitskiy et al. 2020 ViT (arXiv:2010.11929)
│   │   ├── exercises.py
│   │   ├── mini_project.py               # Stub: ViT from scratch on CIFAR-10
│   │   └── tests/
│   │       ├── test_exercises.py
│   │       └── test_mini_project.py
│   ├── 04_3d_vision_nerfs/
│   │   ├── README.md                      # Lesson: NeRFs, volume rendering, 3D Gaussian Splatting
│   │   │                                  #   Paper: Mildenhall et al. 2020 NeRF (arXiv:2003.08934)
│   │   │                                  #   Paper: Kerbl et al. 2023 3DGS (arXiv:2308.14737)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   └── 05_capstone_cv_project/
│       ├── README.md
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       └── tests/
│           └── test_model.py
├── ml_systems/
│   ├── README.md
│   ├── 01_distributed_training/
│   │   ├── README.md                      # Lesson: data parallel, model parallel, pipeline parallel
│   │   │                                  #   Paper: Huang et al. 2018 GPipe (arXiv:1811.06965)
│   │   │                                  #   Paper: Shoeybi et al. 2019 Megatron (arXiv:1909.08053)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 02_quantization/
│   │   ├── README.md                      # Lesson: PTQ, QAT, INT8, INT4, GPTQ, AWQ
│   │   │                                  #   Paper: Frantar et al. 2022 GPTQ (arXiv:2210.17323)
│   │   │                                  #   Paper: Lin et al. 2023 AWQ (arXiv:2306.00978)
│   │   ├── exercises.py
│   │   ├── mini_project.py               # Stub: quantize a model, benchmark speed vs accuracy
│   │   └── tests/
│   │       ├── test_exercises.py
│   │       └── test_mini_project.py
│   ├── 03_distillation/
│   │   ├── README.md                      # Lesson: knowledge distillation, teacher-student
│   │   │                                  #   Paper: Hinton et al. 2015 (arXiv:1503.02531)
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   ├── 04_serving_inference/
│   │   ├── README.md                      # Lesson: ONNX, TensorRT, KV cache, speculative decoding
│   │   │                                  #   Paper: Leviathan et al. 2022 Spec Decoding (arXiv:2211.17192)
│   │   ├── exercises.py
│   │   ├── mini_project.py               # Stub: build a serving pipeline with batching
│   │   └── tests/
│   │       ├── test_exercises.py
│   │       └── test_mini_project.py
│   ├── 05_profiling_optimization/
│   │   ├── README.md                      # Lesson: GPU profiling, memory optimization, mixed precision
│   │   │                                  #   Go bridge: systems thinking transfers directly
│   │   ├── exercises.py
│   │   ├── notebook.ipynb
│   │   └── tests/
│   │       └── test_exercises.py
│   └── 06_capstone_ml_system/
│       ├── README.md
│       ├── system.py
│       ├── benchmark.py
│       └── tests/
│           └── test_system.py
└── bayesian_ml/
    ├── README.md
    ├── 01_gaussian_processes/
    │   ├── README.md                      # Lesson: GP prior/posterior, kernel selection, GP regression
    │   │                                  #   Math callout: multivariate Gaussians, Cholesky decomposition
    │   │                                  #   Book: Rasmussen & Williams "GP for ML" ch. 2
    │   ├── exercises.py
    │   ├── notebook.ipynb
    │   ├── mini_project.py               # Stub: GP regression + uncertainty on 1D/2D data
    │   └── tests/
    │       ├── test_exercises.py
    │       └── test_mini_project.py
    ├── 02_variational_inference/
    │   ├── README.md                      # Lesson: mean-field VI, ELBO, CAVI, SVI
    │   │                                  #   Math callout: exponential families, ELBO derivation
    │   │                                  #   Paper: Blei et al. 2017 VI Review (arXiv:1601.00670)
    │   ├── exercises.py
    │   ├── notebook.ipynb
    │   └── tests/
    │       └── test_exercises.py
    ├── 03_mcmc_methods/
    │   ├── README.md                      # Lesson: MH, Gibbs, HMC, NUTS
    │   │                                  #   Math callout: detailed balance, Hamiltonian dynamics
    │   │                                  #   Paper: Neal 2011 MCMC using Hamiltonian (arXiv:1206.1901)
    │   │                                  #   Book: BDA3 ch. 11-12
    │   ├── exercises.py
    │   ├── notebook.ipynb
    │   ├── mini_project.py               # Stub: HMC sampler from scratch
    │   └── tests/
    │       ├── test_exercises.py
    │       └── test_mini_project.py
    ├── 04_bayesian_deep_learning/
    │   ├── README.md                      # Lesson: MC Dropout, Bayes by Backprop, deep ensembles
    │   │                                  #   Paper: Blundell et al. 2015 (arXiv:1505.05424)
    │   │                                  #   Paper: Lakshminarayanan et al. 2017 (arXiv:1612.01474)
    │   ├── exercises.py
    │   ├── notebook.ipynb
    │   └── tests/
    │       └── test_exercises.py
    ├── 05_bayesian_optimization/
    │   ├── README.md                      # Lesson: acquisition functions, GP surrogates, BO loop
    │   │                                  #   Math callout: EI, UCB, PI derivations
    │   │                                  #   Paper: Snoek et al. 2012 (arXiv:1206.2944)
    │   ├── exercises.py
    │   ├── mini_project.py               # Stub: BO for hyperparameter tuning
    │   └── tests/
    │       ├── test_exercises.py
    │       └── test_mini_project.py
    └── 06_capstone_bayesian_project/
        ├── README.md
        ├── model.py
        ├── inference.py
        └── tests/
            └── test_model.py
```

### Track-Level Details

#### Reinforcement Learning Track

**Module 01: MDPs & Dynamic Programming**

Exercises:
- `value_iteration(transition_probs, rewards, gamma, theta)` -- Iterate V(s) = max_a sum_{s'} P(s'|s,a)[R + gamma*V(s')] until convergence.
- `policy_iteration(transition_probs, rewards, gamma)` -- Alternate policy evaluation and policy improvement.
- `bellman_operator(V, transition_probs, rewards, gamma)` -- Single application of the Bellman optimality operator.
- `verify_contraction(V1, V2, T_V1, T_V2, gamma)` -- ||T(V1) - T(V2)||_inf <= gamma * ||V1 - V2||_inf.

Tests:
- Value iteration converges (successive values get closer). Contraction property holds with factor gamma.
- Policy iteration converges in finite steps. Final policy is optimal.
- On a known gridworld: both algorithms find the same optimal policy.
- Bellman operator is a contraction mapping with factor gamma.

**Module 02: Temporal Difference**

Exercises:
- `td_zero_update(V, state, next_state, reward, alpha, gamma)` -- V(s) += alpha * (r + gamma*V(s') - V(s)).
- `sarsa_update(Q, s, a, r, s_next, a_next, alpha, gamma)` -- On-policy TD control.
- `q_learning_update(Q, s, a, r, s_next, alpha, gamma)` -- Off-policy: Q(s,a) += alpha*(r + gamma*max_a' Q(s',a') - Q(s,a)).
- `td_lambda_return(rewards, values, gamma, lambda_param)` -- Compute lambda-return.
- `eligibility_trace_update(trace, state, gamma, lambda_param)` -- Accumulating trace update.

Tests:
- TD(0) on random walk: values converge to true state values.
- SARSA is on-policy: learns different Q than Q-learning on stochastic environments.
- Q-learning converges to optimal Q on tabular environments (CliffWalking).
- TD(lambda=0) reduces to TD(0). TD(lambda=1) reduces to MC.

**Module 03: Policy Gradients**

Exercises:
- `reinforce_loss(log_probs, returns)` -- -sum(log_prob * G_t).
- `compute_returns(rewards, gamma)` -- G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}.
- `generalized_advantage_estimation(rewards, values, gamma, gae_lambda)` -- GAE: A_t = sum_{l=0}^{T-t} (gamma*lambda)^l * delta_{t+l}.
- `actor_critic_loss(log_probs, advantages, values, returns, value_coeff)` -- Combined actor-critic loss.
- `entropy_bonus(action_probs)` -- -sum(p * log p) for exploration.

Tests:
- REINFORCE loss gradient: positive advantage increases action probability, negative decreases it.
- Returns are correctly discounted (verify against manual computation).
- GAE with lambda=0 is TD error. GAE with lambda=1 is MC advantage.
- Entropy bonus is maximized for uniform distribution. Zero for deterministic.
- Actor-critic converges on CartPole within 500 episodes.

**Module 04: PPO & Modern Policy Gradient**

Exercises:
- `ppo_clip_loss(ratio, advantage, epsilon)` -- min(r*A, clip(r, 1-eps, 1+eps)*A).
- `trust_region_check(old_policy, new_policy, kl_threshold)` -- Check KL(old || new) < threshold.
- `compute_ratio(new_log_prob, old_log_prob)` -- exp(new - old).
- `ppo_value_loss(predicted, target, old_predicted, clip_range)` -- Clipped value loss.

Tests:
- PPO clip: when advantage > 0, ratio is clipped above at 1+eps. When advantage < 0, clipped below at 1-eps.
- Ratio is 1 when old and new policies are identical.
- PPO agent solves CartPole (reward > 195 over 100 episodes).
- Value loss clipping prevents large value function updates.

Mini-project: Full PPO implementation on CartPole and LunarLander. Reproduce the "implementation matters" tricks (observation normalization, reward scaling, gradient clipping, etc.).

**Module 05: Model-Based RL**

Exercises:
- `train_dynamics_model(states, actions, next_states)` -- Learn transition model s' = f(s, a).
- `model_predictive_control(dynamics_model, current_state, horizon, n_candidates)` -- Random shooting MPC.
- `dreamer_world_model_loss(reconstruction, kl_loss, reward_prediction)` -- RSSM loss.
- `dyna_q_step(Q, model, real_buffer, model_buffer, n_model_steps)` -- Dyna-Q: real + model-generated experience.

Tests:
- Dynamics model prediction error decreases with more training data.
- MPC plans improve with longer horizon (up to environment horizon).
- Dyna-Q converges faster than pure model-free Q-learning (measure episodes to convergence).

**Module 06: Offline RL**

Exercises:
- `cql_loss(Q_values, dataset_actions, all_actions, alpha)` -- CQL: standard Q loss + alpha * (logsumexp(Q) - E_data[Q]).
- `iql_value_loss(Q, V, tau)` -- IQL: expectile regression loss.
- `distributional_shift_metric(behavior_policy, learned_policy, states)` -- KL divergence between policies.
- `pessimistic_value_estimate(Q_ensemble, actions)` -- Mean - beta * std across ensemble.

Tests:
- CQL pushes down Q-values for OOD actions (actions not in dataset have lower Q).
- IQL: tau=0.5 gives mean, tau->1 gives max.
- Pessimistic estimate is always <= mean estimate.

**Module 07: Capstone -- RL Agent**

**Project:** Build a PPO agent that solves a continuous control task (Mujoco HalfCheetah or Ant via Gymnasium).

Requirements:
1. Implement PPO with all modern tricks (GAE, clipping, entropy bonus, value clipping, observation normalization).
2. Implement a model-based component: learn a dynamics model and use it for data augmentation (Dyna-style).
3. Compare model-free PPO vs model-augmented PPO: sample efficiency and final performance.
4. Produce learning curves with confidence intervals (5 random seeds).
5. Implement and compare against CQL on an offline dataset collected from the trained agent.
6. Write a 2-page report.

---

#### Computer Vision Track

**Module 01: Object Detection**

Exercises:
- `compute_iou(box1, box2)` -- Intersection over Union for axis-aligned bounding boxes.
- `non_max_suppression(boxes, scores, iou_threshold)` -- NMS to remove duplicate detections.
- `anchor_box_generation(feature_map_size, aspect_ratios, scales)` -- Generate anchor boxes.
- `yolo_loss(predictions, targets, lambda_coord, lambda_noobj)` -- YOLO v1 loss function.
- `detr_bipartite_matching(predicted, targets, cost_fn)` -- Hungarian matching for DETR.

Tests:
- IoU of identical boxes is 1. Non-overlapping boxes is 0. IoU in [0, 1].
- NMS with threshold 0.5: overlapping boxes with lower score are removed.
- Anchor boxes: correct number generated (H * W * num_ratios * num_scales).
- YOLO loss: coordinate loss weight > classification loss weight.
- Bipartite matching: each prediction matched to at most one target (and vice versa).

**Module 02: Semantic Segmentation**

Exercises:
- `unet_encoder_block(in_channels, out_channels)` -- Conv-BN-ReLU-Conv-BN-ReLU-MaxPool.
- `unet_decoder_block(in_channels, out_channels)` -- Upsample-Concat(skip)-Conv-BN-ReLU-Conv-BN-ReLU.
- `dice_loss(predictions, targets)` -- 1 - (2 * |P intersection T|) / (|P| + |T|).
- `mean_iou(predictions, targets, num_classes)` -- Mean IoU across all classes.

Tests:
- U-Net encoder: output spatial dims are half of input.
- U-Net decoder: output spatial dims are double of input.
- Dice loss: 0 for perfect prediction, 1 for worst.
- mIoU: 1.0 for perfect segmentation. Handles class imbalance.

**Module 03: Vision Transformers**

Exercises:
- `image_to_patches(image, patch_size)` -- Split H x W image into (H/P * W/P) patches of size P x P.
- `patch_embedding(patches, embedding_dim)` -- Linear projection of flattened patches.
- `vit_cls_token(patch_embeddings)` -- Prepend CLS token and add position embeddings.
- `vit_forward(image, patch_size, num_layers, num_heads, d_model)` -- Full ViT forward pass.

Tests:
- Patch extraction: correct number of patches. Patches reconstruct to original image.
- Patch embedding: output shape is (num_patches, embedding_dim).
- ViT on CIFAR-10: achieves > 85% accuracy with sufficient training (mini-project target).

Mini-project: ViT from scratch on CIFAR-10. Compare against ResNet of similar parameter count.

**Module 04: 3D Vision & NeRFs**

Exercises:
- `ray_generation(camera_intrinsics, camera_pose, image_height, image_width)` -- Generate rays for all pixels.
- `volume_rendering(densities, colors, t_values)` -- C(r) = sum T_i * (1 - exp(-sigma_i * delta_i)) * c_i.
- `positional_encoding_nerf(x, num_frequencies)` -- gamma(p) = (sin(2^0 pi p), cos(2^0 pi p), ...).
- `nerf_mlp_forward(positions, directions, model)` -- Two-stage MLP: position -> density + feature -> color.

Tests:
- Ray directions are unit vectors.
- Volume rendering: transmittance T decreases monotonically along ray. Final transmittance + sum of weights = 1.
- Positional encoding: output dimension = 2 * input_dim * num_frequencies.
- NeRF MLP: density is non-negative (ReLU output). Colors are in [0, 1] (sigmoid output).

**Module 05: Capstone -- CV Project**

**Project:** Build a DETR-style object detector from scratch.

Requirements:
1. Implement the backbone (ResNet-50 feature extractor).
2. Implement the transformer encoder-decoder.
3. Implement bipartite matching loss (Hungarian algorithm).
4. Train on PASCAL VOC or a subset of COCO.
5. Evaluate with mAP. Target: mAP > 30 on VOC.
6. Visualize detections on test images.
7. Compare against a YOLO baseline (can use pretrained).

---

#### ML Systems Track

**Module 01: Distributed Training**

Exercises:
- `all_reduce_ring(gradients, world_size)` -- Simulate ring all-reduce communication pattern.
- `data_parallel_split(batch, world_size)` -- Split batch across workers, simulate gradient sync.
- `pipeline_parallel_schedule(num_microbatches, num_stages)` -- Generate GPipe schedule (F then B).
- `tensor_parallel_linear(input, weight_shards, world_size)` -- Column-parallel linear layer.
- `communication_cost(message_size, bandwidth, latency, world_size, algorithm)` -- Model comm cost.

Tests:
- All-reduce: result is identical to summing all gradients locally.
- Data parallel: each worker processes batch_size/world_size samples.
- Pipeline schedule: no stage is idle for more than (num_stages - 1) microbatches (bubble size).
- Tensor parallel: output is identical to non-parallelized linear layer.

**Module 02: Quantization**

Exercises:
- `symmetric_quantize(tensor, num_bits)` -- Map float to int: q = round(x / scale), scale = max(|x|) / (2^(b-1) - 1).
- `asymmetric_quantize(tensor, num_bits)` -- Map with zero point: q = round(x / scale) + zero_point.
- `dequantize(quantized, scale, zero_point)` -- Reconstruct: x_approx = (q - zero_point) * scale.
- `quantization_error(original, quantized_dequantized)` -- MSE and max absolute error.
- `mixed_precision_forward(model, input, sensitive_layers)` -- FP16 for most, FP32 for sensitive layers.

Tests:
- Quantize then dequantize: max error bounded by scale/2.
- 8-bit quantization error < 1% relative to FP32 on typical weight distributions.
- 4-bit quantization: larger error, but still acceptable for inference (verify on a small model).
- Mixed precision: output matches FP32 within tolerance.

Mini-project: Quantize a BERT model to INT8 and INT4. Benchmark latency, memory, and accuracy on GLUE tasks. Compare GPTQ vs naive PTQ.

**Module 03: Knowledge Distillation**

Exercises:
- `distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)` -- alpha * KL(softmax(t/T), softmax(s/T)) * T^2 + (1-alpha) * CE(s, labels).
- `soft_labels(teacher_logits, temperature)` -- Softened probability distribution.
- `feature_distillation_loss(student_features, teacher_features)` -- MSE between intermediate representations.
- `attention_transfer_loss(student_attention, teacher_attention)` -- Match attention maps.

Tests:
- Distillation loss with T=1 and alpha=0 reduces to standard cross-entropy.
- Higher temperature produces softer distributions (lower entropy difference between classes).
- Distilled student achieves higher accuracy than training from scratch (verify on MNIST: student matches teacher within 2%).

**Module 04: Serving & Inference**

Exercises:
- `kv_cache_update(cache, new_keys, new_values, layer_idx)` -- Append new KV pairs for autoregressive generation.
- `continuous_batching(requests, max_batch_size, max_seq_len)` -- Schedule requests with different lengths.
- `speculative_decoding_step(draft_model, target_model, input_ids, gamma)` -- Draft gamma tokens, verify with target.
- `onnx_export_model(pytorch_model, dummy_input, output_path)` -- Export and verify ONNX graph.

Tests:
- KV cache: generation with cache produces same output as without cache (but faster).
- Continuous batching: throughput >= naive batching throughput.
- Speculative decoding: output distribution is identical to target model distribution (exact).
- ONNX model: output matches PyTorch model within FP32 tolerance.

Mini-project: Build an inference server with continuous batching and KV caching. Benchmark tokens/second on a 7B model. Compare naive vs optimized serving.

**Module 05: Profiling & Optimization**

Exercises:
- `profile_forward_pass(model, input, device)` -- Profile with PyTorch profiler, return time per layer.
- `memory_estimate(model, batch_size, seq_len)` -- Estimate peak GPU memory: params + activations + gradients.
- `gradient_checkpointing_savings(model, num_layers)` -- Compute memory savings: O(sqrt(L)) vs O(L).
- `mixed_precision_speedup(model, input)` -- Compare FP32 vs FP16 throughput.

Tests:
- Memory estimate: within 20% of actual measured peak memory.
- Gradient checkpointing: reduces peak memory by at least 40% on deep models.
- Mixed precision: at least 1.5x speedup on GPU with tensor cores.

**Module 06: Capstone -- ML System**

**Project:** Build a complete model serving system for a 7B parameter language model.

Requirements:
1. Implement INT4 quantization (GPTQ-style) of the model.
2. Implement KV cache with PagedAttention concepts.
3. Build a REST API server with continuous batching.
4. Implement speculative decoding with a small draft model.
5. Benchmark: measure tokens/second, time-to-first-token, throughput at various concurrency levels.
6. Profile memory usage and identify bottlenecks.
7. Compare your system against vLLM or TGI on the same hardware.
8. Write a 2-page report with benchmarks.

---

#### Bayesian ML Track

**Module 01: Gaussian Processes**

Exercises:
- `gp_prior_sample(X, kernel_fn, n_samples)` -- Sample functions from GP prior: f ~ N(0, K(X,X)).
- `gp_posterior(X_train, y_train, X_test, kernel_fn, noise_var)` -- Posterior mean and variance.
- `rbf_kernel_with_hyperparams(x1, x2, length_scale, signal_var)` -- ARD RBF kernel.
- `log_marginal_likelihood(X, y, kernel_fn, noise_var)` -- log p(y|X) = -1/2 y^T K_y^{-1} y - 1/2 log|K_y| - n/2 log(2pi).
- `gp_hyperparameter_optimization(X, y, kernel_fn)` -- Optimize kernel hyperparams by maximizing marginal likelihood.

Tests:
- GP prior samples: mean over many samples converges to 0. Variance matches kernel diagonal.
- GP posterior: passes through training points (within noise). Variance is 0 at training points (noiseless case).
- Marginal likelihood: higher for well-fitting kernels. Can distinguish RBF vs linear kernel on nonlinear data.
- Posterior variance increases away from training points.
- Cholesky decomposition is numerically stable (add jitter if needed).

Mini-project: GP regression on 1D and 2D data with uncertainty visualization. Implement kernel selection via marginal likelihood. Demonstrate that GPs provide well-calibrated uncertainty.

**Module 02: Variational Inference**

Exercises:
- `mean_field_elbo(log_joint, q_samples, q_log_prob)` -- ELBO = E_q[log p(x,z)] - E_q[log q(z)].
- `cavi_update_gaussian(other_params, data, prior_params)` -- Coordinate ascent VI for Gaussian mixture.
- `stochastic_vi_step(elbo_gradient_estimate, params, lr)` -- SVI gradient step.
- `reparameterized_gradient(mu, sigma, n_samples)` -- Gradient of ELBO via reparameterization.
- `black_box_vi_gradient(log_joint, q_sample, q_log_prob)` -- Score function estimator (REINFORCE for VI).

Tests:
- ELBO is a lower bound on log evidence (verify on tractable model).
- CAVI converges: ELBO increases monotonically.
- Reparameterized gradient has lower variance than score function estimator (measure over many samples).
- Mean-field VI on a known posterior: KL(q||p) decreases with iterations.

**Module 03: MCMC Methods**

Exercises:
- `metropolis_hastings_step(current, target_log_prob, proposal_fn)` -- MH accept/reject step.
- `gibbs_sampling_step(state, conditional_samplers)` -- Full sweep of Gibbs sampling.
- `hamiltonian_dynamics(q, p, grad_U, epsilon, L)` -- Leapfrog integrator for HMC.
- `hmc_step(current_q, target_log_prob, target_grad, epsilon, L)` -- Full HMC step with MH correction.
- `effective_sample_size(chain)` -- ESS accounting for autocorrelation.
- `gelman_rubin_diagnostic(chains)` -- R-hat convergence diagnostic.

Tests:
- MH: detailed balance holds (verify empirically on small discrete space).
- MH on known Gaussian: samples have correct mean and variance (KS test).
- HMC leapfrog: Hamiltonian is approximately conserved (|H_new - H_old| < epsilon).
- HMC acceptance rate is high (> 0.65) with well-tuned step size.
- ESS: independent samples have ESS = n. Autocorrelated have ESS < n.
- Gelman-Rubin: R-hat < 1.1 for converged chains. R-hat >> 1 for non-converged.

Mini-project: Implement HMC from scratch. Apply to Bayesian logistic regression. Compare mixing with MH and Gibbs. Verify calibration of posterior predictive.

**Module 04: Bayesian Deep Learning**

Exercises:
- `mc_dropout_predict(model, x, n_forward_passes, dropout_rate)` -- Predictive mean and variance via MC Dropout.
- `bayes_by_backprop_loss(log_likelihood, kl_weight, weight_prior, weight_posterior)` -- ELBO for weight uncertainty.
- `deep_ensemble_predict(models, x)` -- Mean and variance from ensemble.
- `calibration_error(predicted_probs, true_labels, n_bins)` -- Expected calibration error.

Tests:
- MC Dropout uncertainty: higher for OOD inputs than in-distribution.
- Deep ensemble: uncertainty increases away from training data.
- Calibration: well-calibrated model has ECE < 0.05.
- Bayes by Backprop: KL term regularizes weights toward prior.

**Module 05: Bayesian Optimization**

Exercises:
- `expected_improvement(mu, sigma, best_so_far)` -- EI = (mu - f_best) * Phi(z) + sigma * phi(z).
- `upper_confidence_bound(mu, sigma, beta)` -- UCB = mu + beta * sigma.
- `probability_of_improvement(mu, sigma, best_so_far)` -- PI = Phi((mu - f_best) / sigma).
- `bayesian_optimization_loop(objective, bounds, n_iterations, acquisition_fn)` -- Full BO loop with GP surrogate.
- `thompson_sampling(gp_posterior, n_candidates)` -- Sample from GP posterior, maximize.

Tests:
- EI is non-negative. EI is 0 when sigma = 0 and mu <= f_best.
- UCB: higher beta means more exploration (wider search).
- BO loop: finds optimum of Branin function within 50 evaluations (vs 200 for random search).
- Thompson sampling: samples are from the correct GP posterior.

Mini-project: Build a BO system for hyperparameter tuning of a neural network. Compare against random search and grid search on a real task. Demonstrate efficiency gains.

**Module 06: Capstone -- Bayesian Project**

**Project:** Bayesian neural network for medical uncertainty quantification.

Requirements:
1. Implement a BNN using both MC Dropout and Bayes by Backprop on a medical dataset (Skin Cancer MNIST or similar).
2. Implement deep ensembles as a baseline.
3. Evaluate: predictive accuracy, calibration (ECE), OOD detection (AUROC on OOD data).
4. Demonstrate that the model says "I don't know" on OOD examples.
5. Compare uncertainty quality: MC Dropout vs Bayes by Backprop vs Deep Ensembles.
6. Reliability diagrams for each method.
7. Write a 2-page report discussing which method provides the most useful uncertainty estimates.

---

## Phase 3E: Paper Reading & Reproduction

### Directory Structure

```
spiral_3/3e_paper_reproduction/
├── README.md                              # How to read papers, reproduction methodology
├── 01_reading_methodology/
│   ├── README.md                          # Lesson: 3-pass method, finding key contributions
│   │                                      #   How to evaluate: novelty, experimental rigor, limitations
│   │                                      #   Template: paper reading worksheet
│   └── reading_template.md               # Reusable template for paper notes
├── 02_paper_attention_is_all_you_need/
│   ├── README.md                          # Paper-specific notes, what to focus on
│   │                                      #   Paper: Vaswani et al. 2017 (arXiv:1706.03762)
│   ├── reading_notes.md                  # Template for the learner's notes
│   ├── reproduce.py                       # Stub: full Transformer for machine translation
│   ├── data.py                            # Stub: WMT data loading and tokenization
│   ├── train.py                           # Stub: training loop matching paper's setup
│   ├── evaluate.py                        # Stub: BLEU score evaluation
│   └── tests/
│       ├── test_reproduce.py             # Verify architecture dimensions, attention patterns
│       └── test_evaluate.py              # BLEU computation tests
├── 03_paper_denoising_diffusion/
│   ├── README.md                          # Paper-specific notes
│   │                                      #   Paper: Ho et al. 2020 DDPM (arXiv:2006.11239)
│   ├── reading_notes.md
│   ├── reproduce.py                       # Stub: DDPM with U-Net
│   ├── train.py                           # Stub: training loop
│   ├── sample.py                          # Stub: sampling with DDPM and DDIM
│   ├── evaluate.py                        # Stub: FID score computation
│   └── tests/
│       ├── test_reproduce.py
│       └── test_evaluate.py
├── 04_paper_dpo/
│   ├── README.md                          # Paper-specific notes
│   │                                      #   Paper: Rafailov et al. 2023 DPO (arXiv:2305.18290)
│   ├── reading_notes.md
│   ├── reproduce.py                       # Stub: DPO training pipeline
│   ├── data.py                            # Stub: preference data preparation
│   ├── evaluate.py                        # Stub: win-rate evaluation
│   └── tests/
│       ├── test_reproduce.py
│       └── test_evaluate.py
└── 05_capstone_extend_paper/
    ├── README.md                          # Instructions for the final capstone
    ├── proposal.md                        # Template: 1-page research proposal
    ├── experiment.py                      # Stub: your extension experiment
    ├── analyze.py                         # Stub: result analysis
    ├── report.md                          # Template: 4-page write-up
    └── tests/
        └── test_experiment.py
```

### Module Details

#### Module 01: Paper Reading Methodology

The README teaches:
- The 3-pass reading method (Keshav 2007): survey, comprehend, reproduce.
- How to identify the core contribution vs incremental improvements.
- How to evaluate experimental methodology (baselines, ablations, statistical significance).
- How to spot limitations and find open questions.
- How to take structured notes that enable reproduction.

The reading template covers: problem statement, prior work, key insight, method, experiments, ablations, limitations, open questions, and "what I would try next."

#### Module 02: Reproduce "Attention Is All You Need"

**Paper:** Vaswani et al. 2017 (arXiv:1706.03762)

**Why this paper:** It is foundational to modern ML. By this point the learner has built Transformer components in earlier modules; now they reproduce the full system end-to-end for a real task.

**Reproduction requirements:**
1. Implement the full encoder-decoder Transformer for machine translation.
2. Use the WMT 2014 English-German dataset (or a smaller parallel corpus).
3. Implement the exact training schedule from the paper: warmup + inverse square root decay.
4. Implement label smoothing (epsilon = 0.1).
5. Evaluate with BLEU score. The paper achieves 28.4 BLEU; target BLEU > 20 with limited compute.
6. Ablation: reproduce at least one row from Table 3 (varying heads, key dimension, etc.).
7. Attention visualization: plot attention heatmaps for sample translations.

**Tests:**
- Model dimensions match paper: d_model=512, d_ff=2048, h=8, d_k=64.
- Label smoothing loss: with epsilon=0, reduces to standard cross-entropy.
- Learning rate schedule: increases linearly for warmup_steps, then decreases.
- BLEU score computation matches sacrebleu on known reference/hypothesis pairs.
- Encoder self-attention is symmetric under input permutation (verify equivariance).

#### Module 03: Reproduce "Denoising Diffusion Probabilistic Models"

**Paper:** Ho et al. 2020 (arXiv:2006.11239)

**Why this paper:** Diffusion models are the leading generative paradigm. The paper is clear, well-written, and reproducible. The learner has implemented components in Phase 3A; now they put it all together.

**Reproduction requirements:**
1. Implement the U-Net architecture from the paper (with time embeddings, attention blocks).
2. Implement the full training algorithm (Algorithm 1 from the paper).
3. Implement the sampling algorithm (Algorithm 2).
4. Train on CIFAR-10 for at least 200k steps.
5. Implement DDIM sampling and compare sample quality at 50 vs 1000 steps.
6. Compute FID score. Target: FID < 50 (paper achieves 3.17).
7. Extension: implement classifier-free guidance (Ho & Salimans 2022, arXiv:2207.12598).

**Tests:**
- U-Net: output shape matches input shape at all timesteps.
- Forward process at t=T produces near-standard-Gaussian samples.
- Sampling: with enough steps, produces valid images (pixel values in [0, 1]).
- FID computation: verified on known real/fake distributions.

#### Module 04: Reproduce "Direct Preference Optimization"

**Paper:** Rafailov et al. 2023 (arXiv:2305.18290)

**Why this paper:** DPO is a landmark alignment technique that simplifies RLHF. It connects reward modeling, KL-constrained optimization, and direct policy learning in a single elegant framework. Mathematically rich and practically important.

**Reproduction requirements:**
1. Implement the DPO loss derivation: show how the optimal policy under KL-constrained reward maximization leads to the DPO objective.
2. Implement DPO training on a small language model (GPT-2 medium or similar).
3. Use the Anthropic HH-RLHF dataset or UltraFeedback for preference pairs.
4. Compare against a RLHF baseline (reward model + PPO).
5. Evaluate: win rate against SFT baseline using GPT-4 as judge (or a fixed evaluator).
6. Ablation: vary beta and plot the accuracy/diversity tradeoff.
7. Extension: implement IPO (Azar et al. 2023, arXiv:2310.12036) and compare.

**Tests:**
- DPO loss is well-defined (no NaN/Inf) for valid log probabilities.
- With beta -> 0, DPO degenerates (no regularization). With beta -> infinity, no learning.
- DPO gradient: positive for chosen completions, negative for rejected.
- Win rate against SFT baseline > 55%.

#### Module 05: Capstone -- Extend a Paper

**Project:** Choose one of the three reproduced papers and design an original extension.

**Requirements:**
1. Write a 1-page research proposal: hypothesis, method, expected outcome.
2. Implement the extension.
3. Run experiments with proper baselines and ablations.
4. Analyze results statistically (confidence intervals, significance tests where applicable).
5. Write a 4-page report in conference paper format (intro, related work, method, experiments, conclusion).

**Suggested extension ideas:**
- For Transformers: implement and evaluate Mixture of Experts layers; study scaling behavior.
- For DDPM: implement v-prediction parameterization and compare against epsilon-prediction; study few-step distillation.
- For DPO: implement and compare DPO, IPO, and KTO on the same dataset; study the effect of preference data quality.

**Tests:**
- Extension code runs without error.
- Baseline reproduction matches previous module's results.
- Extension produces measurably different results from baseline (positive or negative).

---

## Cross-Cutting Notes

### Prerequisites Between Phases

- **3A** can start independently (builds on Spiral 2 deep learning knowledge).
- **3B** can start independently, but Module 04 (RLHF/DPO) benefits from 3D-RL Module 03 (policy gradients).
- **3C** can start independently; it is pure math. Recommended alongside 3A/3B, not after.
- **3D** tracks can be done in any order. RL is the most self-contained. ML Systems benefits from having trained models in 3A/3B to optimize. Bayesian ML benefits from 3C Module 03 (kernels).
- **3E** should be done last. It synthesizes everything.

### Testing Philosophy for Spiral 3

Tests at this level verify **mathematical properties**, not just correctness:
- **Convergence:** Loss decreases. Algorithm converges to known solution on tractable problems.
- **Bounds:** Generalization bounds are valid (bound >= actual error). KL divergence is non-negative.
- **Invariances:** Permutation invariance/equivariance of GNNs. Translation equivariance of convolutions.
- **Limits:** When a parameter goes to 0 or infinity, behavior matches theoretical prediction.
- **Statistical:** Distributions match expected (KS test, chi-squared test). Estimators are unbiased or have bounded bias.

### Estimated Module Counts

| Phase | Modules | Est. Hours |
|-------|---------|-----------|
| 3A: Advanced Deep Learning | 6 (5 + capstone) | 60-80 |
| 3B: NLP & Language Models | 6 (5 + capstone) | 60-80 |
| 3C: Advanced Math & Theory | 6 (5 + capstone) | 50-70 |
| 3D-RL: Reinforcement Learning | 7 (6 + capstone) | 70-90 |
| 3D-CV: Computer Vision | 5 (4 + capstone) | 50-70 |
| 3D-Sys: ML Systems | 6 (5 + capstone) | 60-80 |
| 3D-Bayes: Bayesian ML | 6 (5 + capstone) | 60-80 |
| 3E: Paper Reproduction | 5 (1 method + 3 papers + capstone) | 80-120 |
| **Total** | **47 modules** | **490-670** |

---

### Critical Files for Implementation

- `/Users/kkondratuk/.claude/plans/luminous-swinging-quilt.md` - Scaffolding pattern to follow (self-contained dirs, stubs-only, TDD, mini-projects, math callouts, style notes)
- `/Users/kkondratuk/PycharmProjects/LearningML/docs/plans/2026-03-22-ml-curriculum-design.md` - Master curriculum design with phase descriptions and repo structure
- `/Users/kkondratuk/PycharmProjects/LearningML/docs/plans/2026-03-22-phase-1a-scientific-python.md` - Reference implementation plan showing the exact pattern for exercise stubs, test files, and commit workflow
- `/Users/kkondratuk/PycharmProjects/LearningML/pyproject.toml` - Will need Spiral 3 dependencies added (torch, transformers, gymnasium, pyro-ppl, gpytorch, etc.)