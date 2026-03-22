# ML Learning Curriculum Design

## Learner Profile

- Experienced software developer, native in Go
- Basic Python familiarity (syntax known, ecosystem not yet fluent)
- Rusty college math (calculus, linear algebra taken but needs refresher)
- Motivated by deep understanding of ML theory and practice
- Learns best by building first, then understanding the theory
- No time constraints — optimize for depth over speed

## Approach: Spiral Curriculum

Three passes through the ML landscape at increasing depth:

1. **Spiral 1 — Foundations & Intuition:** Get productive, build working models, develop intuition
2. **Spiral 2 — Depth & Rigor:** Revisit with math, implement from scratch, understand the theory
3. **Spiral 3 — Advanced Topics & Research Frontier:** Read and implement papers, specialize

Each module follows a consistent pattern: **build something -> understand what you built -> solidify with exercises**.

Math is introduced just-in-time in Spiral 1, given a dedicated deep pass in Spiral 2, and extended to research-level in Spiral 3.

## Spiral 1: Foundations & Intuition

Goal: Get productive with Python's scientific stack and build working ML models. Develop intuition for what algorithms do without full mathematical rigor.

### Phase 1A — Python for Scientific Computing

- Python idioms for Go developers (generators, decorators, duck typing, context managers)
- NumPy: array operations, broadcasting, vectorization
- Pandas: data loading, cleaning, transformation
- Matplotlib/Seaborn: visualization as a thinking tool
- **Project:** Exploratory data analysis on a real dataset

### Phase 1B — Classical ML (Intuition Pass)

- Supervised learning: linear regression, logistic regression, decision trees, random forests, SVMs
- Unsupervised learning: k-means, PCA, hierarchical clustering
- Model evaluation: train/test splits, cross-validation, bias-variance tradeoff
- Feature engineering basics
- Using scikit-learn as the workhorse
- **Project:** End-to-end ML pipeline (housing prices, classification task)

### Phase 1C — Neural Networks (Intuition Pass)

- What is a neural network? Build a simple one from scratch in NumPy
- PyTorch fundamentals: tensors, autograd, modules, training loops
- CNNs for image tasks, RNNs for sequence tasks
- Transfer learning with pretrained models
- **Project:** Image classifier or text sentiment model

### Phase 1D — Math Refresher (Just-in-Time)

Woven throughout 1A-1C, not a separate block:

- Linear algebra: vectors, matrices, dot products, eigenvalues (when they come up in PCA, neural nets)
- Calculus: derivatives, chain rule, gradients (when they come up in backprop)
- Probability: distributions, Bayes' theorem, conditional probability (when they come up in classification)

## Spiral 2: Depth & Rigor

Goal: Revisit everything from Spiral 1 with mathematical depth. Implement algorithms from scratch. Understand the theory well enough to derive, not just apply.

### Phase 2A — Mathematical Foundations (Dedicated Pass)

- Linear algebra: matrix decompositions (SVD, eigendecomposition, QR), vector spaces, linear transformations, positive definiteness
- Multivariate calculus: partial derivatives, Jacobians, Hessians, Taylor approximations
- Probability & statistics: MLE, MAP estimation, conjugate priors, information theory (entropy, KL divergence)
- Optimization: gradient descent variants, convexity, Lagrange multipliers, constrained optimization
- **Exercises:** Implement each concept in NumPy. Prove key results on paper.

### Phase 2B — Classical ML (Theory Pass)

- Derive linear regression from MLE and MAP perspectives
- Logistic regression as a generalized linear model
- SVMs: dual formulation, kernel trick, why it works
- Decision trees: information gain, Gini impurity — derive splitting criteria
- Ensemble methods: why bagging reduces variance, why boosting reduces bias
- Bias-variance tradeoff formal decomposition proof
- Regularization theory: L1 vs L2, why L1 produces sparsity
- **Project:** Implement 3-4 algorithms from scratch, compare against library versions

### Phase 2C — Deep Learning (Theory Pass)

- Backpropagation derived and implemented manually
- Activation functions: why ReLU works, vanishing/exploding gradients mathematically
- Optimization landscape: SGD, momentum, Adam — derive update rules
- Batch normalization, dropout, weight initialization — why each works
- CNN internals: convolution as a linear operation, receptive fields, feature hierarchies
- RNN/LSTM/GRU: gradient flow problem, gating mechanisms derived
- Attention mechanism and Transformers from first principles
- **Project:** Build a Transformer from scratch in PyTorch

### Phase 2D — Practical ML Engineering

- Experiment tracking (MLflow or Weights & Biases)
- Hyperparameter tuning: grid search, random search, Bayesian optimization
- Data pipelines and preprocessing at scale
- Model debugging: what to do when your model doesn't train
- GPU programming basics with CUDA/PyTorch
- **Project:** Take Spiral 1 projects and make them production-quality

## Spiral 3: Advanced Topics & Research Frontier

Goal: Reach the level where you can read and implement current papers. Develop specialization.

### Phase 3A — Advanced Deep Learning

- Generative models: VAEs (derive the ELBO), GANs (minimax formulation, training dynamics, mode collapse)
- Diffusion models: forward/reverse process, score matching, denoising
- Graph neural networks: message passing, spectral vs spatial approaches
- Self-supervised learning: contrastive learning, masked modeling
- **Project:** Implement a generative model from a recent paper

### Phase 3B — NLP & Language Models

- Word embeddings: Word2Vec, GloVe — derive objectives
- Transformer deep dive: positional encodings, multi-head attention scaling
- Pretraining objectives: masked LM, causal LM, T5-style
- Fine-tuning, RLHF, DPO — the alignment pipeline
- Prompt engineering and in-context learning from a theoretical lens
- Scaling laws and emergent abilities
- **Project:** Fine-tune a language model on a custom task

### Phase 3C — Advanced Math & Theory

- Statistical learning theory: VC dimension, PAC learning, generalization bounds
- Information geometry: Fisher information, natural gradient
- Kernel methods and reproducing kernel Hilbert spaces
- Convex optimization: duality theory, KKT conditions, proximal methods
- Measure-theoretic probability (as much as is useful for ML)
- **Exercises:** Work through proofs from key papers

### Phase 3D — Specialization Tracks

All tracks are of interest; pick 1-2 to go deep when the time comes:

- **Reinforcement learning:** MDPs, policy gradients, Q-learning, PPO, model-based RL
- **Computer vision:** object detection, segmentation, 3D vision, NeRFs
- **ML systems:** distributed training, model serving, quantization, distillation
- **Bayesian ML:** Gaussian processes, variational inference, MCMC methods
- **Model performance:** profiling, optimization, benchmarking
- **Project:** Research-level implementation in chosen track(s)

### Phase 3E — Paper Reading & Reproduction

- How to read ML papers effectively
- Reproduce 2-3 influential papers from scratch
- Identify open problems, run experiments, write up findings
- **Capstone:** Reproduce a recent paper and extend it with your own experiment

## Repo Structure

```
LearningML/
├── pyproject.toml
├── spiral_1/
│   ├── 1a_scientific_python/
│   ├── 1b_classical_ml/
│   ├── 1c_neural_networks/
│   └── 1d_math_refresher/
├── spiral_2/
│   ├── 2a_math_foundations/
│   ├── 2b_classical_ml_theory/
│   ├── 2c_deep_learning_theory/
│   └── 2d_ml_engineering/
├── spiral_3/
│   ├── 3a_advanced_deep_learning/
│   ├── 3b_nlp_language_models/
│   ├── 3c_advanced_math/
│   ├── 3d_specialization/
│   └── 3e_paper_reproduction/
├── projects/
├── docs/
│   └── plans/
└── utils/
```

## Collaboration Model

- Each module: brief concept intro, build something, explain theory behind what was built
- Exercises are hands-on Python in this repo — real code committed to git
- Math introduced through code first (NumPy implementations), then formalized
- Connections drawn to Go patterns when helpful
- Learner drives the pace — can speed up, slow down, or go deeper at any point
