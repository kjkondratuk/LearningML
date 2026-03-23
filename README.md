# LearningML

A hands-on ML curriculum designed for experienced software developers. Starts with Python fundamentals and progresses through classical ML, deep learning, and research-level topics. Every concept is learned by building something first, then understanding the theory behind it.

## Who This Is For

You're an experienced developer (in any language) who wants a deep understanding of machine learning --- not just how to call APIs, but how and why things work. You're comfortable writing code but may be rusty on college math. You learn best by building things.

## How It's Organized

The curriculum uses a **spiral approach**: three passes through the same material at increasing depth.

### Spiral 1: Foundations & Intuition

Build working models and develop intuition. Math is introduced just-in-time.

| Phase | What You'll Build | Modules |
|-------|-------------------|---------|
| [1A: Scientific Python](spiral_1/1a_scientific_python/) | Data analysis with NumPy, Pandas, Matplotlib | 5 |
| [1B: Classical ML](spiral_1/1b_classical_ml/) | Regression, trees, SVMs, clustering with scikit-learn | 9 |
| [1C: Neural Networks](spiral_1/1c_neural_networks/) | Neural nets from scratch in NumPy, then PyTorch CNNs/RNNs | 8 |
| [1D: Math Refresher](spiral_1/1d_math_refresher/) | Linear algebra, calculus, probability (reference module) | 4 |

### Spiral 2: Depth & Rigor

Revisit everything with mathematical depth. Implement algorithms from scratch --- no scikit-learn.

| Phase | What You'll Build | Modules |
|-------|-------------------|---------|
| [2A: Math Foundations](spiral_2/2a_math_foundations/) | SVD, Jacobians, MLE/MAP, optimization methods, information theory | 6 |
| [2B: Classical ML Theory](spiral_2/2b_classical_ml_theory/) | Derive and implement LR, SVMs, trees, regularization from first principles | 8 |
| [2C: Deep Learning Theory](spiral_2/2c_deep_learning_theory/) | Backprop, Adam, batch norm derived; Transformer from scratch | 8 |
| [2D: ML Engineering](spiral_2/2d_ml_engineering/) | MLflow, hyperparameter tuning, data pipelines, GPU programming | 6 |

### Spiral 3: Advanced Topics & Research Frontier

Read and implement papers. Specialize in areas that interest you.

| Phase | What You'll Build | Modules |
|-------|-------------------|---------|
| [3A: Advanced Deep Learning](spiral_3/3a_advanced_deep_learning/) | VAEs, GANs, diffusion models, GNNs | 6 |
| [3B: NLP & Language Models](spiral_3/3b_nlp_language_models/) | Word2Vec through RLHF/DPO and scaling laws | 6 |
| [3C: Advanced Math](spiral_3/3c_advanced_math/) | Learning theory, information geometry, RKHS, convex optimization | 6 |
| [3D: Specialization](spiral_3/3d_specialization/) | RL, computer vision, ML systems, Bayesian ML (4 tracks) | 24 |
| [3E: Paper Reproduction](spiral_3/3e_paper_reproduction/) | Reproduce Attention Is All You Need, DDPM, DPO; extend a paper | 5 |

## Getting Started

### Prerequisites

- Python 3.14+ and [uv](https://docs.astral.sh/uv/)
- Basic Python syntax knowledge (variables, functions, classes, loops)

### Setup

```bash
git clone https://github.com/kjkondratuk/LearningML.git
cd LearningML
uv sync --extra dev
```

Install additional dependencies as you progress:

```bash
# When starting Spiral 1C (neural networks):
uv sync --extra spiral1

# When starting Spiral 2:
uv sync --extra spiral2

# When starting Spiral 3:
uv sync --extra spiral3
```

### Working Through a Module

Each module is self-contained. Navigate to it and everything you need is right there:

```bash
cd spiral_1/1a_scientific_python/01_python_idioms
```

**What's inside:**

```
01_python_idioms/
├── README.md          # The lesson: concepts, Go parallels, exercises
├── exercises.py       # Function stubs — this is what you implement
└── tests/
    └── test_exercises.py  # Complete tests — these drive your work
```

**The workflow:**

```bash
# 1. Read the README to understand the concepts
# 2. Run the tests to see them fail
uv run pytest tests/ -v

# 3. Open exercises.py and implement the functions
# 4. Run the tests again to see them pass
uv run pytest tests/ -v

# 5. Move to the next module
```

Some modules also include:
- `mini_project.py` --- a larger exercise that ties the module's concepts together
- `notebook.ipynb` --- interactive exploration companion (when applicable)

### Running All Tests

```bash
# All tests for a phase:
uv run pytest spiral_1/1a_scientific_python/ -v

# All tests for a spiral:
uv run pytest spiral_1/ -v

# Skip slow tests (GPU/training-heavy):
uv run pytest spiral_1/ -v -m "not slow"
```

## Repo Structure

```
LearningML/
├── spiral_1/                     # Foundations & Intuition
│   ├── 1a_scientific_python/     #   Python, NumPy, Pandas, Matplotlib
│   ├── 1b_classical_ml/         #   scikit-learn based ML
│   ├── 1c_neural_networks/      #   NumPy then PyTorch
│   └── 1d_math_refresher/       #   Reference: linear algebra, calculus, probability
├── spiral_2/                     # Depth & Rigor
│   ├── 2a_math_foundations/     #   Dedicated math pass
│   ├── 2b_classical_ml_theory/  #   Derive and implement from scratch
│   ├── 2c_deep_learning_theory/ #   Backprop, Transformers from scratch
│   └── 2d_ml_engineering/       #   Production ML tooling
├── spiral_3/                     # Advanced & Research Frontier
│   ├── 3a_advanced_deep_learning/
│   ├── 3b_nlp_language_models/
│   ├── 3c_advanced_math/
│   ├── 3d_specialization/       #   RL, CV, ML Systems, Bayesian ML
│   └── 3e_paper_reproduction/
├── docs/plans/                   # Curriculum design documents
└── pyproject.toml
```

## Design Principles

- **Build first, understand second.** Every module has you writing code before explaining the theory. The math comes in when you need it.
- **TDD-driven learning.** Tests are complete and ready to run. They tell you exactly what to implement. Red, green, refactor.
- **Self-contained modules.** Each module directory has everything: the lesson, the stubs, the tests. No flipping between files.
- **Go developer friendly.** READMEs draw parallels to Go patterns where they exist, and call out where Python thinks differently.
- **"Do it the wrong way" challenges.** Modules 1A-02 and 1A-03 (and others throughout) deliberately have you implement something the slow/naive way first, then show you the idiomatic approach. The contrast teaches more than any explanation.
- **Math callouts with resources.** Each module lists the math concepts you'll encounter and links to specific resources (3Blue1Brown, textbook chapters) so you can go deeper when you want to.
