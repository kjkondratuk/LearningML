# Phase 1A: Python for Scientific Computing

> **Goal:** Build fluency with Python's scientific stack so that ML code feels natural, not foreign.

You already know how to write production software in Go. This phase isn't about learning programming---it's about learning *Python's* way of doing things, especially the idioms that show up constantly in ML codebases (generators, decorators, broadcasting, vectorized ops, DataFrames).

## Modules

| # | Module | Key Skills | Estimated Time |
|---|--------|-----------|----------------|
| 01 | [Python Idioms](01_python_idioms/) | Generators, decorators, context managers, dataclasses, comprehensions | 3--4 hours |
| 02 | [NumPy Fundamentals](02_numpy_fundamentals/) | Array ops, broadcasting, vectorization, linear algebra basics | 4--6 hours |
| 03 | [Pandas Wrangling](03_pandas_wrangling/) | DataFrames, cleaning, groupby, merges, method chaining | 4--6 hours |
| 04 | [Visualization](04_visualization/) | Matplotlib, distribution plots, scatter/regression, heatmaps | 2--3 hours |
| 05 | [Capstone EDA](05_capstone_eda/) | End-to-end exploratory data analysis on real housing data | 4--6 hours |

## How to Work Through This

1. **Read the module README** to get the concepts and mental models.
2. **Run the tests first** (`pytest tests/`). They all fail. That's the point.
3. **Implement the stubs** in `exercises.py` until the tests go green.
4. **Do the mini-project** (modules 02--03) which combines what you just built.
5. **Do the "wrong way" challenge** (modules 02--03) to viscerally feel why vectorization matters.

## Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy pytest
```

Or if you're using the project's pyproject.toml:

```bash
pip install -e ".[dev]"
```

## Go Developer Mental Model

Throughout these modules, you'll see callouts like this:

> **Go parallel:** Python generators are like Go channels with a single sender---lazy sequences that yield values on demand.

These aren't perfect analogies. They're *bridges* to help you build intuition faster. Once you have the Python mental model, you won't need the Go comparison anymore.

## Math Refresher Links

Several modules touch on math you may not have used since college. We link to specific resources inline, but here are the heavy hitters:

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) --- Watch episodes 1--4 before Module 02.
- [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) --- Useful background, but not required until Phase 1B.
- [StatQuest: Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) --- Great for Modules 03--05.

## Running Tests

From this directory:

```bash
# Run all tests for all modules
pytest

# Run tests for a specific module
pytest 01_python_idioms/tests/

# Run with verbose output
pytest -v 02_numpy_fundamentals/tests/
```

## Completion Criteria

You're done with Phase 1A when:

- [ ] All `pytest` tests pass across all modules
- [ ] You've completed the mini-projects in modules 02 and 03
- [ ] You've done the "wrong way" challenges and can explain *why* vectorization wins
- [ ] Your capstone EDA notebook answers all 5 questions with visualizations
- [ ] You've populated `05_capstone_eda/utils.py` with reusable functions extracted from your work
