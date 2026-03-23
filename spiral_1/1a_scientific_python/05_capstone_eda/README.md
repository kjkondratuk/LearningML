# Module 05: Capstone — Exploratory Data Analysis

> **Time estimate:** 4--6 hours
> **Prerequisites:** Modules 01--04 complete

## The Task

Perform a complete exploratory data analysis on the **Ames Housing dataset**. This is a real dataset with 80 features describing residential homes in Ames, Iowa. It's the standard ML learning dataset because it's rich enough to be interesting but small enough to be manageable.

You'll use everything from the previous modules: NumPy for computation, Pandas for wrangling, Matplotlib/Seaborn for visualization, and all the Python idioms you learned along the way.

## Getting the Data

The Ames Housing dataset is available from multiple sources:

```python
# Option 1: Download from the original source
import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"

# Option 2: Use the version bundled with scikit-learn (Boston — similar but smaller)
# Note: The Boston dataset has ethical issues. Prefer Ames.

# Option 3: Download from Kaggle (requires account)
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

# Option 4: Use OpenML
from sklearn.datasets import fetch_openml
housing = fetch_openml(name="house_prices", as_frame=True)
df = housing.frame
```

Place the data file in this directory or load it programmatically. The exact loading mechanism is up to you — build whatever works and put reusable functions in `utils.py`.

## Questions to Answer

Your analysis must answer these five specific questions with data and visualizations:

### Q1: What is the distribution of sale prices?
- Is it normal, skewed, or multimodal?
- What are the central tendency and spread?
- Are there outliers? How would you define "outlier" for this variable?
- Would a log transform make it more normal?

### Q2: Which features have the strongest linear relationship with sale price?
- Compute correlations between all numeric features and sale price.
- Identify the top 5 most correlated features.
- Are any of these correlations surprising? Are any *expected* correlations weaker than you'd think?

### Q3: How does sale price vary by neighborhood?
- Which neighborhoods have the highest and lowest median prices?
- Which neighborhoods have the most price variability?
- Are there neighborhoods with very few sales (small sample sizes)?

### Q4: What is the relationship between living area and sale price?
- Visualize the relationship with a scatter plot.
- Is the relationship linear? Nonlinear?
- Are there clusters or subgroups visible in the scatter?
- What happens when you color-code by another variable (e.g., overall quality)?

### Q5: How do missing values distribute across features?
- Which features have the most missing data?
- Is the missingness random, or is it structured (e.g., "no garage" encoded as NaN)?
- Which features should you drop vs. impute? Why?

## Required Visualizations

Produce at least these four specific visualizations:

1. **Sale price distribution** — Histogram with KDE overlay. Include a second panel showing the log-transformed distribution for comparison.

2. **Correlation heatmap** — Top 10 features most correlated with sale price. Annotated with correlation values.

3. **Neighborhood price boxplot** — Box plots of sale price grouped by neighborhood, sorted by median price. Rotate x-axis labels for readability.

4. **Living area vs. sale price scatter** — Scatter plot with regression line. Color points by overall quality (or another categorical variable of your choice).

## Feature Recommendation

After your analysis, write a 1-paragraph recommendation (as a comment or markdown cell) answering:

> If you were building a simple linear regression model to predict sale price, which 5--8 features would you include and why? Reference specific findings from your EDA.

This paragraph should demonstrate that you've synthesized your analysis into actionable modeling decisions. Don't just list the top correlations — explain *why* certain features are good choices (or bad choices despite high correlation, e.g., multicollinearity).

## Deliverables

Build your analysis as either:
- A Python script that generates all visualizations and prints answers, OR
- A Jupyter notebook (preferred for interleaving code, output, and narrative)

Either way, extract reusable functions into `utils.py` as you go. Functions that are good candidates for extraction:
- Data loading and initial cleaning
- Plotting helpers (if you customize beyond the Module 04 exercises)
- Summary statistics computation
- Any function you call more than once

## Evaluation Criteria

You're done when:

- [ ] All 5 questions are answered with supporting data
- [ ] All 4 required visualizations are produced and interpretable
- [ ] Your feature recommendation paragraph is written and references your EDA findings
- [ ] `utils.py` contains at least 3 reusable functions extracted from your analysis
- [ ] Code is clean: no copy-pasted blocks, no unexplained magic numbers, functions have docstrings

## Resources

- [Ames Housing Dataset Documentation](http://jse.amstat.org/v19n3/decock.pdf) — Dean De Cock's original paper describing all 80 variables
- [StatQuest: Linear Regression](https://www.youtube.com/watch?v=PaFPbb66DxQ) — Background for your feature recommendation
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html) — Inspiration for visualization styles
