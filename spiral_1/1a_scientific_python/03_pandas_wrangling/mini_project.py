"""Module 03: Mini-Project — Clean Messy Data Pipeline

Implement a full data cleaning pipeline that handles real-world data quality issues.
Run tests with: pytest tests/test_mini_project.py -v
"""

import pandas as pd


def clean_messy_data(df: pd.DataFrame) -> pd.DataFrame:
    """Take a deliberately messy DataFrame and produce a clean, analysis-ready result.

    The input DataFrame has these problems (handle all of them):

    1. Column names have inconsistent casing and extra whitespace
       (e.g., " Price ", "CATEGORY", "  date_Sold").
       -> Normalize to lowercase, stripped, with underscores for spaces.

    2. The "price" column contains string values like "$1,234.56" mixed with
       numeric values and NaNs.
       -> Parse to float, removing "$" and "," characters.

    3. The "date" column contains dates in multiple formats
       (e.g., "2023-01-15", "01/15/2023", "January 15, 2023").
       -> Parse all to datetime.

    4. There are exact duplicate rows.
       -> Remove them.

    5. The "category" column has inconsistent casing ("Electronics", "ELECTRONICS",
       "electronics").
       -> Normalize to lowercase.

    6. The "quantity" column has negative values and extreme outliers
       (values > 1000 are data entry errors).
       -> Clip negatives to 0, clip values > 1000 to NaN, then fill NaN with
          the column median (computed after clipping).

    7. There are still NaN values in numeric columns after the above steps.
       -> Fill remaining numeric NaNs with 0.

    Args:
        df: A messy DataFrame (do not modify in place).

    Returns:
        A clean DataFrame with:
        - Normalized column names
        - No duplicates
        - price as float
        - date as datetime
        - category as lowercase string
        - quantity cleaned and filled
        - No NaN values in numeric columns
        - Reset index
    """
    raise NotImplementedError
