import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def missing_summary(df, name):
    """
    Prints a summary of columns in the DataFrame `df` that contain missing values,
    including the count and percentage of missing entries for each column.
    
    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        name (str): Label or name to use when reporting missing columns.
    
    Output:
        Prints a formatted list of columns with missing values to the console.
    """
    miss = df.isna().sum()
    miss = miss[miss > 0]

    print(f"\n{name} columns with missing values:")
    if miss.empty:
        print("None")
        return

    out = pd.DataFrame({
        "column": miss.index,
        "missing_count": miss.values,
        "missing_pct": (miss.values / len(df) * 100).round(2),
    }).sort_values(["missing_pct", "missing_count"], ascending=False)

    print(out.to_string(index=False))


def clean_industry_column(df, column='INDUSTRY'):
    """
    Cleans and normalizes the industry column with refined regex and standardized naming.
    Does NOT perform One-Hot Encoding to avoid data leakage.
    """
    # 1. Cleaning & Normalization
    # Replace separators with spaces to help regex find word boundaries
    series = (df[column].fillna('UNKNOWN')
              .astype(str)
              .str.upper()
              .str.replace(r'[/_\-&]', ' ', regex=True)
              .str.strip())
    
    # 2. Refined Strategic Mapping
    # Ordered from most specific to most general to avoid misclassification
    mapping = {
        r'.*(SOFTWARE|SAAS|TECH).*': 'TECH',
        r'.*(IT|INFORMATION TECHNOLOGY|COMPUTER|NETWORK).*': 'IT_SERVICES',
        r'.*(FINANCE|BANKING|ACCOUNTING|INSURANCE|INVEST).*': 'FINANCE',
        r'.*(CONSULTING|ADVISORY|STRATEGY).*': 'CONSULTING',
        r'.*(STAFFING|RECRUIT|HR|HUMAN RESOURCE).*': 'HR_STAFFING',
        r'.*(EDUCATION|LEARNING|SCHOOL|UNIVERSITY|ACADEMIC).*': 'EDUCATION',
        r'.*(MARKETING|ADVERTISING|PR|PUBLIC RELATIONS).*': 'MARKETING',
        r'.*(ENERGY|RENEWABLE|UTILITY|OIL|GAS).*': 'ENERGY_UTILITIES',
        r'.*(HEALTH|MEDICAL|PHARMA|HOSPITAL).*': 'HEALTHCARE',
        r'.*(NON PROFIT|CIVIC|RELIGIOUS|GOV).*': 'NON_PROFIT'
    }
    
    # Apply mapping
    for pattern, label in mapping.items():
        series = series.replace(pattern, label, regex=True)
    
    # Return modified dataframe
    df = df.copy()
    df[column] = series
    return df


def plot_column_distribution(df, column, log_scale=False, bins=50, figsize=(8, 4), clip_quantile=None):
    """
    Plot distribution of a dataframe column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    log_scale : bool
        Apply log1p transform (recommended for skewed counts)
    bins : int
    figsize : tuple
    clip_quantile : float or None
        If set (e.g., 0.99), clips extreme tail for better visibility
    """

    if column not in df.columns:
        raise ValueError(f"{column} not in dataframe")

    series = df[column].dropna()

    plt.figure(figsize=figsize)

    # Handle datetime separately
    if np.issubdtype(series.dtype, np.datetime64):
        series.hist(bins=50)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {column}")
        plt.show()
        return

    # Optional clipping for visualization only
    if clip_quantile is not None:
        upper = series.quantile(clip_quantile)
        series = series.clip(upper=upper)

    # Optional log transform
    if log_scale:
        series = np.log1p(series)
        title_suffix = " (log1p)"
    else:
        title_suffix = ""

    plt.hist(series, bins=bins)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}{title_suffix}")
    plt.show()


def cap_outliers(df, columns, method="iqr", iqr_multiplier=1.5, lower_pct=0.00, upper_pct=0.99, return_bounds=False):
    """
    Cap extreme values in specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Columns to cap
    method : str
        "iqr" or "percentile"
    iqr_multiplier : float
        Sensitivity for IQR method
    lower_pct, upper_pct : float
        Used if method="percentile"
    return_bounds : bool
        Return cap thresholds

    Returns
    -------
    capped_df (and optionally bounds dict)
    """

    df = df.copy()
    bounds = {}

    for col in columns:

        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

        elif method == "percentile":
            lower = df[col].quantile(lower_pct)
            upper = df[col].quantile(upper_pct)

        else:
            raise ValueError("method must be 'iqr' or 'percentile'")

        df[col] = df[col].clip(lower=lower, upper=upper)
        bounds[col] = (lower, upper)

    if return_bounds:
        return df, bounds

    return df