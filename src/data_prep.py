import pandas as pd

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
