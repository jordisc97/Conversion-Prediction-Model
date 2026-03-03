# Data Anomalies and Handling Report

## Summary of Findings

### 1. `customers.csv`
- **Missing Values**:
  - `EMPLOYEE_RANGE`: 2 missing values.
  - `INDUSTRY`: 129 missing values (64.5%).
- **Anomalies**:
  - **Negative MRR**: 4 instances of negative Monthly Recurring Revenue (e.g., -61.15).
  - **Inconsistent Industry Names**: Mixed case and variations (e.g., 'COMPUTER_SOFTWARE' vs 'Technology - Software').
  - **Date Format**: `CLOSEDATE` was in string format.

### 2. `noncustomers.csv`
- **Missing Values**:
  - `ALEXA_RANK`: 114 missing values.
  - `EMPLOYEE_RANGE`: 532 missing values.
  - `INDUSTRY`: 3725 missing values (74.5%).
- **Anomalies**:
  - **Duplicate IDs**: 3 IDs appeared multiple times. In these cases, one row had valid data while the other had `NaN` values.
  - **Inconsistent Industry Names**: Similar to customers.

### 3. `usage_actions.csv`
- **Missing Values**: None.
- **Anomalies**:
  - **Outliers**: Several columns have extreme values (e.g., `ACTIONS_CRM_CONTACTS` up to 27,936, while 99th percentile is ~1320).
  - **Date Format**: `WHEN_TIMESTAMP` was in string format.

## Handling Strategies Implemented

### 1. Customers Data
- **Missing Values**:
  - `EMPLOYEE_RANGE` filled with 'Unknown'.
  - `INDUSTRY` filled with 'Unknown'.
- **Negative MRR**: Replaced with the median of non-negative MRR values.
- **Industry Standardization**: Mapped common variations to standard categories (e.g., 'COMPUTER_SOFTWARE' -> 'Technology - Software').
- **Date Conversion**: Converted `CLOSEDATE` to datetime objects.
- **Alexa Rank**: Filled missing values with `max_rank + 1`.

### 2. Non-Customers Data
- **Duplicate IDs**: Removed duplicates, prioritizing rows with non-null values.
- **Missing Values**:
  - `ALEXA_RANK` filled with 16,000,001 (indicating low rank).
  - `EMPLOYEE_RANGE` and `INDUSTRY` filled with 'Unknown'.
- **Industry Standardization**: Applied same mapping as for customers.

### 3. Usage Actions Data
- **Date Conversion**: Converted `WHEN_TIMESTAMP` to datetime objects.
- **Outliers**: Identified but preserved. For modeling, robust scaling or capping at the 99th percentile may be considered.

## Cleaned Files
The cleaned datasets are saved in `data/processed/`:
- `cleaned_customers.csv`
- `cleaned_noncustomers.csv`
- `cleaned_usage_actions.csv`
