# Conversion Prediction Model

## Project Overview

This project implements a machine learning pipeline to predict which free-tier (non-customer) companies are likely to convert to paying customers within the next 30 days. The model is designed to generate a weekly prioritized list of leads for the sales team.

## Objective

**Problem:** Identify high-potential free-tier users who are likely to upgrade to a paid plan.
**Output:** A weekly prioritized list of leads, generated every Sunday, including a propensity score and the top driving feature for each lead.

## Data Description

The model uses the following datasets:
- `customers.csv`: List of paying customers with their conversion dates (`CLOSEDATE`) and monthly recurring revenue (`MRR`).
- `noncustomers.csv`: List of free-tier companies.
- `usage_actions.csv`: Time-series data of user actions (e.g., creating contacts, sending emails) within the platform.
- `company_interactions.csv`: (If applicable) Interactions with the company.

## Methodology

### 1. Feature Engineering (`src/features.py`)
We construct a "point-in-time" feature set to prevent data leakage. For any given "cutoff" date (simulation of a Sunday prediction run), we compute:
- **Rolling Usage Features:** Sums, trends, and active days for key actions (CRM contacts, emails, deals) over 7, 14, 30, and 60-day windows.
- **Recency Features:** Days since last active usage, days since first usage, and usage tenure.
- **Entropy & Diversity:** Measures of how varied a company's usage is across different modules.
- **Company Firmographics:** Industry, employee range, and Alexa rank (log-transformed).

### 2. Backtesting Framework (`src/backtester.py`)
To rigorously evaluate performance, we use a time-based backtesting strategy:
- **Expanding Window:** We simulate weekly prediction cycles over the past 18 months.
- **Leakage Prevention:** The training set for any cutoff includes only data available *before* that date. The target variable is whether a company converts in the *next* 30 days.
- **Cold Start Handling:** New portals with insufficient history are handled via sentinel values or excluded until sufficient data is available.

### 3. Modeling
We employ a **Voting Ensemble (Metamodel)** consisting of:
- **Random Forest:** Captures non-linear interactions and is robust to outliers.
- **LightGBM:** Gradient boosting machine that excels at handling categorical features and missing values.
- **Logistic Regression:** Provides a linear baseline and interpretability.

Feature selection is performed using **Recursive Feature Elimination (RFE)** to identify the top predictive signals.

## Results

The model evaluates leads based on **Precision@K** (how many of the top K leads actually converted) and **Recall@K**.

- **Key Output:** A CSV file (e.g., `reports/sales_call_list_2020-07-27.csv`) containing the ranked list of leads.
- **Interpretability:** Each lead includes a `top_feature_signal` (e.g., "High Deals usage (30d)") derived from SHAP values to help sales reps understand *why* a lead was prioritized.

## Repository Structure

```
.
├── data/                   # Raw data files (not included in repo)
├── reports/                # Generated reports and lead lists
│   ├── company_comparison.html
│   └── usage_comparison.html
├── src/                    # Source code
│   ├── backtester.py       # Backtesting logic and model pipeline
│   ├── data_prep.py        # Data cleaning and preprocessing functions
│   ├── features.py         # Feature engineering class
│   └── main_notebook.ipynb # Main analysis and execution notebook
└── README.md               # This file
```

## Usage

1. **Setup Environment:**
   Ensure you have Python 3.8+ and the required libraries installed.
   ```bash
   pip install pandas numpy scikit-learn lightgbm matplotlib ydata-profiling
   ```

2. **Run the Analysis:**
   Open and run `src/main_notebook.ipynb`. This notebook will:
   - Load and clean the data.
   - Generate features.
   - Run the backtest across multiple historical cutoffs.
   - Print evaluation metrics (ROC-AUC, Precision, Recall).
   - Generate the final sales call list for the most recent cutoff.

## Future Improvements
- **Industry Enrichment:** Integrate third-party data to fill missing industry information.
- **Cohort Analysis:** Validate assumption of no concept drift over longer periods.
- **Deep Learning:** Explore LSTM/Transformers for sequential usage patterns if data volume increases.
