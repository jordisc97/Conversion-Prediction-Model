# Conversion Prediction Model

## Project Overview

This project implements a production-ready ML pipeline to predict which free-tier (non-customer) companies are likely to convert to paying customers within the next 30 days. The model generates a weekly prioritised list of leads for Sales & CS teams, enriched with SHAP-derived explanations and GPT-4o-generated action briefs.

## Objective

**Problem:** Identify high-potential free-tier portals likely to upgrade to a paid plan within 30 days.  
**Output:** A weekly Top-10 ranked lead list (generated every Sunday) with propensity scores, top SHAP signals, and a rep-facing sales brief per lead.

---

## Key Results

| Metric | Metamodel Average | Peak (Jun-20) |
|---|---|---|
| ROC-AUC | 0.81 | 0.90 |
| PR-AUC | 0.18 | 0.37 |
| Precision@10 | 0.23 | 0.40 |
| Recall@10 | 0.21 | 0.38 |

**Baseline prevalence is ~1.5%** (8–17 conversions per ~1,000 free portals per month). A random list would yield ~1–2 hits per 10 leads. The Metamodel averages ~2–3 hits, with peaks of 4 — a meaningful signal given the noise floor.

> **Important caveat:** With 8–17 test positives per fold, point estimates carry wide confidence intervals. Bootstrap CIs should be reported before presenting results to stakeholders as definitive.

---

## Critical Design Decisions

### 1. Training Positive Window (Most Important)
Training positives are restricted to companies that converted **within 90 days before each cutoff**, not all historical converters. A company that converted 18 months ago has a mature feature profile (high usage, many users, long tenure) that looks nothing like a company *about to* convert. Blending them teaches the model to rank established heavy users, not pre-conversion signals. Set `training_positive_window_days=None` to ablate this and confirm the effect.

### 2. Point-in-Time Feature Construction
All features are computed using only data strictly before `cutoff`. Rolling windows use `WHEN_TIMESTAMP < cutoff` as the filter. Recency dates, entropy, and diversity are all derived from the pre-cutoff slice. This prevents any form of future data contamination across the 6 backtest folds.

### 3. Test Positives Excluded from Training
Companies that will convert in the next 30 days (test positives) are removed from the training set entirely. Including them as negatives would introduce ambiguous labels and understate true model performance.

---

## Methodology

### Feature Engineering (`src/features.py`)
Point-in-time feature panel with a MultiIndex of `(company_id, cutoff)`:

- **Rolling usage** (7/14/30/60d): action sums, user sums, active days, active ratio, session intensity (actions per active day), per-module sums and share percentages
- **Trend**: OLS slope of daily actions per window — captures acceleration vs. decay
- **Acceleration ratio**: `actions_sum_30d / (actions_sum_60d + 1)` — the single strongest non-linear signal
- **Recency**: days since last/first usage, usage tenure, recency score (1 / days + 1)
- **Module entropy & diversity**: Shannon entropy across modules, count of modules used — captures breadth of platform adoption
- **Firmographics**: log-transformed Alexa rank, employee range, normalised industry category

### Backtesting Framework (`src/backtester.py`)
- **6 monthly folds** from Feb–Jul 2020, each simulating a Sunday production run
- **Expanding training window**: all data before cutoff; test window is the following 30 days
- **Leakage-safe pipeline**: imputer + scaler + RFE + model are fit exclusively on training data per fold; test data passes through the fitted pipeline only
- **COVID robustness test**: April 2020 fold acts as an involuntary distribution-shift stress test — tree models collapse (RF: 0.53, LGBM: 0.46 ROC), Metamodel holds at 0.79

### Ensemble (`Metamodel`)
Soft-voting ensemble of three independently pipelined models:

- **Random Forest** (200 trees, balanced class weights, RFE→30 features)
- **LightGBM** (200 trees, lr=0.05, is_unbalance=True, RFE→30 features)
- **Logistic Regression** (L2, balanced, StandardScaled, RFE→30 features)

RFE runs independently per fold — correct for leakage but means feature selection is not stable across folds. Future work: report fold-level feature selection frequency.

### Explainability (`src/explainability.py`)
SHAP is computed on **every fold's test set** (not just the diagnostic fold), so every company in every weekly export receives three human-readable conversion drivers:

```
signal_1: High CRM actions (30d) (SHAP: +0.341)
signal_2: Accelerating usage growth (SHAP: +0.187)
signal_3: Broad multi-module adoption (SHAP: +0.112)
```

### LLM Sales Briefs (`src/llm_intelligence.py`)
`SalesIntelligenceAgent` parses SHAP signal strings into structured dicts and injects them alongside company metadata (industry, size, recency) into a GPT-4o prompt. Output is a 2-sentence action brief with a product tier recommendation. Runs only on Top-K leads to contain token cost. Includes a fallback string on API failure so the pipeline never blocks.

---

## Known Limitations

1. **No confidence intervals reported.** All metrics are point estimates over very small positive counts (8–17 per fold). This is the highest-priority fix before stakeholder presentation.
2. **Baselines not in main output.** `run_baselines()` computes random and activity-heuristic comparisons but these are not shown alongside model metrics by default. Add them.
3. **Count-based metric only.** Precision@10 treats all conversions equally. An MRR-weighted P@K would better reflect business value.
4. **Scores are not calibrated probabilities.** Current outputs are ranking scores. Platt scaling or isotonic regression would convert them to true probabilities.
5. **No causal identification.** The model finds correlation with conversion; whether outreach *causes* conversion is unknown without an A/B test.
6. **No distribution-shift detection.** The April 2020 degradation was identified post-hoc. A PSI (Population Stability Index) check between training and test feature distributions should run automatically each fold.

---

## Immediate Next Steps

1. Add bootstrap CIs to all P@K / Rec@K estimates
2. Add baseline comparison (random + activity heuristic) to the main results print
3. Implement MRR-weighted Precision@K as the primary business metric
4. Calibrate propensity scores (Platt scaling)
5. A/B test model-ranked list vs. rep-selected list to measure incremental lift
6. Automate Sunday refresh via Airflow: feature build → score → SHAP → LLM brief → CRM push

---

## Repository Structure

```
.
├── data/                   # Raw data files (not included in repo)
├── reports/                # Generated reports and lead lists
│   ├── company_comparison.html
│   └── usage_comparison.html
├── src/                    # Source code
│   ├── backtester.py       # Core backtesting engine + ensemble pipelines
│   ├── data_prep.py        # Data cleaning, industry normalisation, missing value audit
│   ├── evaluation.py       # EvaluationMixin: baselines + PR curves
│   ├── explainability.py   # ExplainabilityMixin: SHAP beeswarm, waterfall, signal enrichment
│   ├── features.py         # VectorizedUsageFeatureBacktester: point-in-time feature panel
│   ├── llm_intelligence.py # SalesIntelligenceAgent: GPT-4o action briefs
│   └── main_notebook.ipynb # Main analysis and execution notebook
└── README.md               # This file
```

---

## Usage

1. **Setup Environment:**
   ```bash
   pip install pandas numpy scikit-learn lightgbm matplotlib shap openai ydata-profiling
   ```

2. **Run the Analysis:**
   Open `src/main_notebook.ipynb` and run all cells. The notebook will:
   - Load and clean raw data
   - Build the point-in-time feature panel across 6 cutoffs
   - Run the backtest, printing fold-level metrics to stdout
   - Generate SHAP signals and LLM briefs for the Top-10 leads
   - Export `reports/sales_call_list_<date>.csv`

3. **Key parameters in `PropensityBacktester`:**
   - `training_positive_window_days=90` — set to `None` to ablate the recency window
   - `prediction_horizon_days=30` — forward window for conversion label
   - `top_k_leads=10` — size of the weekly lead list
   - `n_features_to_select=30` — RFE target per fold
