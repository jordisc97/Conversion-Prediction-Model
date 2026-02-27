import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class PropensityBacktester:
    """
    A production-ready class to backtest ML models for B2B SaaS propensity.
    """
    
    def __init__(
        self, 
        companies_df: pd.DataFrame, 
        customers_df: pd.DataFrame, 
        features_df: pd.DataFrame,
        prediction_horizon_days: int = 30,
        top_k_leads: int = 10,
        n_features_to_select: Optional[int] = 30,
        company_drop_cols: Optional[List[str]] = None
    ):
        """
        Initializes the backtester with datasets and configuration.
        """
        self.prediction_horizon = pd.Timedelta(days=prediction_horizon_days)
        self.top_k = top_k_leads
        self.n_features_to_select = n_features_to_select
        self.features_df = features_df.copy()
        
        # Default columns to drop to prevent data leakage
        drop_cols = company_drop_cols if company_drop_cols else ["CLOSEDATE", "MRR", "INDUSTRY"]
        
        # Pre-process static lookup tables
        self.company_feats = companies_df.set_index("ID").drop(columns=drop_cols, errors="ignore")
        self.cust_close_map = customers_df.set_index("ID")["CLOSEDATE"]
        
        # Extract unique cutoffs for the backtest loop
        self.all_cutoffs = sorted(self.features_df["cutoff"].unique())
        self.metrics_results = []
        self.priority_lists = []

    def _build_fold_dataset(self, cutoff: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Generates point-in-time training and testing datasets to prevent data leakage.
        """
        # 1. Identify user segments at this point in time
        converted_ids = self.cust_close_map[self.cust_close_map <= cutoff].index
        free_tier_ids = self.company_feats.index.difference(converted_ids)

        # 2. Labeling: Did they convert in the next X days?
        test_target_dates = self.cust_close_map.reindex(free_tier_ids)
        y_test = ((test_target_dates > cutoff) & 
                  (test_target_dates <= cutoff + self.prediction_horizon)).astype(int)

        # 3. Time-travel Usage snapshots
        usage_snap = self.features_df[self.features_df["cutoff"] == cutoff].copy()
        usage_snap["WHEN_TIMESTAMP"] = pd.to_datetime(usage_snap["WHEN_TIMESTAMP"], errors='coerce')
        usage_snap = usage_snap.dropna(subset=["WHEN_TIMESTAMP"])
        
        usage_snap = usage_snap.sort_values("WHEN_TIMESTAMP").drop_duplicates("ID", keep="last")
        usage_snap = usage_snap.set_index("ID").drop(columns=["cutoff", "WHEN_TIMESTAMP"], errors="ignore")

        # Join Static + Usage data
        X_all = self.company_feats[~self.company_feats.index.duplicated(keep='first')].join(usage_snap, how="inner")

        # 4. Train/Test Split
        test_positives = y_test[y_test == 1].index
        stable_negatives = free_tier_ids.difference(test_positives)
        train_ids = converted_ids.union(stable_negatives)
        
        X_train = X_all.loc[X_all.index.intersection(train_ids)].dropna()
        y_train = X_train.index.isin(converted_ids).astype(int)

        X_test = X_all.loc[X_all.index.intersection(free_tier_ids)].dropna()
        y_test = y_test.reindex(X_test.index)

        return X_train, y_train, X_test, y_test

    def _get_model_pipelines(self, total_input_features: int) -> Dict[str, Pipeline]:
        """Defines and returns the ML pipelines using RFE for feature selection."""
        
        # Logic: If n_features is None, drop half. 
        # Otherwise, use the min of the requested number or the total features available.
        target_features = (
            self.n_features_to_select 
            if self.n_features_to_select is not None 
            else total_input_features // 2
        )
        target_features = min(target_features, total_input_features)

        # Shared estimator for RFE
        rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        def get_rfe_step():
            # .set_output(transform="pandas") ensures feature names stay intact for LightGBM
            return RFE(estimator=rfe_estimator, n_features_to_select=target_features).set_output(transform="pandas")
        
        rf_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy="median").set_output(transform="pandas")),
            ('rfe', get_rfe_step()),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1))
        ])
        
        lr_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy="median").set_output(transform="pandas")),
            ('scaler', StandardScaler().set_output(transform="pandas")),
            ('rfe', get_rfe_step()),
            ('clf', LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ])
        
        lgbm_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy="median").set_output(transform="pandas")),
            ('rfe', get_rfe_step()),
            ('clf', LGBMClassifier(n_estimators=200, learning_rate=0.05, is_unbalance=True, random_state=42, verbosity=-1))
        ])

        committee_pipe = VotingClassifier(
            estimators=[('rf', rf_pipe), ('lr', lr_pipe), ('lgbm', lgbm_pipe)],
            voting='soft'
        )

        return {
            "RandomForest": rf_pipe,
            "LogisticReg":  lr_pipe,
            "LightGBM":     lgbm_pipe,
            "Metamodel":    committee_pipe
        }

    def _evaluate_fold(self, cutoff: pd.Timestamp, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Trains models, calculates metrics, and generates lead lists for a single fold."""
        valid_cols = X_train.columns[X_train.notna().any()].tolist()
        X_tr, X_te = X_train[valid_cols], X_test[valid_cols]

        models = self._get_model_pipelines(total_input_features=len(valid_cols))
        fold_metrics = {"cutoff": cutoff, "n_test_pos": int(y_test.sum()), "metrics": []}
        
        # We will use the Metamodel (Ensemble) to generate the priority lead list
        lead_list = pd.DataFrame()

        for name, pipeline in models.items():
            pipeline.fit(X_tr, y_train)
            y_prob = pipeline.predict_proba(X_te)[:, 1]

            # Track metrics
            y_test_arr = np.array(y_test)
            top_k_idx = np.argsort(y_prob)[::-1][:self.top_k]
            hits = y_test_arr[top_k_idx].sum()
            total_pos = y_test_arr.sum()

            fold_metrics["metrics"].append({
                "model": name,
                "roc": roc_auc_score(y_test, y_prob),
                "pr":  average_precision_score(y_test, y_prob),
                "p_at_k": hits / self.top_k,
                "rec_at_k": hits / total_pos if total_pos > 0 else 0.0
            })

            # If this is the Metamodel, capture the full lead list for output
            if name == "Metamodel":
                lead_list = pd.DataFrame({
                    "cutoff": cutoff,
                    "company_id": X_te.index,
                    "propensity_score": y_prob,
                    "converted_actual": y_test.values
                }).sort_values("propensity_score", ascending=False)
                
                lead_list["priority_rank"] = range(1, len(lead_list) + 1)

        return fold_metrics, lead_list

    def run_backtest(self) -> Tuple[List[dict], pd.DataFrame]:
        """Executes backtest across cutoffs. Returns (metrics_summary, lead_priority_table)."""
        self.metrics_results = []
        all_leads = []
        
        header = f"{'Cutoff':<15} | {'Model':<14} | {'ROC':>5} | {'PR':>5} | {f'P@{self.top_k}':>5} | {f'Rec@{self.top_k}':>6}"
        print(header)
        print("-" * len(header))

        for cutoff in self.all_cutoffs:
            try:
                cutoff_ts = pd.Timestamp(cutoff)
                X_tr, y_tr, X_te, y_te = self._build_fold_dataset(cutoff_ts)

                if y_te.sum() < 3 or y_tr.sum() < 5:
                    continue

                metrics, leads = self._evaluate_fold(cutoff_ts, X_tr, y_tr, X_te, y_te)
                self.metrics_results.append(metrics)
                all_leads.append(leads)
                
                print(f"{str(metrics['cutoff'].date()):<15} ({metrics['n_test_pos']} pos)")
                for m in metrics['metrics']:
                    print(f"{' ':>15} | {m['model']:<14} | {m['roc']:>5.2f} | {m['pr']:>5.2f} | {m['p_at_k']:>5.2f} | {m['rec_at_k']:>6.2f}")
                print("-" * len(header))
                
            except Exception as e:
                logging.error(f"Failed at cutoff {cutoff}: {str(e)}")
                
        final_lead_table = pd.concat(all_leads) if all_leads else pd.DataFrame()
        return self.metrics_results, final_lead_table
        
    def plot_results(self):
        """Visualizes the stored backtest results."""
        if not self.metrics_results:
            logging.warning("No results to plot. Run run_backtest() first.")
            return
            
        vis_data = []
        for fold in self.metrics_results:
            for m in fold['metrics']:
                vis_data.append({
                    "cutoff": fold['cutoff'], "model": m['model'],
                    "roc": m['roc'], "pr": m['pr'], "p_at_k": m['p_at_k'], "rec_at_k": m['rec_at_k']
                })

        df_plot = pd.DataFrame(vis_data)
        model_names = ["RandomForest", "LightGBM", "LogisticReg", "Metamodel"]
        colors = {"RandomForest": "#7f8c8d", "LogisticReg": "#3498db", "LightGBM": "#e67e22", "Metamodel": "#2ecc71"}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f"Backtest Analysis: Top {self.top_k} Leads (RFE Selection: {self.n_features_to_select})", fontsize=16, fontweight="bold")

        for i, metric, title in zip(range(3), 
                                    ["roc", "p_at_k", "rec_at_k"], 
                                    ["ROC-AUC", f"Precision @ {self.top_k}", f"Recall @ {self.top_k}"]):
            for model in model_names:
                data = df_plot[df_plot["model"] == model]
                axes[i].plot(data["cutoff"], data[metric], marker='o', label=model, 
                             color=colors[model], linewidth=2 if model=="Metamodel" else 1.5)
            axes[i].set_title(title)
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, linestyle='--')
            axes[i].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()