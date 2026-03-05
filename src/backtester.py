"""
backtester.py
─────────────
Core backtesting engine for B2B SaaS conversion propensity.

Class hierarchy
───────────────
    EvaluationMixin     (evaluation.py)     — run_baselines, plot_pr_curves_for_fold
    ExplainabilityMixin (explainability.py) — plot_feature_importance, run_shap_analysis
        └── PropensityBacktester            — __init__, _build_fold_dataset,
                                              _get_model_pipelines, _evaluate_fold,
                                              run_backtest, plot_results

Usage
─────
    from backtester import PropensityBacktester

    backtester = PropensityBacktester(companies_df, customers_df, features_df)
    metrics, leads = backtester.run_backtest()

    backtester.plot_results()
    baseline_df    = backtester.run_baselines()
    backtester.plot_pr_curves_for_fold()
    imp_df         = backtester.plot_feature_importance()
    leads, shap_df = backtester.run_shap_analysis(leads)
"""
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import EvaluationMixin
from explainability import ExplainabilityMixin


class PropensityBacktester(EvaluationMixin, ExplainabilityMixin):
    """
    Production-ready backtester for B2B SaaS conversion propensity models.

    Inherits evaluation and explainability methods from mixins — see
    evaluation.py and explainability.py for those method signatures.
    """

    def __init__(
        self,
        companies_df: pd.DataFrame,
        customers_df: pd.DataFrame,
        features_df: pd.DataFrame,
        prediction_horizon_days: int = 30,
        top_k_leads: int = 10,
        n_features_to_select: Optional[int] = 30,
        company_drop_cols: Optional[List[str]] = None,
        training_positive_window_days: Optional[int] = 90,
    ):
        """
        Parameters
        ----------
        training_positive_window_days : int or None
            If set, only companies that converted within this many days *before*
            the cutoff are used as training positives. This teaches the model
            "what does a company look like in the weeks before it converts"
            rather than "what does a long-tenured customer look like."
            A company that converted 18 months ago has a very different feature
            profile (full usage history, many users, established patterns) vs.
            one that is about to convert. Blending them inflates the apparent
            signal and biases the model toward established heavy users.
            Set to None to fall back to the original behaviour (all historical
            converters are positives), which is useful as an ablation baseline.
            Recommended range: 60–120 days. Default: 90.
        """
        self.prediction_horizon = pd.Timedelta(days=prediction_horizon_days)
        self.training_positive_window = (
            pd.Timedelta(days=training_positive_window_days)
            if training_positive_window_days is not None
            else None
        )
        self.top_k = top_k_leads
        self.n_features_to_select = n_features_to_select
        self.features_df = features_df.copy()

        # Default columns to drop to prevent data leakage.
        # INDUSTRY (raw string) is dropped here; the IND_* one-hot columns
        # derived from it in data_prep.py ARE included in the feature matrix.
        # MRR and CLOSEDATE are dropped to prevent direct label leakage.
        drop_cols = company_drop_cols if company_drop_cols else ["CLOSEDATE", "MRR", "INDUSTRY"]

        self.company_feats   = companies_df.set_index("ID").drop(columns=drop_cols, errors="ignore")
        self.cust_close_map  = customers_df.set_index("ID")["CLOSEDATE"]
        self.all_cutoffs     = sorted(self.features_df["cutoff"].unique())
        self.metrics_results = []
        self.priority_lists  = []

    def _build_fold_dataset(
        self, cutoff: pd.Timestamp
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Generates point-in-time training and testing datasets to prevent data leakage.

        Training positive definition
        ----------------------------
        When `training_positive_window` is set, only companies that converted
        within the window (cutoff - window, cutoff] are used as training positives.

        Rationale: a company that converted 18 months ago has a mature feature
        profile — high usage counts, many users, long tenure — that looks nothing
        like a company that is *about to* convert. Including long-tenured customers
        as positives teaches the model to rank "established heavy users" highly
        rather than "companies showing pre-conversion signals." Restricting to a
        recent window ensures the positive class reflects the pre-conversion
        behavioural state the model needs to recognise at inference time.

        The trade-off is fewer training positives per fold. With this dataset
        (~200 total converters, ~8-17 per 30-day window) the window is set to
        90 days by default to retain ~25-50 positives per fold while still
        capturing the pre-conversion signal pattern. Set
        `training_positive_window_days=None` to restore the original behaviour
        and use it as an ablation baseline.
        """
        # 1. Companies that have already converted at this point in time —
        #    excluded from the test set (they are no longer prospects).
        converted_ids = self.cust_close_map[self.cust_close_map <= cutoff].index
        free_tier_ids = self.company_feats.index.difference(converted_ids)

        # 2. Test labels: did a current free-tier company convert in the next 30 days?
        test_target_dates = self.cust_close_map.reindex(free_tier_ids)
        y_test = (
            (test_target_dates > cutoff) &
            (test_target_dates <= cutoff + self.prediction_horizon)
        ).astype(int)

        # 3. Training positives: converters within the recent window only.
        if self.training_positive_window is not None:
            window_start = cutoff - self.training_positive_window
            train_positive_ids = self.cust_close_map[
                (self.cust_close_map > window_start) &
                (self.cust_close_map <= cutoff)
            ].index
        else:
            # Ablation / original behaviour: all historical converters are positives.
            train_positive_ids = converted_ids

        # 4. Training negatives: free-tier companies not in the test positive window.
        #    Companies about to convert (test positives) are excluded from training
        #    to avoid ambiguous labels.
        test_positives   = y_test[y_test == 1].index
        stable_negatives = free_tier_ids.difference(test_positives)
        train_ids        = train_positive_ids.union(stable_negatives)

        # 5. Build feature snapshots (usage data already pre-filtered to < cutoff
        #    by VectorizedUsageFeatureBacktester).
        usage_snap = self.features_df[self.features_df["cutoff"] == cutoff].copy()
        usage_snap["WHEN_TIMESTAMP"] = pd.to_datetime(usage_snap["WHEN_TIMESTAMP"], errors="coerce")
        usage_snap = usage_snap.dropna(subset=["WHEN_TIMESTAMP"])
        usage_snap = (
            usage_snap
            .sort_values("WHEN_TIMESTAMP")
            .drop_duplicates("ID", keep="last")
            .set_index("ID")
            .drop(columns=["cutoff", "WHEN_TIMESTAMP"], errors="ignore")
        )

        X_all = (
            self.company_feats[~self.company_feats.index.duplicated(keep="first")]
            .join(usage_snap, how="inner")
        )

        # 6. Assemble train / test matrices.
        #    NaNs in X_train dropped (label integrity requires known feature values).
        #    NaNs in X_test left in — SimpleImputer inside each pipeline handles them
        #    so no real prospect is silently excluded from scoring.
        X_train = X_all.loc[X_all.index.intersection(train_ids)].dropna()
        y_train = pd.Series(
            X_train.index.isin(train_positive_ids).astype(int),
            index=X_train.index,
            name="label",
        )

        X_test = X_all.loc[X_all.index.intersection(free_tier_ids)]  # no dropna — see note above
        y_test = y_test.reindex(X_test.index)

        logging.debug(
            f"Fold {cutoff.date()} | "
            f"train_pos={y_train.sum()} (window={self.training_positive_window}) | "
            f"train_neg={len(y_train) - y_train.sum()} | "
            f"test_pos={y_test.sum()} | test_n={len(y_test)}"
        )

        return X_train, y_train, X_test, y_test

    def _get_model_pipelines(self, total_input_features: int) -> Dict[str, Pipeline]:
        """Defines and returns the ML pipelines using RFE for feature selection."""

        # If n_features is None, drop half.
        # Otherwise use the min of the requested number or total features available.
        target_features = (
            self.n_features_to_select
            if self.n_features_to_select is not None
            else total_input_features // 2
        )
        target_features = min(target_features, total_input_features)

        rfe_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        def get_rfe_step():
            # set_output(transform="pandas") keeps feature names intact for LightGBM
            return RFE(
                estimator=rfe_estimator,
                n_features_to_select=target_features,
            ).set_output(transform="pandas")

        rf_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("rfe",     get_rfe_step()),
            ("clf",     RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ])

        lr_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("scaler",  StandardScaler().set_output(transform="pandas")),
            ("rfe",     get_rfe_step()),
            ("clf",     LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            )),
        ])

        lgbm_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("rfe",     get_rfe_step()),
            ("clf",     LGBMClassifier(
                n_estimators=200, learning_rate=0.05, is_unbalance=True,
                random_state=42, verbosity=-1,
            )),
        ])

        committee_pipe = VotingClassifier(
            estimators=[("rf", rf_pipe), ("lr", lr_pipe), ("lgbm", lgbm_pipe)],
            voting="soft",
        )

        return {
            "RandomForest": rf_pipe,
            "LogisticReg":  lr_pipe,
            "LightGBM":     lgbm_pipe,
            "Metamodel":    committee_pipe,
        }

    def _evaluate_fold(
        self,
        cutoff: pd.Timestamp,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """Trains all models on one fold, computes metrics, and builds the lead list."""
        valid_cols  = X_train.columns[X_train.notna().any()].tolist()
        X_tr, X_te  = X_train[valid_cols], X_test[valid_cols]
        models      = self._get_model_pipelines(total_input_features=len(valid_cols))
        fold_metrics = {"cutoff": cutoff, "n_test_pos": int(y_test.sum()), "metrics": []}
        lead_list   = pd.DataFrame()

        for name, pipeline in models.items():
            pipeline.fit(X_tr, y_train)
            y_prob = pipeline.predict_proba(X_te)[:, 1]

            y_test_arr = np.array(y_test)
            top_k_idx  = np.argsort(y_prob)[::-1][:self.top_k]
            hits       = y_test_arr[top_k_idx].sum()
            total_pos  = y_test_arr.sum()

            fold_metrics["metrics"].append({
                "model":    name,
                "roc":      roc_auc_score(y_test, y_prob),
                "pr":       average_precision_score(y_test, y_prob),
                "p_at_k":   hits / self.top_k,
                "rec_at_k": hits / total_pos if total_pos > 0 else 0.0,
            })

            if name == "Metamodel":
                lead_list = pd.DataFrame({
                    "cutoff":           cutoff,
                    "company_id":       X_te.index,
                    "propensity_score": y_prob,
                    "converted_actual": y_test.values,
                }).sort_values("propensity_score", ascending=False)
                lead_list["priority_rank"] = range(1, len(lead_list) + 1)

        return fold_metrics, lead_list

    def run_backtest(self) -> Tuple[List[dict], pd.DataFrame]:
        """
        Executes the backtest across all cutoffs.

        Returns
        -------
        metrics_summary     : list of dicts    (one per fold)
        lead_priority_table : pd.DataFrame     (ranked leads across all folds)
        """
        self.metrics_results = []
        all_leads = []

        window_label = (
            f"{int(self.training_positive_window.days)}d"
            if self.training_positive_window is not None
            else "all-time"
        )
        print(f"Training positive window : {window_label}")
        print(f"Prediction horizon       : {int(self.prediction_horizon.days)}d")
        print(f"Top-K leads              : {self.top_k}")
        print()

        header = (
            f"{'Cutoff':<15} | {'Model':<14} | {'ROC':>5} | "
            f"{'PR':>5} | {f'P@{self.top_k}':>5} | {f'Rec@{self.top_k}':>6}"
        )
        print(header)
        print("-" * len(header))

        for cutoff in self.all_cutoffs:
            try:
                cutoff_ts               = pd.Timestamp(cutoff)
                X_tr, y_tr, X_te, y_te = self._build_fold_dataset(cutoff_ts)

                if y_te.sum() < 3 or y_tr.sum() < 5:
                    continue

                metrics, leads = self._evaluate_fold(cutoff_ts, X_tr, y_tr, X_te, y_te)
                self.metrics_results.append(metrics)
                all_leads.append(leads)

                print(f"{str(metrics['cutoff'].date()):<15} ({metrics['n_test_pos']} pos)")
                for m in metrics["metrics"]:
                    print(
                        f"{' ':>15} | {m['model']:<14} | {m['roc']:>5.2f} | "
                        f"{m['pr']:>5.2f} | {m['p_at_k']:>5.2f} | {m['rec_at_k']:>6.2f}"
                    )
                print("-" * len(header))

            except Exception as e:
                logging.error(f"Failed at cutoff {cutoff}: {e}")

        final_lead_table = pd.concat(all_leads) if all_leads else pd.DataFrame()
        return self.metrics_results, final_lead_table

    def plot_results(self) -> None:
        """Visualises PR AUC, Precision@K and Recall@K across all backtest folds."""
        if not self.metrics_results:
            logging.warning("No results to plot. Run run_backtest() first.")
            return

        vis_data = [
            {
                "cutoff": fold["cutoff"], "model": m["model"],
                "roc": m["roc"], "pr": m["pr"],
                "p_at_k": m["p_at_k"], "rec_at_k": m["rec_at_k"],
            }
            for fold in self.metrics_results
            for m in fold["metrics"]
        ]

        df_plot     = pd.DataFrame(vis_data)
        model_names = ["RandomForest", "LightGBM", "LogisticReg", "Metamodel"]
        colors      = {
            "RandomForest": "#7f8c8d",
            "LogisticReg":  "#3498db",
            "LightGBM":     "#e67e22",
            "Metamodel":    "#2ecc71",
        }

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(
            f"Backtest Analysis: Top {self.top_k} Leads "
            f"(RFE Selection: {self.n_features_to_select})",
            fontsize=16, fontweight="bold",
        )

        for ax, metric, title in zip(
            axes,
            ["pr", "p_at_k", "rec_at_k"],
            ["PR AUC", f"Precision @ {self.top_k}", f"Recall @ {self.top_k}"],
        ):
            for model in model_names:
                data = df_plot[df_plot["model"] == model]
                ax.plot(
                    data["cutoff"], data[metric],
                    marker="o", label=model, color=colors[model],
                    linewidth=2 if model == "Metamodel" else 1.5,
                )
            ax.set_title(title)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()