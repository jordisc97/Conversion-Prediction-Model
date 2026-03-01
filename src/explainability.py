"""
explainability.py
─────────────────
ExplainabilityMixin for PropensityBacktester.

Provides model interpretability and sales-output enrichment:
  - plot_feature_importance: LightGBM gain-based importance bar chart
  - run_shap_analysis      : SHAP beeswarm + waterfall plots, plus
                             signal_1/signal_2/signal_3 columns on the lead table

These methods assume run_backtest() has already been called and that
self.metrics_results, self._build_fold_dataset(), and
self._get_model_pipelines() are available via the host class.
"""
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import Patch


class ExplainabilityMixin:
    """Mixin that adds feature importance and SHAP analysis to PropensityBacktester."""

    def plot_feature_importance(self, n_top: int = 20) -> pd.DataFrame:
        """
        Trains LightGBM on the fold with the most test positives and plots the
        top-N feature importances (gain) after RFE selection.

        Call after run_backtest(). Returns the importance DataFrame so callers
        can inspect or export the values programmatically.
        """
        if not self.metrics_results:
            logging.warning("No results found. Run run_backtest() first.")
            return pd.DataFrame()

        best_fold = max(self.metrics_results, key=lambda f: f["n_test_pos"])
        cutoff_ts = pd.Timestamp(best_fold["cutoff"])
        print(f"Extracting feature importances from fold: {cutoff_ts.date()}")

        X_tr, y_tr, _, _ = self._build_fold_dataset(cutoff_ts)
        valid_cols        = X_tr.columns[X_tr.notna().any()].tolist()
        X_tr              = X_tr[valid_cols]

        lgbm_pipe = self._get_model_pipelines(total_input_features=len(valid_cols))["LightGBM"]
        lgbm_pipe.fit(X_tr, y_tr)

        selected_features = lgbm_pipe.named_steps["rfe"].get_feature_names_out()
        importances       = lgbm_pipe.named_steps["clf"].feature_importances_

        imp_df = (
            pd.DataFrame({"feature": selected_features, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(n_top)
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        colors_bar = [
            "#2ecc71" if "users"   in f else
            "#3498db" if "actions" in f else
            "#e67e22"
            for f in imp_df["feature"]
        ]
        ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color=colors_bar[::-1])
        ax.set_xlabel("LightGBM Feature Importance (gain)", fontsize=12)
        ax.set_title(f"Top {n_top} Features — LightGBM (Cutoff: {cutoff_ts.date()})", fontsize=13)
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(handles=[
            Patch(facecolor="#2ecc71", label="User features"),
            Patch(facecolor="#3498db", label="Action features"),
            Patch(facecolor="#e67e22", label="Other features"),
        ], loc="lower right")
        plt.tight_layout()
        plt.show()

        print("\nTop 10 features:")
        print(imp_df.head(10).to_string(index=False))
        return imp_df

    def run_shap_analysis(
        self, lead_table: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        SHAP analysis in two parts.

        PART A — Diagnostic plots on the fold with the most test positives:
            1. Beeswarm summary plot  (global: which features matter and in which direction)
            2. Waterfall plot         (local: why the #1 ranked lead scored so high)

        PART B — Signal enrichment across ALL folds:
            3. Runs SHAP on every fold and populates signal_1 / signal_2 / signal_3
               columns on the lead table so every scored company has a human-readable
               explanation ready for the weekly sales export.

        Parameters
        ----------
        lead_table : pd.DataFrame
            Output of run_backtest().

        Returns
        -------
        lead_table_enriched : pd.DataFrame
            Lead table with signal_1, signal_2, signal_3 columns added.
        shap_df_best : pd.DataFrame
            SHAP value matrix for the best (diagnostic) fold.
        """

        # ── feature label helpers ─────────────────────────────────────────────
        STATIC_LABELS = {
            "module_entropy":         "Broad multi-module adoption",
            "ALEXA_RANK_LOG":         "Strong web presence (low Alexa rank)",
            "days_since_first_usage": "Long time on free tier",
            "usage_tenure_days":      "Long active usage tenure",
            "EMPLOYEE_RANGE":         "Company size signal",
            "recency_score":          "Recent platform activity",
            "module_diversity":       "Uses multiple CRM modules",
            "actions_accel":          "Accelerating usage growth",
            "days_since_last_usage":  "Recently active",
        }

        def _label(feat: str) -> str:
            if feat in STATIC_LABELS:
                return STATIC_LABELS[feat]
            for w in ["7d", "14d", "30d", "60d"]:
                module = (
                    feat
                    .replace(f"_sum_{w}", "").replace(f"_trend_{w}", "")
                    .replace(f"active_days_{w}", "active")
                    .replace(f"actions_per_active_day_{w}", "intensity")
                    .replace("actions_crm_", "").replace("actions_", "")
                    .replace("users_crm_", "").replace("users_", "")
                    .upper()
                )
                if f"_sum_{w}" in feat:
                    kind = "user" if feat.startswith("users") else "action"
                    return f"High {module} {kind}s ({w})"
                if f"_trend_{w}" in feat:
                    return f"Accelerating {module} trend ({w})"
                if f"active_days_{w}" in feat:
                    return f"Frequent logins ({w})"
                if f"actions_per_active_day_{w}" in feat:
                    return f"High session intensity ({w})"
            return feat.replace("_", " ").title()

        def _top_signal(shap_row: pd.Series, top_n: int = 3) -> list:
            """Return top N positive SHAP contributors as a list (one string per column)."""
            pos     = shap_row[shap_row > 0].sort_values(ascending=False)
            reasons = [
                f"{_label(feat)} (SHAP: {val:+.3f})"
                for feat, val in pos.head(top_n).items()
            ]
            # Pad so the list always has exactly top_n entries for consistent column assignment
            while len(reasons) < top_n:
                reasons.append("")
            return reasons

        def _fit_and_shap(cutoff_ts: pd.Timestamp):
            """Fit LightGBM on a fold; return (shap_df, X_te_rfe, lgbm_clf, explainer, sv)."""
            X_tr, y_tr, X_te, _ = self._build_fold_dataset(cutoff_ts)
            valid_cols           = X_tr.columns[X_tr.notna().any()].tolist()
            X_tr_v, X_te_v       = X_tr[valid_cols], X_te[valid_cols]

            lgbm_pipe = self._get_model_pipelines(
                total_input_features=len(valid_cols)
            )["LightGBM"]
            lgbm_pipe.fit(X_tr_v, y_tr)

            imputer  = lgbm_pipe.named_steps["imputer"]
            rfe_step = lgbm_pipe.named_steps["rfe"]
            lgbm_clf = lgbm_pipe.named_steps["clf"]

            X_te_rfe    = rfe_step.transform(imputer.transform(X_te_v))
            explainer   = shap.TreeExplainer(lgbm_clf)
            shap_output = explainer.shap_values(X_te_rfe, check_additivity=False)
            sv          = shap_output[1] if isinstance(shap_output, list) else shap_output
            shap_df     = pd.DataFrame(sv, columns=list(X_te_rfe.columns), index=X_te_rfe.index)
            return shap_df, X_te_rfe, lgbm_clf, explainer, sv

        # ── PART A: diagnostic plots on the best fold ─────────────────────────
        best_fold      = max(self.metrics_results, key=lambda f: f["n_test_pos"])
        best_cutoff_ts = pd.Timestamp(best_fold["cutoff"])
        print(
            f"PART A — Diagnostic plots on fold: {best_cutoff_ts.date()}  "
            f"(n_pos={best_fold['n_test_pos']})"
        )

        shap_df_best, X_te_rfe_best, _, explainer_best, sv_best = _fit_and_shap(best_cutoff_ts)
        selected_features_best = list(shap_df_best.columns)

        # Plot 1 — Beeswarm (global feature impact)
        print("\n--- Plot 1/2: SHAP Beeswarm — Global Feature Impact ---")
        shap.summary_plot(
            sv_best, X_te_rfe_best,
            feature_names=selected_features_best,
            plot_type="dot", max_display=20, show=False, plot_size=None,
        )
        plt.gcf().set_size_inches(10, 8)
        plt.title(
            f"SHAP Beeswarm — LightGBM\n"
            f"Cutoff: {best_cutoff_ts.date()} | "
            f"n_test={len(X_te_rfe_best)}, n_pos={best_fold['n_test_pos']}\n"
            f"Red = high feature value  |  Blue = low  |  Right of zero = pushes toward conversion",
            fontsize=10, pad=14,
        )
        plt.tight_layout()
        plt.show()

        # Plot 2 — Waterfall for the #1 ranked lead
        fold_leads = lead_table[lead_table["cutoff"] == best_cutoff_ts].sort_values("priority_rank")
        if not fold_leads.empty:
            top_id     = fold_leads.iloc[0]["company_id"]
            top_score  = fold_leads.iloc[0]["propensity_score"]
            top_actual = int(fold_leads.iloc[0]["converted_actual"])

            if top_id in X_te_rfe_best.index:
                top_pos  = X_te_rfe_best.index.get_loc(top_id)
                base_val = (
                    explainer_best.expected_value[1]
                    if isinstance(explainer_best.expected_value, list)
                    else explainer_best.expected_value
                )
                explanation = shap.Explanation(
                    values        = sv_best[top_pos],
                    base_values   = base_val,
                    data          = X_te_rfe_best.iloc[top_pos].values,
                    feature_names = selected_features_best,
                )
                print("\n--- Plot 2/2: SHAP Waterfall — #1 Ranked Lead ---")
                print(f"  Company ID : {top_id}")
                print(f"  Score      : {top_score:.3f}")
                print(f"  Converted  : {bool(top_actual)}\n")
                shap.waterfall_plot(explanation, max_display=15, show=False)
                plt.gcf().set_size_inches(10, 7)
                plt.title(
                    f"SHAP Waterfall — Company ID {top_id}\n"
                    f"Score: {top_score:.3f}  |  Actually Converted: {bool(top_actual)}\n"
                    f"Bars show each feature's additive contribution from the base rate E[f(x)]",
                    fontsize=10, pad=14,
                )
                plt.tight_layout()
                plt.show()

        # ── PART B: accumulate signals across ALL folds ───────────────────────
        print("\nPART B — Building SHAP signals for all folds...")
        all_signals: dict = {}

        for fold in self.metrics_results:
            cutoff_ts = pd.Timestamp(fold["cutoff"])
            if fold["n_test_pos"] < 3:
                continue
            try:
                shap_df_fold, _, _, _, _ = _fit_and_shap(cutoff_ts)
                for cid, row in shap_df_fold.iterrows():
                    all_signals[(cutoff_ts, cid)] = _top_signal(row)
                print(f"  ✓ {cutoff_ts.date()} — {len(shap_df_fold)} companies enriched")
            except Exception as e:
                print(f"  ✗ {cutoff_ts.date()} — skipped ({e})")

        # ── Enrich lead table with 3 separate signal columns ──────────────────
        FALLBACK       = ["No signal computed", "", ""]
        lead_out       = lead_table.copy()
        signals_matrix = lead_out.apply(
            lambda r: all_signals.get((r["cutoff"], r["company_id"]), FALLBACK),
            axis=1,
        )
        lead_out["signal_1"] = signals_matrix.apply(lambda x: x[0])
        lead_out["signal_2"] = signals_matrix.apply(lambda x: x[1])
        lead_out["signal_3"] = signals_matrix.apply(lambda x: x[2])

        # Print enriched top-10 for the most recent fold
        most_recent = lead_out["cutoff"].max()
        print(f"\n=== ENRICHED WEEKLY SALES CALL LIST — Top 10 (Cutoff: {most_recent.date()}) ===")
        pd.set_option("display.max_colwidth", 55)
        print(
            lead_out[lead_out["cutoff"] == most_recent]
            .head(10)
            [["priority_rank", "company_id", "propensity_score", "converted_actual",
              "signal_1", "signal_2", "signal_3"]]
            .to_string(index=False)
        )

        return lead_out, shap_df_best