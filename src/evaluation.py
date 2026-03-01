"""
evaluation.py
─────────────
EvaluationMixin for PropensityBacktester.

Provides post-run quantitative analysis methods:
  - run_baselines          : random + activity-heuristic baselines vs. model
  - plot_pr_curves_for_fold: precision-recall curves for a single backtest fold

These methods assume run_backtest() has already been called and that
self.metrics_results, self.all_cutoffs, self._build_fold_dataset(), and
self._get_model_pipelines() are available via the host class.
"""
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


class EvaluationMixin:
    """Mixin that adds baseline comparison and PR-curve plotting to PropensityBacktester."""

    def run_baselines(self, top_k: int = 10) -> pd.DataFrame:
        """
        For each backtest fold, compute:
          - Random baseline P@K and Rec@K (averaged over 200 random seeds)
          - Most-active-user heuristic P@K and Rec@K

        Returns a DataFrame mirroring the metrics_summary structure.
        Call after run_backtest(). Merge the returned DataFrame with
        self.metrics_results to compare against Metamodel performance.
        """
        results = []

        for cutoff in self.all_cutoffs:
            cutoff_ts = pd.Timestamp(cutoff)

            try:
                X_tr, y_tr, X_te, y_te = self._build_fold_dataset(cutoff_ts)
            except Exception:
                continue

            if y_te.sum() < 3:
                continue

            y_arr      = np.array(y_te)
            n          = len(y_arr)
            total_pos  = y_arr.sum()
            prevalence = total_pos / n

            # ── Random baseline (Monte Carlo average) ────────────────────────
            rng = np.random.default_rng(42)
            random_hits = [
                y_arr[rng.choice(n, size=min(top_k, n), replace=False)].sum()
                for _ in range(200)
            ]
            avg_random_hits = np.mean(random_hits)

            # ── Activity heuristic baseline ──────────────────────────────────
            # Rank by 30d action volume — the simplest no-ML heuristic a BA could build.
            if "actions_sum_30d" in X_te.columns:
                activity_scores = X_te["actions_sum_30d"].fillna(0).values
            else:
                act_cols = [c for c in X_te.columns if "actions_sum" in c]
                activity_scores = X_te[act_cols].fillna(0).sum(axis=1).values

            top_k_active_idx = np.argsort(activity_scores)[::-1][:top_k]
            active_hits = y_arr[top_k_active_idx].sum()

            results.append({
                "cutoff":          cutoff_ts,
                "n_test_pos":      int(total_pos),
                "prevalence":      prevalence,
                "random_p_at_k":   avg_random_hits / top_k,
                "random_rec_at_k": avg_random_hits / total_pos if total_pos > 0 else 0,
                "active_p_at_k":   active_hits / top_k,
                "active_rec_at_k": active_hits / total_pos if total_pos > 0 else 0,
            })

        return pd.DataFrame(results)

    def plot_pr_curves_for_fold(
        self,
        target_cutoff: Optional[pd.Timestamp] = None,
        top_k: int = 10,
    ) -> None:
        """
        Re-trains all models on the specified cutoff fold and plots their PR curves
        alongside the activity-heuristic baseline and the random (prevalence) line.

        Uses the fold with the most test positives if target_cutoff is None,
        as that fold gives the most statistically reliable PR estimate.

        Call after run_backtest().
        """
        import logging
        if not self.metrics_results:
            logging.warning("No results found. Run run_backtest() first.")
            return

        best_fold = max(self.metrics_results, key=lambda f: f["n_test_pos"])
        if target_cutoff is None:
            target_cutoff = best_fold["cutoff"]

        print(
            f"Plotting PR curves for cutoff: {pd.Timestamp(target_cutoff).date()} "
            f"(n_pos={best_fold['n_test_pos']})"
        )

        cutoff_ts             = pd.Timestamp(target_cutoff)
        X_tr, y_tr, X_te, y_te = self._build_fold_dataset(cutoff_ts)
        valid_cols            = X_tr.columns[X_tr.notna().any()].tolist()
        X_tr, X_te            = X_tr[valid_cols], X_te[valid_cols]
        models                = self._get_model_pipelines(total_input_features=len(valid_cols))

        colors = {
            "RandomForest": "#7f8c8d",
            "LogisticReg":  "#3498db",
            "LightGBM":     "#e67e22",
            "Metamodel":    "#2ecc71",
        }

        # Activity heuristic scores
        if "actions_sum_30d" in X_te.columns:
            heuristic_scores = X_te["actions_sum_30d"].fillna(0).values
        else:
            act_cols = [c for c in X_te.columns if "actions_sum" in c]
            heuristic_scores = X_te[act_cols].fillna(0).sum(axis=1).values

        fig, ax = plt.subplots(figsize=(9, 6))
        prevalence = y_te.mean()
        ax.axhline(
            y=prevalence, color="gray", linestyle=":", linewidth=1.5,
            label=f"Random (prevalence = {prevalence:.1%})",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prec_h, rec_h, _ = precision_recall_curve(y_te, heuristic_scores)
        ap_h = average_precision_score(y_te, heuristic_scores)
        ax.plot(rec_h, prec_h, "--", color="black", linewidth=1.5,
                label=f"Activity Heuristic (AP={ap_h:.2f})")

        for name, pipeline in models.items():
            pipeline.fit(X_tr, y_tr)
            y_prob = pipeline.predict_proba(X_te)[:, 1]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prec, rec, _ = precision_recall_curve(y_te, y_prob)
            ap = average_precision_score(y_te, y_prob)
            ax.plot(
                rec, prec,
                color=colors[name],
                linewidth=2.5 if name == "Metamodel" else 1.5,
                label=f"{name} (AP={ap:.2f})",
            )

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            f"Precision-Recall Curves\n"
            f"Cutoff: {pd.Timestamp(target_cutoff).date()} | "
            f"n_test={len(y_te)}, n_pos={int(y_te.sum())}",
            fontsize=12,
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.show()