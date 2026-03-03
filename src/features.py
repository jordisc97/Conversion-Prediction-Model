import pandas as pd
import numpy as np

def _ols_slope(series: pd.Series) -> float:
    """OLS slope of series over integer positions. Used for trend features."""
    y = series.dropna().values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float) - np.arange(len(y), dtype=float).mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom else 0.0


# ============================================================
# BACKTESTER
# ============================================================

class VectorizedUsageFeatureBacktester:
    """
    Leakage-safe, vectorized feature panel for time-based backtesting.

    Produces a MultiIndex (ID, cutoff) DataFrame where each row is the
    feature state of a portal using ONLY data strictly before the cutoff.

    Cutoffs: last usage date + n_cutoffs Mondays spaced 1 month apart.

    Leakage guards:
      - Rolling features computed on pre-cutoff slice only
      - Recency dates derived from pre-cutoff slice (no negative days)
      - Entropy/diversity derived from pre-cutoff slice
    """

    def __init__(
        self,
        usage_df: pd.DataFrame,
        companies_df: pd.DataFrame,
        action_cols: list,
        user_cols: list,
        windows: dict,
        n_cutoffs: int = 6,
    ):
        self.portal_ids  = companies_df["ID"].unique()
        self.n_cutoffs   = n_cutoffs
        self.action_cols = action_cols
        self.user_cols   = user_cols
        self.windows     = windows

        self.df      = self._prepare(usage_df)
        self.cutoffs = self._generate_cutoffs()

    # ── data prep ────────────────────────────────────────────────

    def _prepare(self, usage_df: pd.DataFrame) -> pd.DataFrame:
        df = usage_df.copy()
        df["WHEN_TIMESTAMP"] = pd.to_datetime(df["WHEN_TIMESTAMP"])
        df = df[df["ID"].isin(self.portal_ids)].sort_values(["ID", "WHEN_TIMESTAMP"])
        df["TOTAL_ACTIONS"] = df[self.action_cols].sum(axis=1)
        df["TOTAL_USERS"]   = df[self.user_cols].sum(axis=1)
        df["ACTIVE_DAY"]    = (df["TOTAL_ACTIONS"] > 0).astype(int)
        return df

    # ── cutoffs ───────────────────────────────────────────────────

    def _generate_cutoffs(self) -> list:
        """Last usage date + n_cutoffs Mondays, each 1 month apart. Deduped."""
        last = self.df["WHEN_TIMESTAMP"].max()
        snap = lambda dt: dt - pd.Timedelta(days=dt.weekday())   # nearest Monday ≤ dt
        anchor = snap(last)
        mondays = [snap(anchor - pd.DateOffset(months=i)) for i in range(self.n_cutoffs)]
        return list(dict.fromkeys([last] + mondays))              # last_date first, deduped

    # ── rolling features ─────────────────────────────────────────

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """All rolling-window features on a pre-filtered slice."""
        grouped = df.set_index("WHEN_TIMESTAMP").groupby("ID")
        frames  = []

        for name, w in self.windows.items():
            roll = grouped.rolling(f"{w}D")

            f = pd.DataFrame({
                # Volume: sum is the only meaningful aggregate —
                # mean/std/max degenerate to the same value for portals
                # with 1 active day in the window (the common case here).
                f"actions_sum_{name}":  roll["TOTAL_ACTIONS"].sum(),
                f"users_sum_{name}":    roll["TOTAL_USERS"].sum(),
                # Frequency: how many days had any activity in the window
                f"active_days_{name}":  roll["ACTIVE_DAY"].sum(),
            })

            # active_ratio: fraction of window days that were active
            f[f"active_ratio_{name}"] = f[f"active_days_{name}"] / w

            # engagement depth: actions per active day (intensity proxy)
            f[f"actions_per_active_day_{name}"] = (
                f[f"actions_sum_{name}"] / (f[f"active_days_{name}"] + 1)
            )

            # trend: OLS slope of daily actions — are they ramping up or down?
            f[f"actions_trend_{name}"] = (
                grouped["TOTAL_ACTIONS"].rolling(f"{w}D").apply(_ols_slope, raw=False)
            )

            # per-module action sums (keep sums only; pct already captures the mix)
            for col in self.action_cols:
                f[f"{col.lower()}_sum_{name}"] = roll[col].sum()

            # user sums per module (volume signal per object type)
            for col in self.user_cols:
                f[f"{col.lower()}_sum_{name}"] = roll[col].sum()

            # module share: what fraction of actions went to each module?
            # Replaces per-module means (redundant with sums at 1 obs/window)
            total = f[f"actions_sum_{name}"].replace(0, np.nan)
            for col in self.action_cols:
                f[f"pct_{col.lower()}_{name}"] = (
                    f[f"{col.lower()}_sum_{name}"] / total
                ).fillna(0)

            # actions per user: intensity of module usage relative to user volume
            # Total actions / (total user + epsilon)
            total_users = f[f"users_sum_{name}"]
            for col in self.action_cols:
                f[f"{col.lower()}_per_user_{name}"] = (
                    f[f"{col.lower()}_sum_{name}"] / (total_users + 1)
                )

            frames.append(f)

        return pd.concat(frames, axis=1).reset_index()

    # ── recency features ──────────────────────────────────────────

    def _recency_features(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Last/first usage dates and derived recency columns. Computed on pre-cutoff slice."""
        active = df[df["TOTAL_ACTIONS"] > 0].groupby("ID")["WHEN_TIMESTAMP"]
        rec = pd.DataFrame({
            "last_usage_date":  active.max(),
            "first_usage_date": active.min(),
        }).reset_index()

        rec["days_since_last_usage"]  = (cutoff - rec["last_usage_date"]).dt.days
        rec["days_since_first_usage"] = (cutoff - rec["first_usage_date"]).dt.days
        rec["usage_tenure_days"]      = (rec["last_usage_date"] - rec["first_usage_date"]).dt.days
        rec["recency_score"]          = 1 / (rec["days_since_last_usage"] + 1)
        return rec.drop(columns=["last_usage_date", "first_usage_date"])

    # ── entropy features ──────────────────────────────────────────

    def _entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Module entropy and diversity. Computed on pre-cutoff slice."""
        totals = df.groupby("ID")[self.action_cols].sum()
        probs  = totals.div(totals.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
        return pd.DataFrame({
            "ID":               totals.index,
            "module_entropy":   (-(probs * np.log(probs + 1e-9)).sum(axis=1)).clip(0).values,
            "module_diversity": (totals > 0).sum(axis=1).values,
        })

    # ── build ─────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """Build and return the full (ID × cutoff) feature panel."""
        snapshots = []

        for cutoff in self.cutoffs:
            sl = self.df[self.df["WHEN_TIMESTAMP"] < cutoff]
            if sl.empty:
                continue

            snap = (
                self._rolling_features(sl)
                .sort_values(["ID", "WHEN_TIMESTAMP"])
                .groupby("ID").tail(1)
            ).copy()
            snap["cutoff"] = cutoff

            snap = snap.merge(self._recency_features(sl, cutoff), on="ID", how="left")
            snap = snap.merge(self._entropy_features(sl),          on="ID", how="left")
            snapshots.append(snap)

        panel = pd.concat(snapshots, ignore_index=True)

        if {"actions_sum_30d", "actions_sum_60d"}.issubset(panel.columns):
            panel["actions_accel"] = panel["actions_sum_30d"] / (panel["actions_sum_60d"] + 1)

        # Full (ID × cutoff) index — zero-usage portals filled with sentinels
        full_idx = pd.MultiIndex.from_product(
            [self.portal_ids, self.cutoffs], names=["ID", "cutoff"]
        )
        panel = panel.set_index(["ID", "cutoff"]).reindex(full_idx).sort_index()

        # Recency sentinel for never-active portals: large number, not 0
        span = int((self.cutoffs[0] - self.cutoffs[-1]).days + 1)
        for col in ["days_since_last_usage", "days_since_first_usage"]:
            if col in panel.columns:
                panel[col] = panel[col].fillna(span)

        return panel.fillna(0)
