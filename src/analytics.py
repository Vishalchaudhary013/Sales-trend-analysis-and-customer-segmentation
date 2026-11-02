"""Descriptive and diagnostic analytics: filters, KPIs, aggregations, correlations.

All functions are pure: they take a DataFrame and return new objects, computing
every number from the data supplied. Columns that are absent (custom datasets)
are handled by returning empty results rather than raising.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src import config

Z_95 = 1.959963984540054  # two-sided 95% normal quantile


# --------------------------------------------------------------------------- #
# Global filters
# --------------------------------------------------------------------------- #


@dataclass
class Filters:
    """Global dashboard filters shared by every page.

    Empty lists mean "no restriction". ``age_range`` is inclusive.
    """

    subjects: list[str] = field(default_factory=list)
    schools: list[str] = field(default_factory=list)
    sexes: list[str] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    age_range: tuple[int, int] | None = None
    exclude_g3_zero: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the filters as plain types (for session state / logging)."""
        return asdict(self)

    def describe(self) -> str:
        """Return a short human-readable description of active filters."""
        parts = []
        for label, values in (
            ("subject", self.subjects),
            ("school", self.schools),
            ("sex", self.sexes),
            ("address", self.addresses),
        ):
            if values:
                parts.append(f"{label} = {', '.join(values)}")
        if self.age_range:
            parts.append(f"age {self.age_range[0]}–{self.age_range[1]}")
        if self.exclude_g3_zero:
            parts.append("G3 = 0 excluded")
        return "; ".join(parts) if parts else "No filters applied"


FILTER_COLUMNS: dict[str, str] = {
    "subjects": config.SUBJECT_COL,
    "schools": "school",
    "sexes": "sex",
    "addresses": "address",
}


def filter_options(df: pd.DataFrame) -> dict[str, Any]:
    """Return the values available for each filter in ``df``.

    Filters whose column is absent are omitted, so the UI can hide them.
    """
    options: dict[str, Any] = {}
    for key, col in FILTER_COLUMNS.items():
        if col in df.columns:
            options[key] = sorted(df[col].dropna().astype(str).unique().tolist())
    if "age" in df.columns and df["age"].notna().any():
        options["age_range"] = (int(df["age"].min()), int(df["age"].max()))
    if "g3_zero" in df.columns:
        options["exclude_g3_zero"] = int(df["g3_zero"].sum())
    return options


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Return the rows of ``df`` matching ``filters`` (a copy).

    Filters on columns that ``df`` lacks are ignored.
    """
    mask = pd.Series(True, index=df.index)
    for key, col in FILTER_COLUMNS.items():
        values = getattr(filters, key)
        if values and col in df.columns:
            mask &= df[col].astype(str).isin(values)
    if filters.age_range and "age" in df.columns:
        lo, hi = filters.age_range
        mask &= df["age"].between(lo, hi)
    if filters.exclude_g3_zero and "g3_zero" in df.columns:
        mask &= ~df["g3_zero"].astype(bool)
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #


def compute_kpis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute headline KPIs.

    Returns:
        ``n_records``, ``n_students`` (approximate, via match keys, or ``None``),
        ``mean_g3``, ``median_g3``, ``pass_rate``, ``fail_rate``, ``n_pass``,
        ``n_fail`` and ``by_subject`` (``{subject: {n, pass_rate}}``).
        Grade metrics are ``None`` when the data has no G3 or no rows.
    """
    kpis: dict[str, Any] = {
        "n_records": int(len(df)),
        "n_students": None,
        "mean_g3": None,
        "median_g3": None,
        "pass_rate": None,
        "fail_rate": None,
        "n_pass": None,
        "n_fail": None,
        "by_subject": {},
    }
    if "student_key" in df.columns and df["student_key"].notna().any():
        kpis["n_students"] = int(df["student_key"].nunique())
    if df.empty or config.OUTCOME_COL not in df.columns:
        return kpis
    g3 = df[config.OUTCOME_COL].dropna()
    if g3.empty:
        return kpis
    passed = df[config.TARGET_COL].dropna()
    kpis.update(
        mean_g3=float(g3.mean()),
        median_g3=float(g3.median()),
        pass_rate=float(passed.mean()),
        fail_rate=float(1 - passed.mean()),
        n_pass=int(passed.sum()),
        n_fail=int((passed == 0).sum()),
    )
    if config.SUBJECT_COL in df.columns:
        for subject, grp in df.groupby(config.SUBJECT_COL, sort=True):
            p = grp[config.TARGET_COL].dropna()
            kpis["by_subject"][str(subject)] = {
                "n": int(len(p)),
                "pass_rate": float(p.mean()) if len(p) else None,
            }
    return kpis


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


def outcome_labels(df: pd.DataFrame) -> pd.Series:
    """Return a ``"Pass"``/``"Fail"`` label series aligned to ``df``."""
    return df[config.TARGET_COL].map(config.OUTCOME_LABELS)


def mean_g3_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Mean G3 per level of ``by`` with group size and a 95% normal CI.

    Returns:
        Columns ``[by, n, mean_g3, std, ci_low, ci_high]`` sorted by ``by``.
        Empty when either column is missing.
    """
    cols = [by, "n", "mean_g3", "std", "ci_low", "ci_high"]
    if by not in df.columns or config.OUTCOME_COL not in df.columns:
        return pd.DataFrame(columns=cols)
    grp = df.dropna(subset=[by, config.OUTCOME_COL]).groupby(by, observed=True)[config.OUTCOME_COL]
    out = grp.agg(n="count", mean_g3="mean", std="std").reset_index()
    half = Z_95 * out["std"].fillna(0) / np.sqrt(out["n"])
    out["ci_low"] = out["mean_g3"] - half
    out["ci_high"] = out["mean_g3"] + half
    return out[cols].sort_values(by).reset_index(drop=True)


def pass_rate_by(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Pass rate per level of ``by`` with counts and a 95% Wilson interval.

    Returns:
        Columns ``[by, n, n_pass, pass_rate, ci_low, ci_high]``. Categorical
        levels with no rows are kept (``n = 0``, NaN rate) so bands stay visible.
    """
    cols = [by, "n", "n_pass", "pass_rate", "ci_low", "ci_high"]
    if by not in df.columns or config.TARGET_COL not in df.columns:
        return pd.DataFrame(columns=cols)
    data = df.dropna(subset=[config.TARGET_COL])
    grp = data.groupby(by, observed=False)[config.TARGET_COL]
    out = grp.agg(n="count", n_pass="sum").reset_index()
    out["n_pass"] = out["n_pass"].astype(int)
    n = out["n"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = out["n_pass"] / n
        denom = 1 + Z_95**2 / n
        centre = (p + Z_95**2 / (2 * n)) / denom
        half = Z_95 * np.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n**2)) / denom
    out["pass_rate"] = p
    out["ci_low"] = centre - half
    out["ci_high"] = centre + half
    return out[cols]


def numeric_feature_columns(df: pd.DataFrame, include_outcome: bool = False) -> list[str]:
    """Return schema numeric columns present in ``df`` (optionally with G3)."""
    return [
        c
        for c in config.NUMERIC_COLUMNS
        if c in df.columns and (include_outcome or c != config.OUTCOME_COL)
    ]


def correlations_with_g3(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of each numeric attribute with G3.

    Returns:
        Columns ``[feature, r, n]`` sorted by ``r`` ascending (so a horizontal
        bar chart reads top-to-bottom from most positive). Constant columns get
        NaN and are dropped.
    """
    if config.OUTCOME_COL not in df.columns:
        return pd.DataFrame(columns=["feature", "r", "n"])
    rows = []
    for col in numeric_feature_columns(df):
        pair = df[[col, config.OUTCOME_COL]].dropna()
        if len(pair) < 3 or pair[col].nunique() < 2 or pair[config.OUTCOME_COL].nunique() < 2:
            continue
        rows.append({"feature": col, "r": float(pair[col].corr(pair[config.OUTCOME_COL])),
                     "n": int(len(pair))})  # fmt: skip
    out = pd.DataFrame(rows, columns=["feature", "r", "n"])
    return out.sort_values("r").reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation matrix of numeric attributes including G3."""
    cols = [c for c in numeric_feature_columns(df, include_outcome=True) if df[c].nunique() > 1]
    return df[cols].corr(method="pearson")


def grade_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count records per G3 value (0–20) and outcome.

    Returns:
        Columns ``[G3, outcome, n]`` with every grade 0–20 present (zero-filled).
    """
    grades = pd.RangeIndex(0, 21, name=config.OUTCOME_COL)
    if config.OUTCOME_COL not in df.columns:
        return pd.DataFrame(columns=[config.OUTCOME_COL, "outcome", "n"])
    counts = (
        df[config.OUTCOME_COL].dropna().astype(int).value_counts().reindex(grades, fill_value=0)
    )
    out = counts.rename("n").reset_index()
    out["outcome"] = np.where(out[config.OUTCOME_COL] >= config.PASS_THRESHOLD, "Pass", "Fail")
    return out[[config.OUTCOME_COL, "outcome", "n"]]


def grade_pair_counts(df: pd.DataFrame, x: str = "G2", y: str = "G3") -> pd.DataFrame:
    """Count records for each (x, y) grade pair, with the outcome label.

    Used for an overplotting-free scatter where marker size encodes count.
    """
    if x not in df.columns or y not in df.columns:
        return pd.DataFrame(columns=[x, y, "n", "outcome"])
    out = df.dropna(subset=[x, y]).groupby([x, y]).size().rename("n").reset_index()
    out["outcome"] = np.where(out[y] >= config.PASS_THRESHOLD, "Pass", "Fail")
    return out


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return count/mean/std/min/quartiles/max for numeric schema columns."""
    cols = numeric_feature_columns(df, include_outcome=True)
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T.round(2)
