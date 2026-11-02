"""Per-customer reason codes for churn predictions.

Method (model-agnostic *occlusion*): for each feature, replace the customer's
value with the training median and re-score. The drop in churn probability is
that feature's contribution for this customer. The top contributions that
*raise* risk become human-readable reasons, phrased with the customer's value
and its percentile among training customers (e.g. "Last purchase 150 days ago
— longer than 82% of customers").
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src import config
from src.modeling import TrainedModel

C = config
MIN_CONTRIBUTION = 0.005  # ignore effects below half a percentage point


def contributions(model: TrainedModel, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
    """Return per-feature occlusion contributions (probability points).

    Args:
        model: Trained model with a reference (training) matrix.
        X: Customers to explain (columns include ``model.features``).

    Returns:
        DataFrame aligned to ``X`` with one column per feature: the churn
        probability minus the probability with that feature set to the
        training median. Positive = the feature raises this customer's risk.
    """
    feats = model.features
    ref = model.reference if model.reference is not None else X[feats]
    medians = ref[feats].median()
    base = model.pipeline.predict_proba(X[feats])[:, 1]
    out = pd.DataFrame(index=X.index, columns=feats, dtype=float)
    for f in feats:
        alt = X[feats].copy()
        alt[f] = medians[f]
        out[f] = base - model.pipeline.predict_proba(alt)[:, 1]
    return out


def _share(x: float) -> str:
    """Format a population share, avoiding impossible "100%" / "0%" claims."""
    if x >= 0.995:
        return "over 99%"
    if x < 0.01:
        return "under 1%"
    return f"{x:.0%}"


def _relative(ref: pd.Series, v: float, more: str, less: str) -> str:
    """Compare ``v`` with reference values, in whichever direction is true.

    Returns e.g. "higher than 82% of customers" or "lower than 64% of
    customers", choosing the direction that describes most of the population.
    """
    r = ref.dropna()
    if not len(r):
        return ""
    above, below = float((r < v).mean()), float((r > v).mean())
    if above >= below:
        return f"{more} than {_share(above)} of customers"
    return f"{less} than {_share(below)} of customers"


def _money(v: float, cur: str) -> str:
    return f"{cur}{v:,.0f}"


def _n(v: float, word: str) -> str:
    """Pluralise ``word`` for count ``v`` ("1 order", "3 orders")."""
    return f"{v:,.0f} {word}{'' if round(v) == 1 else 's'}"


def _phrase(feature: str, v: float, ref: pd.Series, cur: str) -> str:
    """Describe one feature value relative to training customers (direction-true)."""

    def rel(more: str = "higher", less: str = "lower") -> str:
        return _relative(ref, v, more, less)

    templates: dict[str, Callable[[], str]] = {
        "recency_days": lambda: (
            f"Last purchase {_n(v, 'day')} ago — " + rel("longer ago", "more recent")
        ),
        "frequency": lambda: f"{_n(v, 'order')} in total — " + rel("more", "fewer"),
        "monetary": lambda: f"Total spend {_money(v, cur)} — " + rel(),
        "avg_order_value": lambda: f"Average order {_money(v, cur)} — " + rel(),
        "tenure_days": lambda: f"Customer for {_n(v, 'day')} — " + rel("longer", "shorter"),
        "distinct_products": lambda: (
            f"Bought {_n(v, 'distinct product')} — " + rel("more", "fewer")
        ),
        "lines_per_order": lambda: (
            (f"{_n(v, 'line')} per order" if float(v).is_integer() else f"{v:,.1f} lines per order")
            + " — "
            + rel("more", "fewer")
        ),
        "orders_last_90d": lambda: (
            "No orders in the last 90 days"
            if v == 0
            else f"{_n(v, 'order')} in the last 90 days — " + rel("more", "fewer")
        ),
        "revenue_last_90d": lambda: (
            "No spend in the last 90 days"
            if v == 0
            else f"Spend in the last 90 days {_money(v, cur)} — " + rel()
        ),
        "mean_days_between_orders": lambda: (
            "Only one order so far (no repeat purchase)"
            if np.isnan(v)
            else f"An order every {_n(v, 'day')} on average — " + rel("less often", "more often")
        ),
        "cancellation_rate": lambda: f"{v:.0%} of orders cancelled — " + rel(),
        "is_uk": lambda: "Customer outside the UK" if v == 0 else "UK customer",
    }
    maker = templates.get(feature)
    return maker() if maker else f"{C.FEATURE_LABELS.get(feature, feature)}: {v:,.2f}"


def reason_codes(
    model: TrainedModel,
    X: pd.DataFrame,  # noqa: N803 - conventional ML naming
    top_n: int = C.TOP_N_REASONS,
    cur: str = C.CURRENCY_SYMBOL,
    monetary_factor: pd.Series | float = 1.0,
) -> pd.DataFrame:
    """Return the top risk-raising reasons per customer.

    Args:
        model: Trained model.
        X: Customers' features (base currency).
        top_n: Number of reasons per customer.
        cur: Currency prefix for monetary values in the text.
        monetary_factor: Multiplier to show monetary values in the display
            currency (scalar, or per-customer Series aligned to ``X``); the
            percentiles are unaffected.

    Returns:
        Columns ``reason_1..reason_n`` (empty string when fewer reasons apply)
        and ``contribution_1..n`` (probability points), aligned to ``X``.
    """
    contrib = contributions(model, X)
    ref = model.reference if model.reference is not None else X
    money_feats = {"monetary", "avg_order_value", "revenue_last_90d"}
    factor = (
        monetary_factor.reindex(X.index).fillna(1.0)
        if isinstance(monetary_factor, pd.Series)
        else pd.Series(float(monetary_factor), index=X.index)
    )
    rows = []
    for cid, row in contrib.iterrows():
        ranked = row[row > MIN_CONTRIBUTION].sort_values(ascending=False).head(top_n)
        rec: dict[str, object] = {}
        for i in range(1, top_n + 1):
            if i <= len(ranked):
                f = ranked.index[i - 1]
                v = float(X.at[cid, f])
                shown = v * float(factor.at[cid]) if f in money_feats and not np.isnan(v) else v
                # Percentile is computed in base currency; phrase with displayed value.
                text = _phrase(f, v, ref[f], cur)
                if f in money_feats and not np.isnan(v):
                    text = text.replace(_money(v, cur), _money(shown, cur), 1)
                rec[f"reason_{i}"] = text
                rec[f"contribution_{i}"] = float(ranked.iloc[i - 1])
            else:
                rec[f"reason_{i}"] = ""
                rec[f"contribution_{i}"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows, index=X.index)
