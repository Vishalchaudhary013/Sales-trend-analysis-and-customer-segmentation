"""Sales analytics: global filters, KPIs, time series, mix and customer dynamics.

All functions are pure (DataFrame in, new objects out) and compute every figure
from the data they receive. They expect the clean table from
:func:`src.ingestion.prepare` and degrade gracefully when optional columns are
absent (custom datasets).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src import config

C = config  # short alias for column-name constants

FREQ_ALIASES: dict[str, str] = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M"}
WEEKDAYS: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


@dataclass
class Filters:
    """Global dashboard filters shared by every page.

    Attributes:
        date_from: Inclusive start date, or ``None`` for the data minimum.
        date_to: Inclusive end date, or ``None`` for the data maximum.
        countries: Countries to keep; empty means all.
        customer_scope: ``"all"``, ``"identified"`` (has CustomerID) or ``"guest"``.
        include_returns: If True, cancellations and write-offs are netted into
            revenue; otherwise only genuine sales lines are analysed.
        include_non_product: If True, postage/fee lines are included.
    """

    date_from: date | None = None
    date_to: date | None = None
    countries: list[str] = field(default_factory=list)
    customer_scope: str = "all"
    include_returns: bool = False
    include_non_product: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the filters as plain types."""
        out = asdict(self)
        for key in ("date_from", "date_to"):
            out[key] = out[key].isoformat() if out[key] else None
        return out

    def describe(self) -> str:
        """Return a short human-readable description of the active filters."""
        parts = []
        if self.date_from or self.date_to:
            parts.append(f"{self.date_from or 'start'} to {self.date_to or 'end'}")
        if self.countries:
            shown = ", ".join(self.countries[:3]) + ("…" if len(self.countries) > 3 else "")
            parts.append(f"countries: {shown}")
        if self.customer_scope != "all":
            parts.append(f"{self.customer_scope} customers only")
        parts.append("net of returns" if self.include_returns else "completed sales")
        if self.include_non_product:
            parts.append("incl. postage & fees")
        return "; ".join(parts)


def filter_options(df: pd.DataFrame) -> dict[str, Any]:
    """Return values available to each filter (omitting unsupported ones)."""
    opts: dict[str, Any] = {
        "date_range": (df[C.COL_DATE].min().date(), df[C.COL_DATE].max().date()),
        "customer_scope": df[C.COL_HAS_CUSTOMER].any(),
    }
    if C.COL_COUNTRY in df.columns:
        revenue = df.groupby(C.COL_COUNTRY)[C.COL_REVENUE].sum().sort_values(ascending=False)
        opts["countries"] = revenue.index.astype(str).tolist()
    return opts


def line_mask(df: pd.DataFrame, filters: Filters) -> pd.Series:
    """Return the mask of transaction lines that count towards analyses.

    Adjustments are always excluded. Without ``include_returns`` only completed
    sales count: positive quantity and price, not a cancellation, and not later
    reversed by a matching cancellation. With it, every line is netted.
    """
    mask = ~df[C.COL_IS_ADJUST]
    if not filters.include_non_product:
        mask &= df[C.COL_IS_PRODUCT]
    if not filters.include_returns:
        mask &= (
            ~df[C.COL_IS_CANCEL] & (df[C.COL_QTY] > 0) & (df[C.COL_PRICE] > 0)
            & ~df[C.COL_IS_REVERSED]
        )  # fmt: skip
    return mask


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Return the lines matching all filters (a copy).

    The result still contains cancellation lines' *existence* only when
    ``include_returns`` is set; use :func:`scope_filter` for rate metrics that
    need cancellations regardless.
    """
    out = scope_filter(df, filters)
    return out.loc[line_mask(out, filters)].copy()


def scope_filter(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Apply date, country and customer-scope filters only (keeps every line type)."""
    mask = pd.Series(True, index=df.index)
    if filters.date_from:
        mask &= df[C.COL_DATE] >= pd.Timestamp(filters.date_from)
    if filters.date_to:
        mask &= df[C.COL_DATE] < pd.Timestamp(filters.date_to) + pd.Timedelta(days=1)
    if filters.countries and C.COL_COUNTRY in df.columns:
        mask &= df[C.COL_COUNTRY].isin(filters.countries)
    if filters.customer_scope == "identified":
        mask &= df[C.COL_HAS_CUSTOMER]
    elif filters.customer_scope == "guest":
        mask &= ~df[C.COL_HAS_CUSTOMER]
    return df.loc[mask].copy()


# --------------------------------------------------------------------------- #
# KPIs
# --------------------------------------------------------------------------- #


def _orders(df: pd.DataFrame) -> pd.Series:
    """Invoice numbers of non-cancellation lines."""
    return df.loc[~df[C.COL_IS_CANCEL], C.COL_INVOICE]


def compute_kpis(lines: pd.DataFrame, scope: pd.DataFrame | None = None) -> dict[str, Any]:
    """Headline KPIs for the analysed lines.

    Args:
        lines: Output of :func:`apply_filters`.
        scope: Output of :func:`scope_filter` (all line types) used for the
            cancellation rate; when omitted the rate is ``None``.

    Returns:
        ``revenue``, ``orders``, ``customers``, ``aov`` (revenue per order),
        ``units``, ``lines_per_order``, ``revenue_per_customer``,
        ``identified_revenue_share``, ``cancellation_rate`` (cancelled invoices
        / all invoices), ``cancelled_value`` and ``first_date``/``last_date``.
    """
    orders = _orders(lines).nunique()
    customers = lines[C.COL_CUSTOMER].nunique()
    revenue = float(lines[C.COL_REVENUE].sum())
    ident = float(lines.loc[lines[C.COL_HAS_CUSTOMER], C.COL_REVENUE].sum())
    kpis: dict[str, Any] = {
        "revenue": revenue,
        "orders": int(orders),
        "customers": int(customers),
        "aov": revenue / orders if orders else None,
        "units": int(lines.loc[lines[C.COL_QTY] > 0, C.COL_QTY].sum()),
        "lines_per_order": float((~lines[C.COL_IS_CANCEL]).sum() / orders) if orders else None,
        "revenue_per_customer": ident / customers if customers else None,
        "identified_revenue_share": ident / revenue if revenue > 0 else None,
        "cancellation_rate": None,
        "cancelled_value": None,
        "first_date": lines[C.COL_DATE].min() if len(lines) else None,
        "last_date": lines[C.COL_DATE].max() if len(lines) else None,
    }
    if scope is not None and len(scope):
        invoices = scope.loc[~scope[C.COL_IS_ADJUST], [C.COL_INVOICE, C.COL_IS_CANCEL]]
        invoices = invoices.drop_duplicates(C.COL_INVOICE)
        if len(invoices):
            kpis["cancellation_rate"] = float(invoices[C.COL_IS_CANCEL].mean())
        kpis["cancelled_value"] = float(-scope.loc[scope[C.COL_IS_CANCEL], C.COL_REVENUE].sum())
    return kpis


def period_comparison(lines: pd.DataFrame, data_end: pd.Timestamp) -> dict[str, Any] | None:
    """Compare the last *complete* calendar month with the month before it.

    Args:
        lines: Analysed lines.
        data_end: Last timestamp in the full (unfiltered) dataset, used to
            decide whether the final month is complete.

    Returns:
        ``{"current": period, "previous": period, "metrics": {name: (cur, prev,
        pct_change)}}`` or ``None`` if fewer than two complete months exist.
    """
    monthly = time_series(lines, "Monthly", data_end)
    complete = monthly[~monthly["partial"]]
    if len(complete) < 2:
        return None
    cur, prev = complete.iloc[-1], complete.iloc[-2]
    metrics = {}
    for name in ("revenue", "orders", "customers", "aov"):
        c, p = float(cur[name]), float(prev[name])
        metrics[name] = (c, p, (c - p) / p if p else None)
    return {"current": cur["period_label"], "previous": prev["period_label"], "metrics": metrics}


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #


def time_series(lines: pd.DataFrame, freq: str, data_end: pd.Timestamp | None = None,
                data_start: pd.Timestamp | None = None) -> pd.DataFrame:  # fmt: skip
    """Aggregate revenue, orders, customers and AOV per period.

    Args:
        lines: Analysed lines.
        freq: ``"Daily"``, ``"Weekly"`` (weeks start Monday) or ``"Monthly"``.
        data_end: Dataset end; periods ending after it are flagged ``partial``.
        data_start: Dataset start; periods starting before it are flagged too.

    Returns:
        Columns ``period_start, period_label, revenue, orders, customers, units,
        aov, growth, partial``. Empty periods inside the range are zero-filled
        (``aov`` NaN). ``growth`` is the change versus the previous period and
        is NaN when either period is partial.
    """
    cols = ["period_start", "period_label", "revenue", "orders", "customers", "units", "aov",
            "growth", "partial"]  # fmt: skip
    if lines.empty:
        return pd.DataFrame(columns=cols)
    alias = FREQ_ALIASES[freq]  # "W-SUN" = weeks ending Sunday, i.e. starting Monday
    period = lines[C.COL_DATE].dt.to_period(alias)
    grouped = lines.assign(_p=period).groupby("_p")
    out = pd.DataFrame({
        "revenue": grouped[C.COL_REVENUE].sum(),
        "orders": grouped.apply(lambda g: _orders(g).nunique(), include_groups=False),
        "customers": grouped[C.COL_CUSTOMER].nunique(),
        "units": grouped[C.COL_QTY].sum(),
    })  # fmt: skip
    full = pd.period_range(out.index.min(), out.index.max(), freq=out.index.freq)
    out = out.reindex(full, fill_value=0)
    out["aov"] = out["revenue"] / out["orders"].replace(0, np.nan)
    out["growth"] = out["revenue"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    end = data_end if data_end is not None else lines[C.COL_DATE].max()
    start = data_start if data_start is not None else lines[C.COL_DATE].min()
    out["partial"] = (out.index.end_time.normalize() > end.normalize()) | (
        out.index.start_time.normalize() < start.normalize()
    )
    # Growth involving a partial period would compare unequal spans: suppress it.
    out.loc[out["partial"] | out["partial"].shift(1, fill_value=False), "growth"] = np.nan
    out["period_start"] = out.index.start_time
    fmt = {"Daily": "%d %b %Y", "Weekly": "w/c %d %b %Y", "Monthly": "%b %Y"}[freq]
    out["period_label"] = out["period_start"].dt.strftime(fmt)
    return out.reset_index(drop=True)[cols]


def weekday_hour_matrix(lines: pd.DataFrame, value: str = "revenue") -> pd.DataFrame:
    """Revenue or order counts by weekday (rows, Mon–Sun) and hour (columns).

    Weekdays/hours with no transactions are kept as NaN (distinct from zero)
    so the UI can show "no trading" rather than a misleading 0.
    """
    if lines.empty:
        return pd.DataFrame()
    dow = lines[C.COL_DATE].dt.day_of_week
    hour = lines[C.COL_DATE].dt.hour
    if value == "orders":
        tmp = lines.loc[~lines[C.COL_IS_CANCEL]].assign(_d=dow, _h=hour)
        mat = tmp.groupby(["_d", "_h"])[C.COL_INVOICE].nunique().unstack()
    else:
        mat = lines.assign(_d=dow, _h=hour).groupby(["_d", "_h"])[C.COL_REVENUE].sum().unstack()
    hours = range(int(hour.min()), int(hour.max()) + 1)
    mat = mat.reindex(index=range(7), columns=hours)
    mat.index = list(WEEKDAYS)
    return mat


# --------------------------------------------------------------------------- #
# Mix: products and countries
# --------------------------------------------------------------------------- #


def top_products(lines: pd.DataFrame, n: int = 15, by: str = "revenue") -> pd.DataFrame:
    """Top ``n`` products by revenue or units.

    Returns:
        Columns ``StockCode, Description, revenue, units, orders, share`` where
        ``share`` is the product's share of total revenue in ``lines`` and the
        description is the most frequent one recorded for the code.
    """
    cols = [C.COL_STOCK, C.COL_DESC, "revenue", "units", "orders", "share"]
    if lines.empty:
        return pd.DataFrame(columns=cols)
    g = lines.groupby(C.COL_STOCK)
    out = pd.DataFrame({
        "revenue": g[C.COL_REVENUE].sum(),
        "units": g[C.COL_QTY].sum(),
        "orders": g[C.COL_INVOICE].nunique(),
    })  # fmt: skip
    if C.COL_DESC in lines.columns:
        desc = (
            lines.dropna(subset=[C.COL_DESC])
            .groupby(C.COL_STOCK)[C.COL_DESC]
            .agg(lambda s: s.value_counts().index[0])
        )
        out[C.COL_DESC] = desc
    else:
        out[C.COL_DESC] = pd.NA
    total = lines[C.COL_REVENUE].sum()
    out["share"] = out["revenue"] / total if total else np.nan
    key = "units" if by == "units" else "revenue"
    return out.sort_values(key, ascending=False).head(n).reset_index()[cols]


def country_summary(lines: pd.DataFrame) -> pd.DataFrame:
    """Revenue, orders, customers, AOV and revenue share per country (desc)."""
    cols = [C.COL_COUNTRY, "revenue", "orders", "customers", "aov", "share"]
    if lines.empty or C.COL_COUNTRY not in lines.columns:
        return pd.DataFrame(columns=cols)
    g = lines.groupby(C.COL_COUNTRY)
    out = pd.DataFrame({
        "revenue": g[C.COL_REVENUE].sum(),
        "orders": g.apply(lambda x: _orders(x).nunique(), include_groups=False),
        "customers": g[C.COL_CUSTOMER].nunique(),
    })  # fmt: skip
    out["aov"] = out["revenue"] / out["orders"].replace(0, np.nan)
    out["share"] = out["revenue"] / out["revenue"].sum()
    return out.sort_values("revenue", ascending=False).reset_index()[cols]


def large_lines(scope: pd.DataFrame, threshold: int = config.LARGE_LINE_QUANTITY) -> pd.DataFrame:
    """Return lines whose absolute quantity is at least ``threshold`` (outlier review)."""
    wanted = (C.COL_INVOICE, C.COL_DATE, C.COL_STOCK, C.COL_DESC, C.COL_QTY, C.COL_REVENUE,
              C.COL_CUSTOMER, C.COL_IS_REVERSED)  # fmt: skip
    cols = [c for c in wanted if c in scope.columns]
    return scope.loc[scope[C.COL_QTY].abs() >= threshold, cols].reset_index(drop=True)


def revenue_concentration(lines: pd.DataFrame, key: str = C.COL_STOCK) -> dict[str, float]:
    """Share of revenue from the top 1%, 10% and 20% of ``key`` values (Pareto check)."""
    rev = lines.groupby(key)[C.COL_REVENUE].sum().sort_values(ascending=False)
    rev = rev[rev > 0]
    if rev.empty:
        return {}
    total = rev.sum()
    return {f"top_{int(p * 100)}pct": float(rev.head(max(1, int(np.ceil(len(rev) * p)))).sum()
                                             / total) for p in (0.01, 0.1, 0.2)}  # fmt: skip


# --------------------------------------------------------------------------- #
# Customer dynamics
# --------------------------------------------------------------------------- #


def new_vs_returning(lines: pd.DataFrame, history: pd.DataFrame,
                     data_end: pd.Timestamp | None = None) -> pd.DataFrame:  # fmt: skip
    """Monthly counts of new and returning identified customers.

    A customer is *new* in the month of their first purchase in ``history``
    (the full, unfiltered-by-date sales lines), so date filters do not turn
    returning customers into new ones. Customers in the first month of the data
    are all "new" by construction (left-censoring); that month is flagged.

    Returns:
        Columns ``period_start, period_label, new, returning, partial, censored``.
    """
    cols = ["period_start", "period_label", "new", "returning", "partial", "censored"]
    ident = lines.loc[lines[C.COL_HAS_CUSTOMER] & ~lines[C.COL_IS_CANCEL]]
    if ident.empty:
        return pd.DataFrame(columns=cols)
    hist = history.loc[history[C.COL_HAS_CUSTOMER] & ~history[C.COL_IS_CANCEL]]
    first = hist.groupby(C.COL_CUSTOMER)[C.COL_DATE].min().dt.to_period("M")
    month = ident[C.COL_DATE].dt.to_period("M")
    active = pd.DataFrame({"cust": ident[C.COL_CUSTOMER], "month": month}).drop_duplicates()
    active["is_new"] = active["month"] == active["cust"].map(first)
    counts = active.groupby("month")["is_new"].agg(new="sum", total="count")
    counts["returning"] = counts["total"] - counts["new"]
    end = data_end if data_end is not None else history[C.COL_DATE].max()
    out = pd.DataFrame({
        "period_start": counts.index.start_time,
        "period_label": counts.index.strftime("%b %Y"),
        "new": counts["new"].astype(int).to_numpy(),
        "returning": counts["returning"].astype(int).to_numpy(),
        "partial": (counts.index.end_time.normalize() > end.normalize()),
        "censored": counts.index == hist[C.COL_DATE].min().to_period("M"),
    })  # fmt: skip
    return out.reset_index(drop=True)[cols]


def cancellation_trend(scope: pd.DataFrame) -> pd.DataFrame:
    """Monthly cancelled-invoice rate and cancelled value from all line types."""
    cols = ["period_start", "period_label", "invoices", "cancelled", "rate", "cancelled_value"]
    data = scope.loc[~scope[C.COL_IS_ADJUST]]
    if data.empty:
        return pd.DataFrame(columns=cols)
    inv = data.drop_duplicates(C.COL_INVOICE).assign(_m=lambda d: d[C.COL_DATE].dt.to_period("M"))
    agg = inv.groupby("_m")[C.COL_IS_CANCEL].agg(invoices="count", cancelled="sum")
    value = (
        -data.loc[data[C.COL_IS_CANCEL]]
        .groupby(data[C.COL_DATE].dt.to_period("M"))[C.COL_REVENUE]
        .sum()
    )
    agg["cancelled_value"] = value.reindex(agg.index, fill_value=0.0)
    agg["rate"] = agg["cancelled"] / agg["invoices"]
    agg["period_start"] = agg.index.start_time
    agg["period_label"] = agg.index.strftime("%b %Y")
    return agg.reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- #
# Part-to-whole breakdowns (donut charts)
# --------------------------------------------------------------------------- #


def _top_with_other(values: pd.Series, top_n: int) -> pd.DataFrame:
    """Keep the ``top_n`` largest entries and fold the rest into ``"Other"``."""
    values = values[values > 0].sort_values(ascending=False)
    head = values.head(top_n)
    rest = values.iloc[top_n:].sum()
    if rest > 0:
        head = pd.concat([head, pd.Series({"Other": rest})])
    total = head.sum()
    out = head.rename("value").rename_axis("label").reset_index()
    out["share"] = out["value"] / total if total else np.nan
    return out


def revenue_share(lines: pd.DataFrame, by: str = C.COL_COUNTRY,
                  top_n: int = config.MAX_PIE_SLICES - 1) -> pd.DataFrame:  # fmt: skip
    """Revenue share by ``by`` (top ``top_n`` + "Other").

    Returns:
        Columns ``label, value, share``; empty when ``by`` is missing.
    """
    if by not in lines.columns or lines.empty:
        return pd.DataFrame(columns=["label", "value", "share"])
    return _top_with_other(lines.groupby(by)[C.COL_REVENUE].sum(), top_n)


def revenue_by_customer_type(lines: pd.DataFrame) -> pd.DataFrame:
    """Revenue from identified customers versus guest (no CustomerID) sales."""
    if lines.empty:
        return pd.DataFrame(columns=["label", "value", "share"])
    kind = np.where(lines[C.COL_HAS_CUSTOMER], "Identified customers", "Guest sales")
    return _top_with_other(lines.groupby(kind)[C.COL_REVENUE].sum(), 2)


def invoice_status_mix(scope: pd.DataFrame) -> pd.DataFrame:
    """Count of completed versus cancelled invoices (adjustments excluded)."""
    data = scope.loc[~scope[C.COL_IS_ADJUST]].drop_duplicates(C.COL_INVOICE)
    if data.empty:
        return pd.DataFrame(columns=["label", "value", "share"])
    kind = np.where(data[C.COL_IS_CANCEL], "Cancelled", "Completed")
    return _top_with_other(pd.Series(kind).value_counts().astype(float), 2)
