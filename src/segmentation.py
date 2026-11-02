"""Customer segmentation: RFM scoring, rule-based segments and K-Means clustering.

RFM is computed per identified customer from *completed sales* (see
:func:`src.analytics.line_mask`) so monetary values are positive:

* **Recency**: calendar days from the customer's last purchase date to the snapshot date
  (last invoice date in the data + :data:`config.RFM_SNAPSHOT_OFFSET_DAYS`).
* **Frequency**: number of distinct invoices.
* **Monetary**: total revenue in the base currency.

Scores are quintiles (1–5, 5 = best). Recency is scored in reverse (recent =
5). Frequency is ranked before binning because many customers tie on 1 order.

K-Means runs on ``log1p`` + standardised R, F and M inside a scikit-learn
Pipeline with a fixed seed. The recommended ``k`` is the Kneedle elbow of the
inertia curve; silhouette and Davies–Bouldin scores are reported alongside.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config

C = config
FEATURES = list(config.RFM_FEATURES)


# --------------------------------------------------------------------------- #
# RFM
# --------------------------------------------------------------------------- #


def snapshot_date(lines: pd.DataFrame) -> pd.Timestamp:
    """Return the RFM reference date: last invoice day + the configured offset."""
    return lines[C.COL_DATE].max().normalize() + pd.Timedelta(days=C.RFM_SNAPSHOT_OFFSET_DAYS)


def compute_rfm(lines: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute recency, frequency and monetary value per identified customer.

    Args:
        lines: Completed-sales lines (cancellations/returns already excluded).
        snapshot: Reference date; defaults to :func:`snapshot_date`.

    Returns:
        DataFrame indexed by ``CustomerID`` with ``recency`` (days),
        ``frequency`` (invoices), ``monetary`` (revenue), ``first_purchase``,
        ``last_purchase`` and ``country`` (most frequent, if available).
        Customers with non-positive monetary value are dropped.
    """
    data = lines.loc[lines[C.COL_HAS_CUSTOMER] & (lines[C.COL_REVENUE] > 0)]
    cols = [*FEATURES, "first_purchase", "last_purchase", "country"]
    if data.empty:
        return pd.DataFrame(columns=cols).rename_axis(C.COL_CUSTOMER)
    snap = snapshot if snapshot is not None else snapshot_date(data)
    g = data.groupby(C.COL_CUSTOMER)
    out = pd.DataFrame(
        {
            "recency": (snap - g[C.COL_DATE].max().dt.normalize()).dt.days.astype(int),
            "frequency": g[C.COL_INVOICE].nunique().astype(int),
            "monetary": g[C.COL_REVENUE].sum().astype(float),
            "first_purchase": g[C.COL_DATE].min(),
            "last_purchase": g[C.COL_DATE].max(),
        }
    )
    if C.COL_COUNTRY in data.columns:
        out["country"] = g[C.COL_COUNTRY].agg(
            lambda s: s.mode().iat[0] if s.notna().any() else pd.NA
        )
    else:
        out["country"] = pd.NA
    return out[out["monetary"] > 0][cols]


def _quintile(values: pd.Series, reverse: bool = False) -> pd.Series:
    """Score values into 1..q by rank-based quantiles (ties broken by order)."""
    q = C.RFM_QUANTILES
    n = len(values)
    if n == 0:
        return pd.Series(dtype=int)
    ranks = values.rank(method="first", ascending=not reverse)
    bins = np.ceil(ranks / n * q).clip(1, q)
    return bins.astype(int)


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Add ``R``, ``F``, ``M`` quintile scores, ``RFM`` code and ``segment``.

    R is reversed so that the most recent customers score 5. Segments follow
    :data:`config.RFM_SEGMENT_RULES` on the R and F scores (first match wins).
    """
    out = rfm.copy()
    out["R"] = _quintile(out["recency"], reverse=True)
    out["F"] = _quintile(out["frequency"])
    out["M"] = _quintile(out["monetary"])
    out["RFM"] = out["R"].astype(str) + out["F"].astype(str) + out["M"].astype(str)
    out["segment"] = [assign_segment(r, f) for r, f in zip(out["R"], out["F"], strict=True)]
    return out


def assign_segment(r: int, f: int) -> str:
    """Return the named segment for recency score ``r`` and frequency score ``f``."""
    code = f"{r}{f}"
    for pattern, name in C.RFM_SEGMENT_RULES:
        if re.fullmatch(pattern, code):
            return name
    raise ValueError(f"No segment rule matches R={r}, F={f}.")  # pragma: no cover


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarise customers, revenue and mean R/F/M per segment, in display order.

    Returns:
        Columns ``segment, customers, customer_share, revenue, revenue_share,
        mean_recency, mean_frequency, mean_monetary, action``.
    """
    cols = [
        "segment",
        "customers",
        "customer_share",
        "revenue",
        "revenue_share",
        "mean_recency",
        "mean_frequency",
        "mean_monetary",
        "action",
    ]
    if scored.empty:
        return pd.DataFrame(columns=cols)
    g = scored.groupby("segment")
    out = pd.DataFrame(
        {
            "customers": g.size(),
            "revenue": g["monetary"].sum(),
            "mean_recency": g["recency"].mean(),
            "mean_frequency": g["frequency"].mean(),
            "mean_monetary": g["monetary"].mean(),
        }
    )
    out["customer_share"] = out["customers"] / out["customers"].sum()
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    order = [s for s in C.RFM_SEGMENT_ORDER if s in out.index]
    out = out.loc[order].rename_axis("segment").reset_index()
    out["action"] = out["segment"].map(C.RFM_SEGMENT_ACTIONS)
    return out[cols]


def rf_grid(scored: pd.DataFrame) -> pd.DataFrame:
    """Count customers on the R (rows, 5 at top) × F (columns) score grid."""
    grid = scored.groupby(["R", "F"]).size().unstack(fill_value=0)
    return grid.reindex(index=range(5, 0, -1), columns=range(1, 6), fill_value=0)


# --------------------------------------------------------------------------- #
# K-Means
# --------------------------------------------------------------------------- #


def make_pipeline(k: int) -> Pipeline:
    """Return the clustering pipeline: log1p → standardise → K-Means (seeded)."""
    return Pipeline(
        [
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=C.RANDOM_STATE)),
        ]
    )


def evaluate_k(rfm: pd.DataFrame, k_range: tuple[int, int] = C.KMEANS_K_RANGE) -> pd.DataFrame:
    """Fit K-Means for each k and report inertia, silhouette and Davies–Bouldin.

    Returns:
        Columns ``k, inertia, silhouette, davies_bouldin, smallest_cluster``.
        Values of k that are not smaller than the number of customers are skipped.
    """
    rows = []
    x = rfm[FEATURES].to_numpy(dtype=float)
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(rfm):
            break
        pipe = make_pipeline(k).fit(x)
        z = pipe[:-1].transform(x)
        labels = pipe["kmeans"].labels_
        rows.append(
            {
                "k": k,
                "inertia": float(pipe["kmeans"].inertia_),
                "silhouette": float(silhouette_score(z, labels)),
                "davies_bouldin": float(davies_bouldin_score(z, labels)),
                "smallest_cluster": int(np.bincount(labels).min()),
            }
        )
    return pd.DataFrame(
        rows, columns=["k", "inertia", "silhouette", "davies_bouldin", "smallest_cluster"]
    )


def kneedle_k(evaluation: pd.DataFrame) -> int | None:
    """Return the elbow of the inertia curve (Kneedle: max distance below the chord).

    Both axes are min-max normalised; the elbow is the k whose point lies
    furthest below the straight line joining the first and last points.
    """
    if len(evaluation) < 3:
        return None if evaluation.empty else int(evaluation["k"].iloc[0])
    x = evaluation["k"].to_numpy(dtype=float)
    y = evaluation["inertia"].to_numpy(dtype=float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
    distance = (1 - xn) - yn  # chord from (0, 1) to (1, 0) for a decreasing curve
    return int(x[int(np.argmax(distance))])


@dataclass
class ClusterResult:
    """A fitted clustering.

    Attributes:
        k: Number of clusters.
        labels: Cluster number per customer (aligned to the RFM index), ordered
            so cluster 1 has the highest mean monetary value.
        profile: Per-cluster summary (see :func:`cluster_profile`).
        pipeline: The fitted scikit-learn pipeline.
    """

    k: int
    labels: pd.Series
    profile: pd.DataFrame
    pipeline: Pipeline


def fit_clusters(rfm: pd.DataFrame, k: int) -> ClusterResult:
    """Fit the pipeline with ``k`` clusters and return ordered labels and profiles.

    Raises:
        ValueError: If there are not more customers than clusters.
    """
    if len(rfm) <= k:
        raise ValueError(f"Need more than {k} customers to form {k} clusters; got {len(rfm)}.")
    x = rfm[FEATURES].to_numpy(dtype=float)
    pipe = make_pipeline(k).fit(x)
    raw = pd.Series(pipe["kmeans"].labels_, index=rfm.index)
    order = rfm.groupby(raw)["monetary"].mean().sort_values(ascending=False).index
    mapping = {old: new for new, old in enumerate(order, start=1)}
    labels = raw.map(mapping).rename("cluster")
    return ClusterResult(k, labels, cluster_profile(rfm, labels), pipe)


def cluster_profile(rfm: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Summarise each cluster and give it a descriptive, data-derived name.

    Names are ``"C<n> — <value tier> · <recency tier>"``. Tiers compare the
    cluster's median monetary value and median recency with the tertiles of all
    customers (value: Low / Mid / High, or "Top" above the 90th percentile;
    recency: active / cooling / lapsed).
    The ``C<n>`` prefix keeps names unique; cluster 1 has the highest mean spend.

    Returns:
        Columns ``cluster, name, customers, customer_share, revenue,
        revenue_share, median_recency, median_frequency, median_monetary``
        plus ``z_recency, z_frequency, z_monetary`` (cluster mean of the
        standardised log features, for profile heatmaps).
    """
    logz = np.log1p(rfm[FEATURES])
    logz = (logz - logz.mean()) / logz.std(ddof=0)
    g = rfm.groupby(labels)
    out = pd.DataFrame(
        {
            "customers": g.size(),
            "revenue": g["monetary"].sum(),
            "median_recency": g["recency"].median(),
            "median_frequency": g["frequency"].median(),
            "median_monetary": g["monetary"].median(),
        }
    )
    for f in FEATURES:
        out[f"z_{f}"] = logz.groupby(labels)[f].mean()
    out["customer_share"] = out["customers"] / out["customers"].sum()
    out["revenue_share"] = out["revenue"] / out["revenue"].sum()
    m_lo, m_hi, m_top = rfm["monetary"].quantile([1 / 3, 2 / 3, 0.9])
    r_lo, r_hi = rfm["recency"].quantile([1 / 3, 2 / 3])

    def tier(value: float, lo: float, hi: float, labels: tuple[str, str, str]) -> str:
        return labels[0] if value <= lo else labels[1] if value <= hi else labels[2]

    def value_tier(m: float) -> str:
        return (
            "Top value"
            if m > m_top
            else tier(m, m_lo, m_hi, ("Low value", "Mid value", "High value"))
        )

    out["name"] = [
        f"C{c} — {value_tier(m)} · {tier(r, r_lo, r_hi, ('active', 'cooling', 'lapsed'))}"
        for c, m, r in zip(out.index, out["median_monetary"], out["median_recency"], strict=True)
    ]
    out = out.rename_axis("cluster").reset_index()
    return out[
        [
            "cluster",
            "name",
            "customers",
            "customer_share",
            "revenue",
            "revenue_share",
            "median_recency",
            "median_frequency",
            "median_monetary",
            "z_recency",
            "z_frequency",
            "z_monetary",
        ]
    ]


def segment_cluster_crosstab(scored: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Count customers by RFM segment (rows) and K-Means cluster (columns)."""
    tab = pd.crosstab(scored["segment"], labels.reindex(scored.index))
    order = [s for s in C.RFM_SEGMENT_ORDER if s in tab.index]
    return tab.loc[order]


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


@dataclass
class SegmentationResult:
    """Everything the Customer Segments page needs.

    Attributes:
        customers: One row per customer: R/F/M values and scores, segment,
            cluster and cluster name (monetary in the base currency).
        summary: :func:`segment_summary` output.
        evaluation: :func:`evaluate_k` output.
        recommended_k: Kneedle elbow (or ``None`` if not computable).
        clusters: The fitted clustering for the chosen k (or ``None``).
        snapshot: RFM reference date.
    """

    customers: pd.DataFrame
    summary: pd.DataFrame
    evaluation: pd.DataFrame
    recommended_k: int | None
    clusters: ClusterResult | None
    snapshot: pd.Timestamp | None


def segment_customers(lines: pd.DataFrame, k: int | None = None) -> SegmentationResult:
    """Run RFM scoring, k evaluation and clustering on completed-sales lines.

    Args:
        lines: Completed-sales lines in the base currency.
        k: Number of clusters; defaults to the recommended (elbow) k, capped at
            :data:`config.KMEANS_MAX_DISPLAY_K`.
    """
    rfm = compute_rfm(lines)
    if rfm.empty:
        return SegmentationResult(rfm, segment_summary(rfm), pd.DataFrame(), None, None, None)
    snap = snapshot_date(lines.loc[lines[C.COL_HAS_CUSTOMER]])
    scored = score_rfm(rfm)
    evaluation = evaluate_k(rfm)
    recommended = kneedle_k(evaluation)
    chosen = k or recommended
    clusters = None
    if chosen is not None and len(rfm) > chosen:
        chosen = min(chosen, C.KMEANS_MAX_DISPLAY_K)
        clusters = fit_clusters(rfm, chosen)
        names = clusters.profile.set_index("cluster")["name"]
        scored["cluster"] = clusters.labels
        scored["cluster_name"] = scored["cluster"].map(names)
    return SegmentationResult(
        scored, segment_summary(scored), evaluation, recommended, clusters, snap
    )


def with_display_currency(
    result: SegmentationResult, converted_lines: pd.DataFrame
) -> SegmentationResult:
    """Restate monetary outputs in the display currency without re-segmenting.

    Segments, clusters and names stay as computed in the base currency (so a
    currency switch never moves a customer). ``monetary`` in the customer
    table, the segment summary and the cluster profile are recomputed from
    ``converted_lines`` (the same completed-sales lines, converted).
    """
    if result.customers.empty:
        return result
    data = converted_lines.loc[
        converted_lines[C.COL_HAS_CUSTOMER] & (converted_lines[C.COL_REVENUE] > 0)
    ]
    display = data.groupby(C.COL_CUSTOMER)[C.COL_REVENUE].sum()
    customers = result.customers.copy()
    customers["monetary"] = display.reindex(customers.index).astype(float)
    clusters = result.clusters
    if clusters is not None:
        profile = cluster_profile(customers[FEATURES], clusters.labels)
        profile["name"] = clusters.profile["name"].to_numpy()
        clusters = ClusterResult(clusters.k, clusters.labels, profile, clusters.pipeline)
    return SegmentationResult(
        customers,
        segment_summary(customers),
        result.evaluation,
        result.recommended_k,
        clusters,
        result.snapshot,
    )


def save_customers(customers: pd.DataFrame, path=config.PROCESSED_CUSTOMERS) -> None:
    """Persist the per-customer segmentation table to ``data/processed``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(path, index=True)
