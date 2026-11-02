from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analytics as a
from src import config, currency, ingestion
from src import segmentation as s


def _lines(rows):
    df = pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "StockCode",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ],
    )
    df["Description"] = "ITEM"
    clean = ingestion.load_table(df, "t").data
    return clean[a.line_mask(clean, a.Filters())]


@pytest.fixture
def three():
    return _lines(
        [
            ("1", "A", 1, "2011-01-01 10:00", 10.0, "1", "UK"),
            ("2", "A", 2, "2011-03-01 10:00", 10.0, "1", "UK"),
            ("3", "A", 5, "2011-03-30 10:00", 10.0, "2", "France"),
            ("4", "A", 1, "2011-02-01 10:00", 7.0, None, "UK"),  # guest: excluded
            ("C5", "A", -1, "2011-03-31 10:00", 10.0, "3", "UK"),  # cancellation only
        ]
    )


def test_compute_rfm_values(three):
    rfm = s.compute_rfm(three)
    assert list(rfm.index) == ["1", "2"]
    snap = s.snapshot_date(three)
    assert snap == pd.Timestamp("2011-03-31")  # last day (30 Mar) + 1
    assert rfm.loc["1", "recency"] == 30 and rfm.loc["2", "recency"] == 1
    assert rfm.loc["1", "frequency"] == 2 and rfm.loc["1", "monetary"] == pytest.approx(30)
    assert rfm.loc["2", "country"] == "France"
    assert s.compute_rfm(three.iloc[0:0]).empty


def test_quintile_scoring_and_segments():
    rng = np.random.default_rng(0)
    rfm = pd.DataFrame(
        {
            "recency": rng.integers(0, 365, 500),
            "frequency": rng.integers(1, 4, 500),
            "monetary": rng.gamma(2, 300, 500),
        }
    )
    scored = s.score_rfm(rfm)
    for col in "RFM":
        assert scored[col].between(1, 5).all()
        assert scored[col].value_counts().max() <= 101  # rank-based: near-equal quintiles
    best = scored["recency"].idxmin()
    assert scored.loc[best, "R"] == 5  # most recent scores 5
    assert set(scored["segment"]) <= set(config.RFM_SEGMENT_ORDER)
    assert s.assign_segment(5, 5) == "Champions" and s.assign_segment(1, 1) == "Hibernating"
    assert s.assign_segment(2, 5) == "Can't lose them" and s.assign_segment(5, 1) == "New customers"
    summary = s.segment_summary(scored)
    assert summary["customers"].sum() == 500
    assert summary["customer_share"].sum() == pytest.approx(1)
    assert summary["revenue_share"].sum() == pytest.approx(1)
    assert list(summary["segment"]) == [
        x for x in config.RFM_SEGMENT_ORDER if x in set(scored["segment"])
    ]
    assert summary["action"].notna().all()
    grid = s.rf_grid(scored)
    assert grid.shape == (5, 5) and grid.to_numpy().sum() == 500
    assert s.segment_summary(scored.iloc[0:0]).empty
    assert s._quintile(pd.Series(dtype=float)).empty


def _blobs(seed=0):
    """Three well-separated synthetic customer groups."""
    rng = np.random.default_rng(seed)
    parts = []
    for r, f, m, n in ((5, 30, 8000, 60), (60, 5, 900, 120), (300, 1, 80, 150)):
        parts.append(
            pd.DataFrame(
                {
                    "recency": np.clip(rng.normal(r, r * 0.1 + 1, n), 0, None).round(),
                    "frequency": np.clip(rng.normal(f, f * 0.1 + 0.2, n), 1, None).round(),
                    "monetary": rng.normal(m, m * 0.1, n),
                }
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df.index = [f"c{i}" for i in range(len(df))]
    return df


def test_evaluate_k_and_kneedle():
    ev = s.evaluate_k(_blobs(), (2, 6))
    assert ev["k"].tolist() == [2, 3, 4, 5, 6]
    assert ev["inertia"].is_monotonic_decreasing
    assert ev.set_index("k")["silhouette"].idxmax() == 3
    assert s.kneedle_k(ev) == 3
    assert s.kneedle_k(ev.head(1)) == 2 and s.kneedle_k(ev.head(0)) is None
    tiny = _blobs().head(4)
    assert s.evaluate_k(tiny, (2, 10))["k"].max() == 3  # k < n customers


def test_fit_clusters_deterministic_and_ordered():
    rfm = _blobs()
    r1, r2 = s.fit_clusters(rfm, 3), s.fit_clusters(rfm, 3)
    assert (r1.labels == r2.labels).all()  # fixed seed
    prof = r1.profile
    assert prof["cluster"].tolist() == [1, 2, 3]
    assert prof["median_monetary"].is_monotonic_decreasing  # cluster 1 = highest spend
    assert prof["customers"].tolist() == [60, 120, 150]
    assert prof["name"].is_unique and prof["name"].iloc[0].startswith("C1 — Top value")
    assert prof["name"].iloc[-1].endswith("lapsed")
    with pytest.raises(ValueError):
        s.fit_clusters(rfm.head(3), 3)


def test_crosstab_and_end_to_end(clean):
    res = s.segment_customers(clean[a.line_mask(clean, a.Filters())], k=3)
    assert res.clusters.k == 3 and res.customers["cluster"].between(1, 3).all()
    tab = s.segment_cluster_crosstab(res.customers, res.clusters.labels)
    assert tab.to_numpy().sum() == len(res.customers)
    empty = s.segment_customers(clean.iloc[0:0])
    assert empty.customers.empty and empty.clusters is None


def test_display_currency_keeps_labels(clean):
    lines = clean[a.line_mask(clean, a.Filters())]
    res = s.segment_customers(lines, k=3)
    doubled = currency.convert(lines, currency.CurrencySetting("XXX", "custom", 2.0), None)
    shown = s.with_display_currency(res, doubled)
    assert (shown.customers["segment"] == res.customers["segment"]).all()
    assert (shown.customers["cluster"] == res.customers["cluster"]).all()
    assert shown.customers["monetary"].sum() == pytest.approx(2 * res.customers["monetary"].sum())
    assert (shown.clusters.profile["name"] == res.clusters.profile["name"]).all()
    assert shown.summary["revenue"].sum() == pytest.approx(2 * res.summary["revenue"].sum())


def test_save_customers(tmp_path, clean):
    res = s.segment_customers(clean[a.line_mask(clean, a.Filters())], k=2)
    path = tmp_path / "c.csv"
    s.save_customers(res.customers, path)
    assert len(pd.read_csv(path)) == len(res.customers)


@pytest.mark.skipif(not config.RAW_FILE.exists(), reason="dataset not downloaded")
def test_real_data_segmentation():
    data = ingestion.load_bundled_cached().data
    res = s.segment_customers(data[a.line_mask(data, a.Filters())])
    assert len(res.customers) == 4_324
    assert res.recommended_k == 5
    assert res.snapshot == pd.Timestamp("2011-12-10")
    assert res.evaluation.set_index("k")["silhouette"].idxmax() == 2
