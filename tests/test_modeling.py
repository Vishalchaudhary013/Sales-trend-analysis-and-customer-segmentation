from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, explain, ingestion, modeling, storage


def churn_transactions(n_customers: int = 80, seed: int = 0) -> pd.DataFrame:
    """Synthetic shop: half the customers stop buying ~150 days before the end (not real data)."""
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp("2011-01-03"), pd.Timestamp("2011-12-09")
    rows, inv = [], 600000
    for c in range(n_customers):
        churner = c % 2 == 0
        last = end - pd.Timedelta(days=150 if churner else 0)
        n_orders = int(rng.integers(2, 6) if churner else rng.integers(4, 12))
        days = np.sort(rng.integers(0, (last - start).days, n_orders))
        for d in days:
            inv += 1
            ts = start + pd.Timedelta(days=int(d), hours=10)
            for _ in range(int(rng.integers(1, 4))):
                rows.append(
                    (
                        str(inv),
                        str(rng.choice(["A1", "B2", "C3"])),
                        int(rng.integers(1, 10)),
                        f"{ts:%Y-%m-%d %H:%M}",
                        float(rng.choice([1.5, 3.0, 9.9])),
                        str(10000 + c),
                        "United Kingdom" if c % 3 else "France",
                    )
                )
    rows.append((str(inv + 1), "A1", 1, f"{end:%Y-%m-%d %H:%M}", 2.0, None, "United Kingdom"))
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
    return ingestion.load_table(df, "synthetic").data


@pytest.fixture(scope="module")
def data():
    return churn_transactions()


@pytest.fixture(scope="module")
def ds(data):
    return modeling.build_dataset(data)


def test_dataset_labels_and_no_leakage(data, ds):
    assert ds.cutoff == pd.Timestamp("2011-12-09") - pd.Timedelta(days=90)
    assert list(ds.X.columns) == list(config.CHURN_FEATURES_FULL)
    assert set(ds.y.unique()) == {0, 1}
    # churners (even ids) stopped 150 days before the end -> labelled churned
    even = ds.y[[int(c) % 2 == 0 for c in ds.y.index]]
    assert even.mean() == 1.0
    # features use only pre-cutoff data: recency is measured to the cut-off
    last_before = (
        data[(data.InvoiceDate < ds.cutoff) & data.HasCustomer]
        .groupby("CustomerID")["InvoiceDate"]
        .max()
    )
    expected = (ds.cutoff.normalize() - last_before.dt.normalize()).dt.days
    assert (ds.X["recency_days"] == expected.reindex(ds.X.index)).all()
    assert ds.X["cancellation_rate"].between(0, 1).all()
    assert 0 < ds.churn_rate < 1


def test_features_single_order_gap_is_nan(data):
    feats = modeling.build_features(data, pd.Timestamp("2011-01-20"))
    single = feats[feats["frequency"] == 1]
    assert single["mean_days_between_orders"].isna().all()
    assert modeling.build_features(data, pd.Timestamp("2010-01-01")).empty


def test_build_dataset_rejects_short_history(data):
    short = data[data.InvoiceDate < "2011-04-01"]
    with pytest.raises(ValueError, match="History spans"):
        modeling.build_dataset(short)


def test_feature_settings_and_models():
    assert modeling.feature_list("A") == list(config.CHURN_FEATURES_RFM)
    assert set(modeling.feature_list("A")) < set(modeling.feature_list("B"))
    with pytest.raises(ValueError):
        modeling.feature_list("C")
    with pytest.raises(ValueError):
        modeling.make_model("SVM")
    lr = modeling.make_model("Logistic Regression", "balanced")
    assert [s for s, _ in lr.steps] == ["impute", "log", "scale", "model"]
    assert lr["model"].class_weight == "balanced"
    assert modeling.make_model("Decision Tree")["model"].max_depth == 5
    assert modeling.make_model("Random Forest")["model"].n_estimators == 300


def test_folds_shared_and_stratified(ds):
    f1, f2 = modeling.make_folds(ds.y), modeling.make_folds(ds.y)
    assert len(f1) == 5
    for (_, a_te), (_, b_te) in zip(f1, f2, strict=True):
        assert (a_te == b_te).all()
    assert sorted(np.concatenate([te for _, te in f1]).tolist()) == list(range(len(ds.y)))


def test_cross_validation_and_baseline(ds):
    res = modeling.evaluate_all(ds)
    assert set(res) == {(s, m) for s in "AB" for m in (modeling.BASELINE, *config.MODEL_NAMES)}
    base = res[("B", modeling.BASELINE)]
    assert base.oof_proba.notna().all() and set(base.oof_proba.unique()) <= {0.0, 1.0}
    table = modeling.results_table(res, ds.y)
    b = table[(table.setting == "B") & (table.model == modeling.BASELINE)].iloc[0]
    assert b["roc_auc"] == 0.5
    lr = table[(table.setting == "B") & (table.model == "Logistic Regression")].iloc[0]
    assert lr["roc_auc"] > b["roc_auc"] + 0.15  # clearly better than the baseline
    assert {"accuracy_fold_std", "recall_fold_mean"} <= set(table.columns)
    assert modeling.best_model(table, "B") in config.MODEL_NAMES
    assert modeling.best_model(table.iloc[0:0], "B") == config.DEFAULT_MODEL


def test_metrics_threshold_roc_confusion():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = modeling.classification_metrics(y, p, 0.5)
    assert m == pytest.approx(
        {
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "roc_auc": 0.75,
            "pr_auc": m["pr_auc"],
            "retained_recall": 0.5,
        }
    )
    assert np.isnan(
        modeling.classification_metrics(np.array([1, 1]), np.array([0.2, 0.8]))["roc_auc"]
    )
    curve = modeling.threshold_curve(pd.Series(y), pd.Series(p))
    assert curve["recall"].is_monotonic_decreasing
    assert curve["flagged_share"].iloc[0] == 1.0
    cm = modeling.confusion(pd.Series(y), pd.Series(p), 0.5)
    assert cm.to_numpy().tolist() == [[1, 1], [1, 1]]
    assert len(modeling.roc_points(pd.Series(y), pd.Series(p))) >= 3


def test_train_save_load_score(tmp_path, data, ds):
    tm = modeling.train_final(ds, "B", "Logistic Regression", None, 0.5, {"roc_auc": 0.9}, "abc")
    meta = tm.metadata
    for key in (
        "model",
        "setting",
        "features",
        "params",
        "preprocessing",
        "validation",
        "cv_metrics",
        "trained_at",
        "cutoff",
        "threshold",
        "versions",
    ):
        assert key in meta
    assert meta["validation"]["random_state"] == 42 and meta["data_sha256"] == "abc"
    path = modeling.save_model(tm, tmp_path / "m.joblib")
    assert path.with_suffix(".json").exists()
    back = modeling.load_model(path)
    assert back is not None and back.features == tm.features
    assert modeling.load_model(tmp_path / "missing.joblib") is None
    (tmp_path / "bad.joblib").write_text("not a model")
    assert modeling.load_model(tmp_path / "bad.joblib") is None
    scored = modeling.score_current(back, data)
    assert scored["churn_probability"].between(0, 1).all()
    assert scored["churn_probability"].is_monotonic_decreasing
    assert set(scored["risk_tier"]) <= {"High", "Medium", "Low"}
    imp = modeling.feature_importance(back.pipeline, back.features)
    assert imp["value"].abs().is_monotonic_decreasing and imp["label"].notna().all()
    rf = modeling.train_final(ds, "A", "Random Forest", "balanced", 0.5)
    assert (modeling.feature_importance(rf.pipeline, rf.features)["value"] >= 0).all()


@pytest.mark.parametrize(
    ("p", "tier"),
    [(0.95, "High"), (0.7, "High"), (0.69, "Medium"), (0.4, "Medium"), (0.39, "Low"), (0.0, "Low")],
)
def test_risk_tiers(p, tier):
    assert modeling.risk_tier(p) == tier


def test_reason_codes(data, ds):
    tm = modeling.train_final(ds, "B", "Logistic Regression", None, 0.5)
    scored = modeling.score_current(tm, data)
    contrib = explain.contributions(tm, scored.head(10))
    assert list(contrib.columns) == tm.features and contrib.notna().all().all()
    rc = explain.reason_codes(tm, scored.head(10), cur="$", monetary_factor=2.0)
    assert list(rc.columns) == [
        "reason_1",
        "contribution_1",
        "reason_2",
        "contribution_2",
        "reason_3",
        "contribution_3",
    ]
    top = rc.iloc[0]
    assert top["reason_1"] and top["contribution_1"] >= top["contribution_2"] >= 0
    assert "£" not in " ".join(rc["reason_1"])


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("recency_days", 200, "longer ago than"),
        ("recency_days", 1, "more recent than"),
        ("frequency", 1, "1 order in total"),
        ("frequency", 90, "more than"),
        ("orders_last_90d", 0, "No orders in the last 90 days"),
        ("mean_days_between_orders", float("nan"), "Only one order"),
        ("is_uk", 0, "outside the UK"),
        ("monetary", 10_000, "over 99%"),
    ],
)
def test_phrases_follow_direction(feature, value, expected):
    ref = pd.Series(np.arange(1, 101, dtype=float))
    assert expected in explain._phrase(feature, value, ref, "£")


def test_storage_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    uid = storage.record_upload("a.csv", 10, "f" * 64, True, "custom", 0, 2, "x", db_path=db)
    mid = storage.record_model("Random Forest", "B", "m.joblib", {"k": 1}, 0.7, db_path=db)
    rid = storage.record_prediction_run(
        "Random Forest",
        "B",
        0.5,
        100,
        40,
        {"High": 10, "Medium": 30, "Low": 60},
        "abc",
        "GBP",
        db_path=db,
    )
    assert (uid, mid, rid) == (1, 1, 1)
    up = storage.upload_history(db_path=db)
    assert up.loc[0, "filename"] == "a.csv" and up.loc[0, "valid"] == 1
    assert storage.model_history(db_path=db).loc[0, "roc_auc"] == 0.7
    run = storage.prediction_history(db_path=db).iloc[0]
    assert (run["n_high"], run["n_medium"], run["n_low"]) == (10, 30, 60)
    storage.record_upload("b.csv", 5, "e" * 64, False, None, db_path=db)
    assert storage.upload_history(db_path=db)["filename"].tolist() == ["b.csv", "a.csv"]


@pytest.mark.skipif(not config.RAW_FILE.exists(), reason="dataset not downloaded")
def test_real_dataset_reference():
    real = modeling.build_dataset(ingestion.load_bundled_cached().data)
    assert len(real.y) == 3_364
    assert real.churn_rate == pytest.approx(0.43, abs=0.005)


def test_model_status_detects_version_mismatch(tmp_path, ds):
    import json

    tm = modeling.train_final(ds, "A", "Logistic Regression", None, 0.5)
    path = modeling.save_model(tm, tmp_path / "m.joblib")
    assert modeling.model_status(path) == ("ok", "")
    meta = json.loads(path.with_suffix(".json").read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    path.with_suffix(".json").write_text(json.dumps(meta))
    status, reason = modeling.model_status(path)
    assert status == "incompatible" and "0.0.1" in reason
    assert modeling.load_model(path) is None
    path.with_suffix(".json").unlink()
    assert modeling.model_status(path)[0] == "incompatible"
    assert modeling.model_status(tmp_path / "none.joblib")[0] == "missing"
