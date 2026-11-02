from __future__ import annotations

import json

import pytest

from scripts import train
from src import config, ingestion
from tests.test_modeling import churn_transactions


def test_train_cli_writes_reports_and_model(tmp_path, monkeypatch):
    data = churn_transactions(60, seed=1)
    monkeypatch.setattr(
        ingestion,
        "load_bundled_cached",
        lambda: ingestion.LoadResult(data, ingestion.ValidationReport("s", 1, 1), None, "sha"),
    )
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(config, "MODEL_FILE", tmp_path / "models" / "m.joblib")
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    from src import storage

    monkeypatch.setattr(storage, "record_model", lambda *a, **k: 1)
    assert train.main(["--class-weight", "balanced", "--threshold", "0.4"]) == 0
    payload = json.loads((tmp_path / "metrics" / "metrics.json").read_text())
    assert len(payload["results"]) == 8  # 2 settings x (3 models + baseline)
    assert payload["context"]["class_weight"] == "balanced"
    assert (tmp_path / "models" / "m.joblib").exists()
    md = (tmp_path / "metrics" / "metrics.md").read_text()
    assert "Setting A" in md and "Baseline (majority class)" in md


def test_train_cli_rejects_bad_threshold():
    with pytest.raises(SystemExit):
        train.main(["--threshold", "1.5"])
