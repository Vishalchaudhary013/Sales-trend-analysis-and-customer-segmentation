from __future__ import annotations

import pytest

from scripts import export_figures
from src import ingestion, static_figures
from tests.test_modeling import churn_transactions


@pytest.fixture(scope="module")
def inputs():
    return static_figures.prepare_inputs(churn_transactions(60, seed=2))


def test_export_all_figures(tmp_path, inputs):
    paths = static_figures.export_all(inputs, tmp_path, dpi=40)
    assert [p.stem for p in paths] == list(static_figures.FIGURES)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)
    assert all(desc for _, desc in static_figures.FIGURES.values())


def test_export_without_models_uses_placeholders(tmp_path):
    data = churn_transactions(30, seed=3)
    inp = static_figures.prepare_inputs(data, with_models=False)
    assert inp.cv is None and inp.ds is None
    assert len(static_figures.export_all(inp, tmp_path, dpi=30)) == len(static_figures.FIGURES)


def test_short_history_skips_models_gracefully(tmp_path):
    data = churn_transactions(30, seed=4)
    short = data[data.InvoiceDate < "2011-03-01"]
    inp = static_figures.prepare_inputs(short)
    assert inp.cv is None
    static_figures.export_all(inp, tmp_path, dpi=30)


def test_empty_view_renders_placeholders(tmp_path, inputs):
    empty = static_figures.ReportInputs(
        inputs.lines.iloc[0:0],
        inputs.scope.iloc[0:0],
        inputs.data_start,
        inputs.data_end,
        static_figures.segmentation.segment_customers(inputs.lines.iloc[0:0]),
        None,
        None,
    )
    assert len(static_figures.export_all(empty, tmp_path, dpi=30)) == len(static_figures.FIGURES)


def test_export_script(tmp_path, monkeypatch, capsys):
    data = churn_transactions(40, seed=5)
    monkeypatch.setattr(
        ingestion,
        "load_bundled_cached",
        lambda: ingestion.LoadResult(data, ingestion.ValidationReport("s", 1, 1)),
    )
    assert export_figures.main(["--out", str(tmp_path), "--dpi", "30", "--no-models"]) == 0
    assert "Exported 13 figures" in capsys.readouterr().out
    monkeypatch.setattr(
        ingestion,
        "load_bundled_cached",
        lambda: ingestion.LoadResult(None, ingestion.ValidationReport("s", 0, 0)),
    )
    assert export_figures.main(["--out", str(tmp_path)]) == 1
