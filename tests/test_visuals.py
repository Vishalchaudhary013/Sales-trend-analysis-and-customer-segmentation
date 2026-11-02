from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src import ingestion, visuals
from tests.conftest import make_students


@pytest.fixture
def prepared() -> pd.DataFrame:
    df = ingestion.merge_subjects(
        {"math": make_students(30, 1), "portuguese": make_students(50, 2)}
    )
    return ingestion.prepare(df)


def test_build_figures_all_slugs(prepared):
    figs = visuals.build_figures(prepared)
    assert set(figs) == set(visuals.FIGURE_TITLES) == set(visuals.STATIC_BUILDERS)
    assert set(visuals.FIGURE_DESCRIPTIONS) == set(visuals.FIGURE_TITLES)
    for fig in figs.values():
        assert isinstance(fig, go.Figure) and fig.data


def test_fail_is_not_colour_only(prepared):
    fig = visuals.build_figures(prepared)["g3_distribution"]
    fail = next(t for t in fig.data if t.name == "Fail")
    assert fail.marker.pattern.shape == "/"
    scatter = visuals.build_figures(prepared)["g2_vs_g3"]
    symbols = {t.name: t.marker.symbol for t in scatter.data if t.name in ("Pass", "Fail")}
    assert symbols["Pass"] != symbols["Fail"]


def test_axes_are_labelled(prepared):
    for slug, fig in visuals.build_figures(prepared).items():
        if slug == "correlation_heatmap":
            continue
        assert fig.layout.xaxis.title.text or fig.layout.yaxis.title.text, slug


def test_empty_states():
    empty = pd.DataFrame()
    figs = visuals.build_figures(ingestion.prepare(make_students(5)).iloc[0:0])
    assert all(isinstance(f, go.Figure) for f in figs.values())
    assert visuals.fig_g3_box(empty, "higher", "g3_by_higher").layout.annotations


def test_export_static_figures(tmp_path, prepared):
    paths = visuals.export_static_figures(prepared, tmp_path, dpi=50)
    assert [p.stem for p in paths] == list(visuals.STATIC_BUILDERS)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_export_static_empty(tmp_path, prepared):
    paths = visuals.export_static_figures(prepared.iloc[0:0], tmp_path, dpi=40)
    assert len(paths) == len(visuals.STATIC_BUILDERS)
