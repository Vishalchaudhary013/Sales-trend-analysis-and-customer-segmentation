from __future__ import annotations

import plotly.graph_objects as go

from src import analytics as a
from src import visuals as v


def _figs(clean):
    lines = a.apply_filters(clean, a.Filters())
    end, start = clean.InvoiceDate.max(), clean.InvoiceDate.min()
    ts = a.time_series(lines, "Monthly", end, start)
    return {
        "trend": v.fig_trend(ts, "revenue"),
        "trend_aov": v.fig_trend(a.time_series(lines, "Daily", end, start), "aov", "Daily"),
        "growth": v.fig_growth(ts),
        "heatmap": v.fig_weekday_hour(a.weekday_hour_matrix(lines)),
        "products": v.fig_top_products(a.top_products(lines, 5)),
        "products_units": v.fig_top_products(a.top_products(lines, 5, "units"), "units"),
        "countries": v.fig_countries(a.country_summary(lines)),
        "new_returning": v.fig_new_returning(a.new_vs_returning(lines, lines)),
        "cancellations": v.fig_cancellations(a.cancellation_trend(clean)),
    }


def test_all_figures_render_with_titles(clean):
    for name, fig in _figs(clean).items():
        assert isinstance(fig, go.Figure) and fig.data, name
        assert fig.layout.title.text, name


def test_partial_period_is_marked(clean):
    lines = a.apply_filters(clean, a.Filters())
    ts = a.time_series(lines, "Monthly", clean.InvoiceDate.max() - __import__("pandas").Timedelta(
        days=1), clean.InvoiceDate.min())  # fmt: skip
    fig = v.fig_trend(ts)
    assert any(t.name == "Partial period" and t.mode == "markers" for t in fig.data)


def test_new_customers_not_colour_only(clean):
    fig = _figs(clean)["new_returning"]
    new = next(t for t in fig.data if t.name.startswith("New"))
    assert new.marker.pattern.shape == "/"


def test_heatmap_no_trading_annotation():
    import numpy as np
    import pandas as pd

    m = pd.DataFrame(np.ones((7, 3)), index=list(a.WEEKDAYS), columns=[9, 10, 11])
    m.loc["Sat"] = np.nan
    fig = v.fig_weekday_hour(m)
    assert any("No trading" in ann.text for ann in fig.layout.annotations)


def test_empty_states(clean):
    empty = clean.iloc[0:0]
    for fig in (
        v.fig_trend(a.time_series(empty, "Monthly")),
        v.fig_growth(a.time_series(empty, "Monthly")),
        v.fig_weekday_hour(a.weekday_hour_matrix(empty)),
        v.fig_top_products(a.top_products(empty)),
        v.fig_countries(a.country_summary(empty)),
        v.fig_new_returning(a.new_vs_returning(empty, clean)),
        v.fig_cancellations(a.cancellation_trend(empty)),
    ):
        assert fig.layout.annotations and not fig.data


def test_money_format():
    assert v.money(1234.5) == "£1,234" or v.money(1234.5) == "£1,235"
    assert v.money(None) == "—"
