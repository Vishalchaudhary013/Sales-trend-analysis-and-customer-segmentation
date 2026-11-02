"""Plotly figure factories for the sales dashboard.

Every function is pure: it takes an aggregate from :mod:`src.analytics` and
returns a :class:`plotly.graph_objects.Figure`. One colour system (Okabe-Ito
based, validated for colour-vision deficiency) is used throughout; meaning is
never carried by colour alone (labels, hatching, marker styles, annotations).
Static PNG exports are added in a later milestone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config

C = config
_INK = "#1A1A1A"
_MUTED = "#5F6368"
_GRID = "#E6E8EB"
_NO_DATA = "#D9DCE0"
CUR = config.CURRENCY_SYMBOL

METRIC_LABELS: dict[str, str] = {
    "revenue": f"Revenue ({CUR})",
    "orders": "Orders (invoices)",
    "customers": "Active customers",
    "aov": f"Average order value ({CUR})",
    "units": "Units sold",
}
FIGURE_DESCRIPTIONS: dict[str, str] = {
    "trend": "Selected metric per period. Hollow markers and a dashed segment mark partial "
    "periods at the edges of the data, which are not comparable with full periods.",
    "growth": "Change in revenue versus the previous period. Growth involving a partial "
    "period is not shown.",
    "heatmap": "Revenue by weekday and hour of invoice. Grey cells mean no trading at all "
    "(e.g. the shop recorded no Saturday orders).",
    "products": "Top products in the current view. Share = share of total revenue.",
    "countries": "Revenue by customer country (top 15).",
    "new_returning": "Identified customers active each month, split by whether it was their "
    "first purchase month. * The first month of data counts everyone as new (no earlier "
    "history).",
    "cancellations": "Share of invoices that are cancellations, per month.",
    "mix_country": "Share of revenue by customer country (top 5, the rest grouped as Other).",
    "mix_customer": "Share of revenue from identified customers versus guest checkouts.",
    "mix_invoices": "Share of invoices that were completed versus cancelled.",
}


def money(value: float | None, decimals: int = 0) -> str:
    """Format a currency amount (``£1,234``), or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{CUR}{value:,.{decimals}f}"


def apply_theme(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply the shared layout (fonts, grid, margins, hover, white background)."""
    fig.update_layout(
        template=config.PLOT_TEMPLATE,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 17}},
        font={"family": config.FONT_FAMILY, "size": config.BASE_FONT_SIZE, "color": _INK},
        margin={"l": 64, "r": 24, "t": 64, "b": 56},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.2,
    )
    fig.update_xaxes(title_text=x_title, gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, gridcolor=_GRID, zeroline=False)
    return fig


def empty_figure(message: str) -> go.Figure:
    """Return a blank figure carrying an explanatory message (empty state)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 15, "color": _MUTED})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return apply_theme(fig, "")


def _hover_value(metric: str) -> str:
    if metric in ("revenue", "aov"):
        return f"{CUR}%{{y:,.2f}}"
    return "%{y:,}"


def fig_trend(ts: pd.DataFrame, metric: str = "revenue", freq: str = "Monthly") -> go.Figure:
    """Line chart of one metric over time, with partial periods visibly marked."""
    if ts.empty:
        return empty_figure("No sales in the current view.")
    label = METRIC_LABELS[metric]
    full = ts["partial"].eq(False)
    fig = go.Figure()
    # Solid line through full periods; dashed through the whole series underneath
    # so the partial ends stay connected but visibly different.
    fig.add_scatter(x=ts["period_start"], y=ts[metric], mode="lines", name="Partial period",
                    line={"color": C.SINGLE_SERIES_COLOR, "width": 2, "dash": "dot"},
                    hoverinfo="skip", showlegend=bool((~full).any()))  # fmt: skip
    fig.add_scatter(
        x=ts["period_start"].where(full),
        y=ts[metric].where(full),
        mode="lines+markers",
        name=label,
        line={"color": C.SINGLE_SERIES_COLOR, "width": 2},
        marker={"size": 8 if freq != "Daily" else 4},
        customdata=ts[["period_label"]],
        hovertemplate=f"%{{customdata[0]}}<br>{label}: {_hover_value(metric)}<extra></extra>",
        connectgaps=False,
    )
    part = ts[~full]
    if len(part):
        fig.add_scatter(
            x=part["period_start"],
            y=part[metric],
            mode="markers",
            name="Partial period",
            showlegend=False,
            marker={
                "size": 10,
                "color": "white",
                "line": {"color": C.SINGLE_SERIES_COLOR, "width": 2},
            },  # fmt: skip
            customdata=part[["period_label"]],
            hovertemplate=f"%{{customdata[0]}} (partial)<br>{label}: {_hover_value(metric)}"
            "<extra></extra>",
        )
    fig.update_yaxes(rangemode="tozero", tickprefix=CUR if metric in ("revenue", "aov") else "")
    return apply_theme(fig, f"{label} — {freq.lower()}", "Period", label)


def fig_growth(ts: pd.DataFrame, freq: str = "Monthly") -> go.Figure:
    """Bar chart of period-on-period revenue growth; sign shown by colour and label."""
    data = ts.dropna(subset=["growth"]) if not ts.empty else ts
    if data.empty:
        return empty_figure("At least two complete periods are needed to show growth.")
    colors = np.where(data["growth"] >= 0, C.PALETTE["blue"], C.PALETTE["vermillion"])
    fig = go.Figure(
        go.Bar(
            x=data["period_label"],
            y=data["growth"],
            marker={"color": colors},
            text=[f"{g:+.0%}" for g in data["growth"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}<br>Revenue growth: %{y:+.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": _MUTED, "width": 1})
    fig.update_yaxes(tickformat="+.0%")
    fig.update_xaxes(type="category")
    unit = {"Daily": "day", "Weekly": "week", "Monthly": "month"}.get(freq, "period")
    return apply_theme(fig, f"Revenue growth vs previous {unit}", "Period", "Change in revenue")


def fig_weekday_hour(matrix: pd.DataFrame) -> go.Figure:
    """Sequential heatmap of revenue by weekday x hour; no-trading cells in grey."""
    if matrix.empty:
        return empty_figure("No sales in the current view.")
    hours = [f"{h:02d}:00" for h in matrix.columns]
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_heatmap(  # grey background layer for "no trading"
        z=np.where(np.isnan(z), 1, np.nan), x=hours, y=matrix.index,
        colorscale=[[0, _NO_DATA], [1, _NO_DATA]], showscale=False, hoverinfo="skip",
    )  # fmt: skip
    fig.add_heatmap(
        z=z,
        x=hours,
        y=matrix.index,
        colorscale=C.SEQUENTIAL_SCALE,
        colorbar={"title": f"Revenue ({CUR})", "tickprefix": CUR},
        hovertemplate=f"%{{y}} %{{x}}<br>Revenue: {CUR}%{{z:,.0f}}<extra></extra>",
        xgap=2,
        ygap=2,
    )
    for day in matrix.index[matrix.isna().all(axis=1)]:
        fig.add_annotation(x=hours[len(hours) // 2], y=day, text="No trading recorded",
                           showarrow=False, font={"color": _INK, "size": 12})  # fmt: skip
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return apply_theme(fig, "Revenue by weekday and hour", "Hour of day", "")


def fig_top_products(top: pd.DataFrame, by: str = "revenue") -> go.Figure:
    """Horizontal bars of top products (largest at the top) with value labels."""
    if top.empty:
        return empty_figure("No product data in the current view.")
    data = top.iloc[::-1]
    names = [f"{d if isinstance(d, str) else ''} ({s})"[:48]
             for d, s in zip(data[C.COL_DESC], data[C.COL_STOCK], strict=True)]  # fmt: skip
    values = data["units"] if by == "units" else data["revenue"]
    text = [f"{v:,.0f}" if by == "units" else money(v) for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": C.SINGLE_SERIES_COLOR},
            text=text,
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack(
                [data["revenue"], data["units"], data["orders"], data["share"]], axis=-1
            ),  # fmt: skip
            hovertemplate=(
                f"%{{y}}<br>Revenue: {CUR}%{{customdata[0]:,.2f}}<br>Units: %{{customdata[1]:,}}"
                "<br>Orders: %{customdata[2]:,}<br>Share of revenue: %{customdata[3]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    x_title = "Units sold" if by == "units" else f"Revenue ({CUR})"
    ranked = "units" if by == "units" else "revenue"
    apply_theme(fig, f"Top {len(data)} products by {ranked}", x_title, "")
    if by != "units":
        fig.update_xaxes(tickprefix=CUR)
    fig.update_layout(height=max(380, 30 * len(data) + 120),
                      margin={"l": 320, "r": 90, "t": 64, "b": 56})  # fmt: skip
    return fig


def fig_countries(summary: pd.DataFrame, n: int = 15) -> go.Figure:
    """Horizontal bars of revenue by country (top ``n``), share in labels."""
    if summary.empty:
        return empty_figure("No country data in the current view.")
    data = summary.head(n).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["revenue"],
            y=data[C.COL_COUNTRY],
            orientation="h",
            marker={"color": C.SINGLE_SERIES_COLOR},
            text=[
                f"{money(r)} ({s:.1%})" for r, s in zip(data["revenue"], data["share"], strict=True)
            ],  # fmt: skip
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["orders"], data["customers"], data["aov"]], axis=-1),
            hovertemplate=(
                f"%{{y}}<br>Revenue: {CUR}%{{x:,.2f}}<br>Orders: %{{customdata[0]:,}}"
                f"<br>Customers: %{{customdata[1]:,}}<br>AOV: {CUR}%{{customdata[2]:,.2f}}"
                "<extra></extra>"
            ),
        )
    )
    apply_theme(fig, f"Revenue by country (top {len(data)})", f"Revenue ({CUR})", "")
    fig.update_xaxes(tickprefix=CUR)
    fig.update_layout(height=max(380, 30 * len(data) + 120),
                      margin={"l": 160, "r": 150, "t": 64, "b": 56})  # fmt: skip
    return fig


def fig_new_returning(df: pd.DataFrame) -> go.Figure:
    """Stacked bars of new vs returning customers per month (new is hatched)."""
    if df.empty:
        return empty_figure("Customer IDs are needed for this chart.")
    labels = [f"{lbl}{' (partial)' if p else ''}{' *' if c else ''}"
              for lbl, p, c in zip(df["period_label"], df["partial"], df["censored"],
                                   strict=True)]  # fmt: skip
    fig = go.Figure()
    fig.add_bar(x=labels, y=df["returning"], name="Returning",
                marker={"color": C.PALETTE["blue"], "line": {"color": "white", "width": 1}},
                hovertemplate="%{x}<br>Returning customers: %{y:,}<extra></extra>")  # fmt: skip
    fig.add_bar(x=labels, y=df["new"], name="New (first purchase)",
                marker={"color": C.PALETTE["orange"], "pattern": {"shape": "/", "fgcolor": "white"},
                        "line": {"color": "white", "width": 1}},
                hovertemplate="%{x}<br>New customers: %{y:,}<extra></extra>")  # fmt: skip
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    return apply_theme(fig, "New vs returning customers per month", "Month", "Customers")


def fig_cancellations(df: pd.DataFrame) -> go.Figure:
    """Line of monthly cancellation rate (share of invoices) with value labels."""
    if df.empty:
        return empty_figure("No invoices in the current view.")
    fig = go.Figure(
        go.Scatter(
            x=df["period_label"],
            y=df["rate"],
            mode="lines+markers+text",
            line={"color": C.PALETTE["vermillion"], "width": 2},
            marker={"size": 8},
            text=[f"{r:.0%}" for r in df["rate"]],
            textposition="top center",
            customdata=np.stack([df["cancelled"], df["invoices"], df["cancelled_value"]], axis=-1),
            hovertemplate=(
                "%{x}<br>Cancellation rate: %{y:.1%}<br>%{customdata[0]:,} of "
                f"%{{customdata[1]:,}} invoices<br>Cancelled value: {CUR}%{{customdata[2]:,.0f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(type="category")
    return apply_theme(fig, "Cancellation rate per month", "Month", "Share of invoices cancelled")


def fig_donut(mix: pd.DataFrame, title: str, value_kind: str = "money",
              centre: str | None = None) -> go.Figure:  # fmt: skip
    """Donut (circle) chart for a part-to-whole breakdown of at most six slices.

    Slices use the fixed categorical order (``"Other"`` is always grey), are
    separated by white gaps and carry direct "label + %" text, so identity
    never depends on colour alone.

    Args:
        mix: Output of an analytics breakdown (``label, value, share``).
        title: Chart title.
        value_kind: ``"money"`` or ``"count"`` (hover formatting).
        centre: Optional text in the hole (e.g. the total).
    """
    if mix.empty or mix["value"].sum() <= 0:
        return empty_figure("No data for this breakdown in the current view.")
    if len(mix) > C.MAX_PIE_SLICES:
        raise ValueError(f"A donut supports at most {C.MAX_PIE_SLICES} slices; got {len(mix)}.")
    colors, i = [], 0
    for label in mix["label"]:
        if label == "Other":
            colors.append(C.OTHER_COLOR)
        else:
            colors.append(C.CATEGORICAL_ORDER[i % len(C.CATEGORICAL_ORDER)])
            i += 1
    value_fmt = f"{CUR}%{{value:,.0f}}" if value_kind == "money" else "%{value:,.0f}"
    # Legend entries carry the share so every slice is identified in text.
    labels = [f"{lbl} · {sh:.1%}" for lbl, sh in zip(mix["label"], mix["share"], strict=True)]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=mix["value"],
            hole=0.55,
            sort=False,
            direction="clockwise",
            marker={"colors": colors, "line": {"color": "white", "width": 2}},
            texttemplate="%{percent:.0%}",
            textposition="inside",
            insidetextorientation="horizontal",
            textfont={"size": 13, "color": "white"},
            hovertemplate=f"%{{label}}<br>{value_fmt}<br>%{{percent:.1%}}<extra></extra>",
        )
    )
    if centre:
        fig.add_annotation(text=centre, showarrow=False, font={"size": 15, "color": _INK})
    apply_theme(fig, title)
    # Slices too small for their label show no text (rather than overlapping);
    # the legend and hover still identify them.
    fig.update_layout(
        height=420, showlegend=True, margin={"l": 24, "r": 24, "t": 64, "b": 24},
        uniformtext={"minsize": 11, "mode": "hide"},
        legend={"orientation": "v", "y": 0.5, "x": 1.02, "xanchor": "left", "yanchor": "middle"},
    )  # fmt: skip
    return fig


def compact_money(value: float | None) -> str:
    """Short currency label for tight spaces: ``£9.82M``, ``£436k``, ``£950``."""
    if value is None or pd.isna(value):
        return "—"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            return f"{CUR}{value / size:.3g}{suffix}"
    return f"{CUR}{value:,.0f}"
