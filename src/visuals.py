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

METRIC_NAMES: dict[str, str] = {
    "revenue": "Revenue",
    "orders": "Orders (invoices)",
    "customers": "Active customers",
    "aov": "Average order value",
    "units": "Units sold",
}
MONEY_METRICS: frozenset[str] = frozenset({"revenue", "aov"})


def metric_label(metric: str, cur: str = CUR) -> str:
    """Axis/legend label for a metric, with the currency for monetary metrics."""
    name = METRIC_NAMES[metric]
    return f"{name} ({cur.strip()})" if metric in MONEY_METRICS else name


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


def money(value: float | None, decimals: int = 0, cur: str = CUR) -> str:
    """Format a currency amount (``£1,234``), or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{cur}{value:,.{decimals}f}"


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


def _hover_value(metric: str, cur: str = CUR) -> str:
    if metric in ("revenue", "aov"):
        return f"{cur}%{{y:,.2f}}"
    return "%{y:,}"


def fig_trend(
    ts: pd.DataFrame, metric: str = "revenue", freq: str = "Monthly", cur: str = CUR
) -> go.Figure:
    """Line chart of one metric over time, with partial periods visibly marked."""
    if ts.empty:
        return empty_figure("No sales in the current view.")
    label = metric_label(metric, cur)
    full = ts["partial"].eq(False)
    fig = go.Figure()
    # Solid line through full periods; dashed through the whole series underneath
    # so the partial ends stay connected but visibly different.
    fig.add_scatter(
        x=ts["period_start"],
        y=ts[metric],
        mode="lines",
        name="Partial period",
        line={"color": C.SINGLE_SERIES_COLOR, "width": 2, "dash": "dot"},
        hoverinfo="skip",
        showlegend=bool((~full).any()),
    )
    fig.add_scatter(
        x=ts["period_start"].where(full),
        y=ts[metric].where(full),
        mode="lines+markers",
        name=label,
        line={"color": C.SINGLE_SERIES_COLOR, "width": 2},
        marker={"size": 8 if freq != "Daily" else 4},
        customdata=ts[["period_label"]],
        hovertemplate=f"%{{customdata[0]}}<br>{label}: {_hover_value(metric, cur)}<extra></extra>",
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
            },
            customdata=part[["period_label"]],
            hovertemplate=f"%{{customdata[0]}} (partial)<br>{label}: {_hover_value(metric, cur)}"
            "<extra></extra>",
        )
    fig.update_yaxes(rangemode="tozero", tickprefix=cur if metric in ("revenue", "aov") else "")
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


def fig_weekday_hour(matrix: pd.DataFrame, cur: str = CUR) -> go.Figure:
    """Sequential heatmap of revenue by weekday x hour; no-trading cells in grey."""
    if matrix.empty:
        return empty_figure("No sales in the current view.")
    hours = [f"{h:02d}:00" for h in matrix.columns]
    z = matrix.to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_heatmap(  # grey background layer for "no trading"
        z=np.where(np.isnan(z), 1, np.nan),
        x=hours,
        y=matrix.index,
        colorscale=[[0, _NO_DATA], [1, _NO_DATA]],
        showscale=False,
        hoverinfo="skip",
    )
    fig.add_heatmap(
        z=z,
        x=hours,
        y=matrix.index,
        colorscale=C.SEQUENTIAL_SCALE,
        colorbar={"title": f"Revenue ({cur})", "tickprefix": cur},
        hovertemplate=f"%{{y}} %{{x}}<br>Revenue: {cur}%{{z:,.0f}}<extra></extra>",
        xgap=2,
        ygap=2,
    )
    for day in matrix.index[matrix.isna().all(axis=1)]:
        fig.add_annotation(
            x=hours[len(hours) // 2],
            y=day,
            text="No trading recorded",
            showarrow=False,
            font={"color": _INK, "size": 12},
        )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return apply_theme(fig, "Revenue by weekday and hour", "Hour of day", "")


def fig_top_products(top: pd.DataFrame, by: str = "revenue", cur: str = CUR) -> go.Figure:
    """Horizontal bars of top products (largest at the top) with value labels."""
    if top.empty:
        return empty_figure("No product data in the current view.")
    data = top.iloc[::-1]
    names = [
        f"{d if isinstance(d, str) else ''} ({s})"[:48]
        for d, s in zip(data[C.COL_DESC], data[C.COL_STOCK], strict=True)
    ]
    values = data["units"] if by == "units" else data["revenue"]
    text = [f"{v:,.0f}" if by == "units" else money(v, cur=cur) for v in values]
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
            ),
            hovertemplate=(
                f"%{{y}}<br>Revenue: {cur}%{{customdata[0]:,.2f}}<br>Units: %{{customdata[1]:,}}"
                "<br>Orders: %{customdata[2]:,}<br>Share of revenue: %{customdata[3]:.1%}"
                "<extra></extra>"
            ),
        )
    )
    x_title = "Units sold" if by == "units" else f"Revenue ({cur})"
    ranked = "units" if by == "units" else "revenue"
    apply_theme(fig, f"Top {len(data)} products by {ranked}", x_title, "")
    if by != "units":
        fig.update_xaxes(tickprefix=cur)
    fig.update_layout(
        height=max(380, 30 * len(data) + 120), margin={"l": 320, "r": 90, "t": 64, "b": 56}
    )
    return fig


def fig_countries(summary: pd.DataFrame, n: int = 15, cur: str = CUR) -> go.Figure:
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
                f"{money(r, cur=cur)} ({s:.1%})"
                for r, s in zip(data["revenue"], data["share"], strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["orders"], data["customers"], data["aov"]], axis=-1),
            hovertemplate=(
                f"%{{y}}<br>Revenue: {cur}%{{x:,.2f}}<br>Orders: %{{customdata[0]:,}}"
                f"<br>Customers: %{{customdata[1]:,}}<br>AOV: {cur}%{{customdata[2]:,.2f}}"
                "<extra></extra>"
            ),
        )
    )
    apply_theme(fig, f"Revenue by country (top {len(data)})", f"Revenue ({cur})", "")
    fig.update_xaxes(tickprefix=cur)
    fig.update_layout(
        height=max(380, 30 * len(data) + 120), margin={"l": 160, "r": 150, "t": 64, "b": 56}
    )
    return fig


def fig_new_returning(df: pd.DataFrame) -> go.Figure:
    """Stacked bars of new vs returning customers per month (new is hatched)."""
    if df.empty:
        return empty_figure("Customer IDs are needed for this chart.")
    labels = [
        f"{lbl}{' (partial)' if p else ''}{' *' if c else ''}"
        for lbl, p, c in zip(df["period_label"], df["partial"], df["censored"], strict=True)
    ]
    fig = go.Figure()
    fig.add_bar(
        x=labels,
        y=df["returning"],
        name="Returning",
        marker={"color": C.PALETTE["blue"], "line": {"color": "white", "width": 1}},
        hovertemplate="%{x}<br>Returning customers: %{y:,}<extra></extra>",
    )
    fig.add_bar(
        x=labels,
        y=df["new"],
        name="New (first purchase)",
        marker={
            "color": C.PALETTE["orange"],
            "pattern": {"shape": "/", "fgcolor": "white"},
            "line": {"color": "white", "width": 1},
        },
        hovertemplate="%{x}<br>New customers: %{y:,}<extra></extra>",
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    return apply_theme(fig, "New vs returning customers per month", "Month", "Customers")


def fig_cancellations(df: pd.DataFrame, cur: str = CUR) -> go.Figure:
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
                f"%{{customdata[1]:,}} invoices<br>Cancelled value: {cur}%{{customdata[2]:,.0f}}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_xaxes(type="category")
    return apply_theme(fig, "Cancellation rate per month", "Month", "Share of invoices cancelled")


def fig_donut(
    mix: pd.DataFrame,
    title: str,
    value_kind: str = "money",
    centre: str | None = None,
    cur: str = CUR,
    colors: list[str] | None = None,
    legend_below: bool = False,
) -> go.Figure:
    """Donut (circle) chart for a part-to-whole breakdown of at most six slices.

    Slices use the fixed categorical order (``"Other"`` is always grey), are
    separated by white gaps and carry direct "label + %" text, so identity
    never depends on colour alone.

    Args:
        mix: Output of an analytics breakdown (``label, value, share``).
        title: Chart title.
        value_kind: ``"money"`` or ``"count"`` (hover formatting).
        centre: Optional text in the hole (e.g. the total).
        cur: Currency prefix for monetary values.
        colors: Explicit slice colours (e.g. cluster colours, so colour follows
            the entity across charts); defaults to the fixed categorical order.
        legend_below: Place the legend under the chart (for narrow columns).
    """
    if mix.empty or mix["value"].sum() <= 0:
        return empty_figure("No data for this breakdown in the current view.")
    if len(mix) > C.MAX_PIE_SLICES:
        raise ValueError(f"A donut supports at most {C.MAX_PIE_SLICES} slices; got {len(mix)}.")
    explicit = colors
    colors, i = [], 0
    for label in mix["label"]:
        if explicit is not None:
            colors.append(explicit[len(colors)])
        elif label == "Other":
            colors.append(C.OTHER_COLOR)
        else:
            colors.append(C.CATEGORICAL_ORDER[i % len(C.CATEGORICAL_ORDER)])
            i += 1
    value_fmt = f"{cur}%{{value:,.0f}}" if value_kind == "money" else "%{value:,.0f}"
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
        height=420,
        showlegend=True,
        margin={"l": 24, "r": 24, "t": 64, "b": 24},
        uniformtext={"minsize": 11, "mode": "hide"},
        legend={"orientation": "v", "y": 0.5, "x": 1.02, "xanchor": "left", "yanchor": "middle"},
    )
    if legend_below:
        fig.update_layout(
            height=520,
            legend={
                "orientation": "v",
                "y": -0.05,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            margin={"l": 24, "r": 24, "t": 64, "b": 150},
        )
    return fig


def compact_money(value: float | None, cur: str = CUR) -> str:
    """Short currency label for tight spaces: ``£9.82M``, ``£436k``, ``£950``."""
    if value is None or pd.isna(value):
        return "—"
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= size:
            return f"{cur}{value / size:.3g}{suffix}"
    return f"{cur}{value:,.0f}"


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def cluster_style(cluster: int) -> tuple[str, str]:
    """Return the (colour, marker symbol) of a cluster number (1-based)."""
    i = (int(cluster) - 1) % len(C.CLUSTER_COLORS)
    return C.CLUSTER_COLORS[i], C.CLUSTER_SYMBOLS[i]


def fig_segment_bars(summary: pd.DataFrame) -> go.Figure:
    """Side-by-side bars: share of customers and share of revenue per RFM segment."""
    if summary.empty:
        return empty_figure("No identified customers in the current view.")
    from plotly.subplots import make_subplots

    data = summary.iloc[::-1]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=("Share of customers", "Share of revenue"),
    )
    for col, (field, count) in enumerate(
        (("customer_share", "customers"), ("revenue_share", "revenue")), start=1
    ):
        fig.add_bar(
            x=data[field],
            y=data["segment"],
            orientation="h",
            row=1,
            col=col,
            marker={"color": C.SINGLE_SERIES_COLOR},
            showlegend=False,
            cliponaxis=False,
            text=[f"{v:.0%}" for v in data[field]],
            textposition="outside",
            customdata=data[[count]],
            hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
        )
        fig.update_xaxes(
            tickformat=".0%", range=[0, max(0.05, data[field].max() * 1.25)], row=1, col=col
        )
    apply_theme(fig, "RFM segments: customers vs revenue")
    fig.update_layout(height=460, margin={"l": 150, "r": 40, "t": 90, "b": 40})
    return fig


def fig_rf_grid(grid: pd.DataFrame) -> go.Figure:
    """Heatmap of customer counts by Recency score (rows) × Frequency score (cols)."""
    if grid.empty or grid.to_numpy().sum() == 0:
        return empty_figure("No identified customers in the current view.")
    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(),
            x=[f"F{c}" for c in grid.columns],
            y=[f"R{r}" for r in grid.index],
            colorscale=C.SEQUENTIAL_SCALE,
            text=grid.to_numpy(),
            texttemplate="%{text:,}",
            xgap=2,
            ygap=2,
            colorbar={"title": "Customers"},
            hovertemplate="%{y} · %{x}<br>Customers: %{z:,}<extra></extra>",
        )
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    apply_theme(
        fig,
        "Customers by Recency and Frequency score",
        "Frequency score (5 = most orders)",
        "Recency score (5 = most recent)",
    )
    fig.update_layout(height=420)
    return fig


def fig_k_selection(
    evaluation: pd.DataFrame, recommended: int | None, chosen: int | None
) -> go.Figure:
    """Two panels (no dual axis): inertia (elbow) and silhouette score by k."""
    if evaluation.empty:
        return empty_figure("Not enough customers to evaluate clusterings.")
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=("Inertia (lower = tighter clusters)", "Silhouette (higher = better)"),
    )
    for col, field, fmt in ((1, "inertia", ",.0f"), (2, "silhouette", ".3f")):
        fig.add_scatter(
            x=evaluation["k"],
            y=evaluation[field],
            mode="lines+markers",
            row=1,
            col=col,
            line={"color": C.SINGLE_SERIES_COLOR, "width": 2},
            marker={"size": 8},
            showlegend=False,
            hovertemplate=f"k = %{{x}}<br>{field}: %{{y:{fmt}}}<extra></extra>",
        )
        if chosen in set(evaluation["k"]):
            row = evaluation.loc[evaluation["k"] == chosen].iloc[0]
            fig.add_scatter(
                x=[chosen],
                y=[row[field]],
                mode="markers",
                row=1,
                col=col,
                marker={
                    "size": 14,
                    "color": "white",
                    "line": {"color": C.PALETTE["vermillion"], "width": 3},
                },
                name=f"Selected k = {chosen}",
                showlegend=False,
                hovertemplate=f"Selected k = {chosen}<extra></extra>",
            )
    if recommended is not None:
        for col in (1, 2):
            fig.add_vline(
                x=recommended, line={"dash": "dash", "color": _MUTED, "width": 1}, row=1, col=col
            )
    title = "Choosing the number of clusters"
    if recommended is not None:
        title += f" — elbow at k = {recommended} (dashed line)"
    apply_theme(fig, title)
    fig.update_xaxes(dtick=1, title_text="Number of clusters (k)")
    fig.update_layout(height=420, margin={"t": 100}, showlegend=False)
    return fig


def fig_cluster_scatter(
    customers: pd.DataFrame, profile: pd.DataFrame, x: str = "recency", cur: str = CUR
) -> go.Figure:
    """Plot customers on ``x`` vs monetary (log scale), colour + symbol by cluster."""
    if customers.empty or "cluster" not in customers.columns:
        return empty_figure("No clusters to show.")
    names = profile.set_index("cluster")["name"]
    fig = go.Figure()
    for cl in sorted(customers["cluster"].dropna().unique()):
        part = customers[customers["cluster"] == cl]
        color, symbol = cluster_style(cl)
        fig.add_scattergl(
            x=part[x],
            y=part["monetary"],
            mode="markers",
            name=names.get(cl, f"C{cl}"),
            marker={
                "color": color,
                "symbol": symbol,
                "size": 6,
                "opacity": 0.6,
                "line": {"width": 0.5, "color": "white"},
            },
            customdata=np.stack(
                [part.index.astype(str), part["recency"], part["frequency"], part["segment"]],
                axis=-1,
            ),
            hovertemplate=(
                "Customer %{customdata[0]}<br>Recency: %{customdata[1]} days<br>"
                "Orders: %{customdata[2]}<br>"
                f"Spend: {cur}%{{y:,.0f}}<br>RFM segment: %{{customdata[3]}}"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    x_titles = {
        "recency": "Recency (days since last purchase)",
        "frequency": "Frequency (orders, log scale)",
    }
    fig.update_yaxes(type="log", tickprefix=cur, dtick=1)  # labels at powers of ten only
    if x == "frequency":
        fig.update_xaxes(type="log", dtick=1)
    apply_theme(
        fig,
        "Customers by cluster",
        x_titles.get(x, x),
        f"Monetary value ({cur.strip()}, log scale)",
    )
    fig.update_layout(
        height=520,
        legend={"orientation": "v", "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"},
    )
    return fig


def fig_cluster_profile(profile: pd.DataFrame) -> go.Figure:
    """Diverging heatmap of cluster means of the standardised log R, F, M (blue = better)."""
    if profile.empty:
        return empty_figure("No clusters to show.")
    # Recency is sign-flipped ("recentness") so that blue = better in every column.
    z = np.column_stack(
        [-profile["z_recency"], profile["z_frequency"], profile["z_monetary"]]
    ).astype(float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Recentness (higher = bought more recently)", "Frequency", "Monetary"],
            y=profile["name"],
            zmid=0,
            zmin=-2,
            zmax=2,
            colorscale=C.DIVERGING_SCALE,
            text=np.round(z, 2),
            texttemplate="%{text:+.2f}",
            xgap=2,
            ygap=2,
            colorbar={"title": "z-score"},
            hovertemplate="%{y}<br>%{x}: %{z:+.2f} SD from average<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    apply_theme(fig, "Cluster profiles (standard deviations from the average customer)")
    fig.update_layout(height=120 + 60 * len(profile), margin={"l": 240, "t": 110})
    return fig
