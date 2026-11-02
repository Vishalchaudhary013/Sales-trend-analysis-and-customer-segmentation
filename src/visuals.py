"""Figure factories: interactive Plotly figures and static Matplotlib exports.

Every function is pure: it receives data (usually an aggregate produced by
:mod:`src.analytics`) and returns a figure. Colour roles come from
:mod:`src.config` (Okabe-Ito based, colour-blind safe). Pass/Fail is always
encoded redundantly - by colour *and* by hatch pattern / marker symbol / label.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src import analytics, config  # noqa: E402

FIGURE_TITLES: dict[str, str] = {
    "g3_distribution": "Distribution of final grade (G3) by outcome",
    "mean_g3_by_studytime": "Mean final grade by weekly study time",
    "mean_g3_by_failures": "Mean final grade by number of past failures",
    "correlation_with_g3": "Correlation of numeric attributes with G3",
    "g2_vs_g3": "Second-period grade (G2) vs final grade (G3)",
    "g3_by_subject": "Final grade by subject",
    "g3_by_higher": "Final grade by higher-education aspiration",
    "pass_rate_by_absence_band": "Pass rate by absence band",
    "correlation_heatmap": "Correlation matrix of numeric attributes",
}
FIGURE_DESCRIPTIONS: dict[str, str] = {
    "g3_distribution": "Count of records at each G3 value; bars below 10 are fails (hatched).",
    "mean_g3_by_studytime": "Mean G3 with 95% CI per study-time band; n per group shown.",
    "mean_g3_by_failures": "Mean G3 with 95% CI by past class failures; n per group shown.",
    "correlation_with_g3": "Pearson r of each numeric attribute with G3, sorted.",
    "g2_vs_g3": "Grade pairs; marker area = number of records, symbol = outcome.",
    "g3_by_subject": "Box plots of G3 for mathematics and Portuguese.",
    "g3_by_higher": "Box plots of G3 by whether the student wants higher education.",
    "pass_rate_by_absence_band": "Share passing per absence band with 95% Wilson CI and n.",
    "correlation_heatmap": "Pairwise Pearson correlations among numeric attributes and G3.",
}

_INK = "#1A1A1A"
_MUTED = "#5F6368"
_GRID = "#E6E8EB"


# --------------------------------------------------------------------------- #
# Plotly helpers
# --------------------------------------------------------------------------- #


def _label(col: str) -> str:
    return config.AXIS_LABELS.get(col, col)


def _value_label(col: str, value: object) -> str:
    return config.VALUE_LABELS.get(col, {}).get(value, str(value))  # type: ignore[arg-type]


def apply_theme(fig: go.Figure, title: str, x_title: str = "", y_title: str = "") -> go.Figure:
    """Apply the shared layout (fonts, grid, margins, hover) to ``fig``."""
    fig.update_layout(
        template=config.PLOT_TEMPLATE,
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 17}},
        font={"family": config.FONT_FAMILY, "size": config.BASE_FONT_SIZE, "color": _INK},
        margin={"l": 60, "r": 24, "t": 64, "b": 56},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
        bargap=0.15,
        plot_bgcolor="white",
        paper_bgcolor="white",
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


# --------------------------------------------------------------------------- #
# Plotly figures
# --------------------------------------------------------------------------- #


def fig_g3_distribution(counts: pd.DataFrame) -> go.Figure:
    """Bar histogram of G3 (0–20), coloured and hatched by outcome."""
    if counts.empty or counts["n"].sum() == 0:
        return empty_figure("No records match the current filters.")
    fig = go.Figure()
    for outcome in ("Fail", "Pass"):
        part = counts[counts["outcome"] == outcome]
        fig.add_bar(
            x=part[config.OUTCOME_COL],
            y=part["n"],
            name=outcome,
            marker={
                "color": config.OUTCOME_COLORS[outcome],
                "pattern": {"shape": config.OUTCOME_PATTERNS[outcome], "fgcolor": "white"},
                "line": {"color": "white", "width": 1},
            },
            customdata=np.full(len(part), outcome),
            hovertemplate="G3 = %{x}<br>%{customdata}<br>Records: %{y}<extra></extra>",
        )
    fig.add_vline(
        x=config.PASS_THRESHOLD - 0.5,
        line={"color": _MUTED, "dash": "dash", "width": 1.5},
        annotation={"text": f"Pass threshold (G3 ≥ {config.PASS_THRESHOLD})", "font_size": 12},
    )
    fig.update_xaxes(dtick=1, range=[-0.6, 20.6])
    return apply_theme(fig, FIGURE_TITLES["g3_distribution"], _label("G3"), "Number of records")


def fig_mean_g3(stats: pd.DataFrame, by: str, slug: str) -> go.Figure:
    """Mean G3 per group with 95% CI error bars and ``n`` above each bar."""
    if stats.empty:
        return empty_figure(f"'{by}' or G3 not available for the current data.")
    labels = [_value_label(by, v) for v in stats[by]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=stats["mean_g3"],
            marker={"color": config.SINGLE_SERIES_COLOR},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": stats["ci_high"] - stats["mean_g3"],
                "arrayminus": stats["mean_g3"] - stats["ci_low"],
                "color": _MUTED,
                "thickness": 1.5,
            },
            text=[f"n={n}" for n in stats["n"]],
            textposition="outside",
            customdata=np.stack([stats["n"], stats["ci_low"], stats["ci_high"]], axis=-1),
            hovertemplate=(
                "%{x}<br>Mean G3: %{y:.2f}<br>95% CI: %{customdata[1]:.2f}–%{customdata[2]:.2f}"
                "<br>n = %{customdata[0]}<extra></extra>"
            ),
            cliponaxis=False,
        )
    )
    fig.update_yaxes(range=[0, 20])
    fig.update_xaxes(type="category")
    return apply_theme(fig, FIGURE_TITLES[slug], _label(by), "Mean final grade (G3)")


def fig_correlation_bar(corr: pd.DataFrame) -> go.Figure:
    """Sorted horizontal bar of Pearson r with G3; sign encoded by colour and label."""
    if corr.empty:
        return empty_figure("Not enough numeric data to compute correlations.")
    colors = [config.PALETTE["blue"] if r >= 0 else config.PALETTE["vermillion"] for r in corr["r"]]
    fig = go.Figure(
        go.Bar(
            x=corr["r"],
            y=corr["feature"],
            orientation="h",
            marker={"color": colors},
            text=[f"{r:+.2f}" for r in corr["r"]],
            textposition="outside",
            customdata=corr["n"],
            hovertemplate="%{y}<br>r = %{x:.3f}<br>n = %{customdata}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_xaxes(range=[-1, 1], zeroline=True, zerolinecolor=_MUTED)
    fig.update_layout(height=max(380, 26 * len(corr)))
    return apply_theme(
        fig, FIGURE_TITLES["correlation_with_g3"], "Pearson correlation with G3 (r)", ""
    )


def fig_g2_g3(pairs: pd.DataFrame) -> go.Figure:
    """G2 vs G3 bubble scatter: area = record count, symbol + colour = outcome."""
    if pairs.empty:
        return empty_figure("G2 and G3 are required for this chart.")
    fig = go.Figure()
    max_n = pairs["n"].max()
    for outcome in ("Fail", "Pass"):
        part = pairs[pairs["outcome"] == outcome]
        fig.add_scatter(
            x=part["G2"],
            y=part["G3"],
            mode="markers",
            name=outcome,
            marker={
                "color": config.OUTCOME_COLORS[outcome],
                "symbol": config.OUTCOME_SYMBOLS[outcome],
                "size": 8 + 22 * np.sqrt(part["n"] / max_n),
                "opacity": 0.75,
                "line": {"color": "white", "width": 1},
            },
            customdata=np.stack([part["n"], np.full(len(part), outcome)], axis=-1),
            hovertemplate=(
                "G2 = %{x}, G3 = %{y}<br>%{customdata[1]}<br>Records: %{customdata[0]}"
                "<extra></extra>"
            ),
        )
    fig.add_scatter(
        x=[0, 20], y=[0, 20], mode="lines", name="G3 = G2",
        line={"color": _MUTED, "dash": "dot", "width": 1}, hoverinfo="skip",
    )  # fmt: skip
    fig.update_xaxes(range=[-0.5, 20.5], dtick=2)
    fig.update_yaxes(range=[-0.5, 20.5], dtick=2)
    return apply_theme(fig, FIGURE_TITLES["g2_vs_g3"], _label("G2"), _label("G3"))


def fig_g3_box(df: pd.DataFrame, by: str, slug: str) -> go.Figure:
    """Box plots of G3 for each level of ``by`` (n in each category label)."""
    if by not in df.columns or config.OUTCOME_COL not in df.columns or df.empty:
        return empty_figure(f"'{by}' or G3 not available for the current data.")
    fig = go.Figure()
    for level in sorted(df[by].dropna().unique()):
        values = df.loc[df[by] == level, config.OUTCOME_COL].dropna()
        color = config.SUBJECT_COLORS.get(str(level), config.SINGLE_SERIES_COLOR)
        fig.add_box(
            y=values,
            name=f"{_value_label(by, level)} (n={len(values)})",
            marker={"color": color, "size": 5},
            line={"width": 1.5},
            boxpoints="outliers",
            boxmean=True,
            hoverinfo="y",
        )
    fig.update_yaxes(range=[-0.5, 20.5])
    fig.update_layout(showlegend=False)
    return apply_theme(fig, FIGURE_TITLES[slug], _label(by), _label("G3"))


def fig_pass_rate_bands(rates: pd.DataFrame, by: str = config.ABSENCE_BAND_COL) -> go.Figure:
    """Pass rate per ordered band with Wilson CI and ``n`` labels."""
    if rates.empty or rates["n"].sum() == 0:
        return empty_figure("Absences and G3 are required for this chart.")
    fig = go.Figure(
        go.Bar(
            x=rates[by].astype(str),
            y=rates["pass_rate"],
            marker={"color": config.SINGLE_SERIES_COLOR},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": rates["ci_high"] - rates["pass_rate"],
                "arrayminus": rates["pass_rate"] - rates["ci_low"],
                "color": _MUTED,
                "thickness": 1.5,
            },
            text=[f"n={n}" for n in rates["n"]],
            textposition="outside",
            customdata=np.stack([rates["n"], rates["n_pass"]], axis=-1),
            hovertemplate=(
                "Absences %{x}<br>Pass rate: %{y:.1%}<br>%{customdata[1]} of %{customdata[0]} "
                "passed<extra></extra>"
            ),
            cliponaxis=False,
        )
    )
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(type="category")
    return apply_theme(fig, FIGURE_TITLES["pass_rate_by_absence_band"], _label(by), "Pass rate")


def fig_corr_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Diverging heatmap (−1 … +1, neutral at 0) with values printed in cells."""
    if matrix.empty:
        return empty_figure("Not enough numeric data to compute correlations.")
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=config.DIVERGING_SCALE,
            text=np.round(matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar={"title": "r"},
            hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-45)
    fig.update_layout(height=640)
    return apply_theme(fig, FIGURE_TITLES["correlation_heatmap"])


def build_figures(df: pd.DataFrame) -> dict[str, go.Figure]:
    """Build every exploratory Plotly figure for ``df``, keyed by figure slug."""
    return {
        "g3_distribution": fig_g3_distribution(analytics.grade_counts(df)),
        "mean_g3_by_studytime": fig_mean_g3(
            analytics.mean_g3_by(df, "studytime"), "studytime", "mean_g3_by_studytime"
        ),
        "mean_g3_by_failures": fig_mean_g3(
            analytics.mean_g3_by(df, "failures"), "failures", "mean_g3_by_failures"
        ),
        "correlation_with_g3": fig_correlation_bar(analytics.correlations_with_g3(df)),
        "g2_vs_g3": fig_g2_g3(analytics.grade_pair_counts(df)),
        "g3_by_subject": fig_g3_box(df, config.SUBJECT_COL, "g3_by_subject"),
        "g3_by_higher": fig_g3_box(df, "higher", "g3_by_higher"),
        "pass_rate_by_absence_band": fig_pass_rate_bands(
            analytics.pass_rate_by(df, config.ABSENCE_BAND_COL)
        ),
        "correlation_heatmap": fig_corr_heatmap(analytics.correlation_matrix(df)),
    }


# --------------------------------------------------------------------------- #
# Matplotlib static exports (300 dpi PNG)
# --------------------------------------------------------------------------- #


def _mpl_axes(title: str, xlabel: str, ylabel: str, size: tuple[float, float] = (8, 5)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    ax.set_title(title, loc="left", fontsize=13, color=_INK)
    ax.set_xlabel(xlabel, fontsize=11, color=_INK)
    ax.set_ylabel(ylabel, fontsize=11, color=_INK)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _mpl_empty(title: str, message: str) -> Figure:
    fig, ax = _mpl_axes(title, "", "")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=_MUTED, transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def mpl_g3_distribution(counts: pd.DataFrame) -> Figure:
    """Render the static Matplotlib version of :func:`fig_g3_distribution`."""
    title = FIGURE_TITLES["g3_distribution"]
    if counts.empty or counts["n"].sum() == 0:
        return _mpl_empty(title, "No records")
    fig, ax = _mpl_axes(title, _label("G3"), "Number of records")
    for outcome, hatch in (("Fail", "///"), ("Pass", "")):
        part = counts[counts["outcome"] == outcome]
        ax.bar(part["G3"], part["n"], color=config.OUTCOME_COLORS[outcome], hatch=hatch,
               edgecolor="white", linewidth=0.8, label=outcome)  # fmt: skip
    ax.axvline(config.PASS_THRESHOLD - 0.5, color=_MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(range(0, 21))
    ax.legend(frameon=False)
    return fig


def mpl_mean_g3(stats: pd.DataFrame, by: str, slug: str) -> Figure:
    """Render the static Matplotlib version of :func:`fig_mean_g3`."""
    if stats.empty:
        return _mpl_empty(FIGURE_TITLES[slug], "Not available")
    fig, ax = _mpl_axes(FIGURE_TITLES[slug], _label(by), "Mean final grade (G3)")
    labels = [_value_label(by, v) for v in stats[by]]
    err = np.vstack([stats["mean_g3"] - stats["ci_low"], stats["ci_high"] - stats["mean_g3"]])
    bars = ax.bar(labels, stats["mean_g3"], color=config.SINGLE_SERIES_COLOR, yerr=err,
                  ecolor=_MUTED, capsize=4)  # fmt: skip
    for bar, n, hi in zip(bars, stats["n"], stats["ci_high"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.3, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, 20)
    return fig


def mpl_correlation_bar(corr: pd.DataFrame) -> Figure:
    """Render the static Matplotlib version of :func:`fig_correlation_bar`."""
    title = FIGURE_TITLES["correlation_with_g3"]
    if corr.empty:
        return _mpl_empty(title, "Not available")
    fig, ax = _mpl_axes(title, "Pearson correlation with G3 (r)", "", (8, max(4, 0.32 * len(corr))))
    colors = [config.PALETTE["blue"] if r >= 0 else config.PALETTE["vermillion"] for r in corr["r"]]
    ax.barh(corr["feature"], corr["r"], color=colors)
    for y, r in enumerate(corr["r"]):
        ax.text(r + (0.02 if r >= 0 else -0.02), y, f"{r:+.2f}", va="center",
                ha="left" if r >= 0 else "right", fontsize=8)  # fmt: skip
    ax.axvline(0, color=_MUTED, linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.grid(axis="x", color=_GRID)
    ax.grid(axis="y", visible=False)
    return fig


def mpl_g2_g3(pairs: pd.DataFrame) -> Figure:
    """Render the static Matplotlib version of :func:`fig_g2_g3`."""
    title = FIGURE_TITLES["g2_vs_g3"]
    if pairs.empty:
        return _mpl_empty(title, "Not available")
    fig, ax = _mpl_axes(title, _label("G2"), _label("G3"), (7, 6))
    max_n = pairs["n"].max()
    for outcome, marker in (("Fail", "X"), ("Pass", "o")):
        part = pairs[pairs["outcome"] == outcome]
        ax.scatter(part["G2"], part["G3"], s=20 + 380 * part["n"] / max_n, marker=marker,
                   color=config.OUTCOME_COLORS[outcome], alpha=0.7, edgecolors="white",
                   linewidths=0.6, label=outcome)  # fmt: skip
    ax.plot([0, 20], [0, 20], linestyle=":", color=_MUTED, linewidth=1, label="G3 = G2")
    ax.set_xlim(-0.5, 20.5)
    ax.set_ylim(-0.5, 20.5)
    ax.set_xticks(range(0, 21, 2))
    ax.set_yticks(range(0, 21, 2))
    ax.legend(frameon=False, loc="upper left")
    return fig


def mpl_g3_box(df: pd.DataFrame, by: str, slug: str) -> Figure:
    """Render the static Matplotlib version of :func:`fig_g3_box`."""
    if by not in df.columns or config.OUTCOME_COL not in df.columns or df.empty:
        return _mpl_empty(FIGURE_TITLES[slug], "Not available")
    fig, ax = _mpl_axes(FIGURE_TITLES[slug], _label(by), _label("G3"))
    levels = sorted(df[by].dropna().unique())
    data = [df.loc[df[by] == lv, config.OUTCOME_COL].dropna() for lv in levels]
    box = ax.boxplot(data, patch_artist=True, showmeans=True, widths=0.5,
                     medianprops={"color": _INK, "linewidth": 1.5},
                     meanprops={"marker": "D", "markerfacecolor": "white",
                                "markeredgecolor": _INK, "markersize": 6},
                     tick_labels=[f"{_value_label(by, lv)}\n(n={len(d)})"
                                  for lv, d in zip(levels, data, strict=True)])  # fmt: skip
    for patch, lv in zip(box["boxes"], levels, strict=True):
        patch.set_facecolor(config.SUBJECT_COLORS.get(str(lv), config.SINGLE_SERIES_COLOR))
        patch.set_alpha(0.75)
    ax.set_ylim(-0.5, 20.5)
    return fig


def mpl_pass_rate_bands(rates: pd.DataFrame, by: str = config.ABSENCE_BAND_COL) -> Figure:
    """Render the static Matplotlib version of :func:`fig_pass_rate_bands`."""
    title = FIGURE_TITLES["pass_rate_by_absence_band"]
    if rates.empty or rates["n"].sum() == 0:
        return _mpl_empty(title, "Not available")
    fig, ax = _mpl_axes(title, _label(by), "Pass rate")
    p = rates["pass_rate"].fillna(0)
    err = np.vstack([(p - rates["ci_low"]).fillna(0), (rates["ci_high"] - p).fillna(0)])
    bars = ax.bar(rates[by].astype(str), p, color=config.SINGLE_SERIES_COLOR, yerr=err,
                  ecolor=_MUTED, capsize=4)  # fmt: skip
    for bar, n, hi in zip(bars, rates["n"], rates["ci_high"].fillna(0), strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + 0.02, f"n={n}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    return fig


def mpl_corr_heatmap(matrix: pd.DataFrame) -> Figure:
    """Render the static Matplotlib version of :func:`fig_corr_heatmap`."""
    title = FIGURE_TITLES["correlation_heatmap"]
    if matrix.empty:
        return _mpl_empty(title, "Not available")
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    im = ax.imshow(matrix.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 else _INK)  # fmt: skip
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(title, loc="left", fontsize=13)
    return fig


STATIC_BUILDERS: dict[str, Callable[[pd.DataFrame], Figure]] = {
    "g3_distribution": lambda d: mpl_g3_distribution(analytics.grade_counts(d)),
    "mean_g3_by_studytime": lambda d: mpl_mean_g3(
        analytics.mean_g3_by(d, "studytime"), "studytime", "mean_g3_by_studytime"
    ),
    "mean_g3_by_failures": lambda d: mpl_mean_g3(
        analytics.mean_g3_by(d, "failures"), "failures", "mean_g3_by_failures"
    ),
    "correlation_with_g3": lambda d: mpl_correlation_bar(analytics.correlations_with_g3(d)),
    "g2_vs_g3": lambda d: mpl_g2_g3(analytics.grade_pair_counts(d)),
    "g3_by_subject": lambda d: mpl_g3_box(d, config.SUBJECT_COL, "g3_by_subject"),
    "g3_by_higher": lambda d: mpl_g3_box(d, "higher", "g3_by_higher"),
    "pass_rate_by_absence_band": lambda d: mpl_pass_rate_bands(
        analytics.pass_rate_by(d, config.ABSENCE_BAND_COL)
    ),
    "correlation_heatmap": lambda d: mpl_corr_heatmap(analytics.correlation_matrix(d)),
}


def export_static_figures(
    df: pd.DataFrame, out_dir: Path = config.FIGURES_DIR, dpi: int = config.EXPORT_DPI
) -> list[Path]:
    """Render every exploratory chart with Matplotlib and save PNGs.

    Args:
        df: (Filtered) prepared dataset.
        out_dir: Destination folder, created if needed.
        dpi: Output resolution.

    Returns:
        Paths of the written files, in :data:`STATIC_BUILDERS` order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with plt.rc_context({"font.size": 10, "font.family": "sans-serif"}):
        for slug, builder in STATIC_BUILDERS.items():
            fig = builder(df)
            path = out_dir / f"{slug}.png"
            fig.savefig(path, dpi=dpi, facecolor="white")
            plt.close(fig)
            written.append(path)
    return written
