"""Static report figures (Matplotlib, 300 dpi PNG).

Every figure is computed from the same :mod:`src` functions the dashboard
uses, so exported numbers match the interactive views. Colours follow the
dashboard's validated palette; categories are also distinguished by hatching,
markers or labels so the PNGs remain readable in greyscale print.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

from src import analytics, config, modeling, segmentation  # noqa: E402

C = config
INK, MUTED, GRID = "#1A1A1A", "#5F6368", "#E6E8EB"
CUR = config.CURRENCY_SYMBOL


@dataclass
class ReportInputs:
    """Everything needed to draw the report figures (computed once)."""

    lines: pd.DataFrame
    scope: pd.DataFrame
    data_start: pd.Timestamp
    data_end: pd.Timestamp
    seg: segmentation.SegmentationResult
    ds: modeling.ChurnDataset | None
    cv: dict[tuple[str, str], modeling.CVResult] | None


def prepare_inputs(
    data: pd.DataFrame, filters: analytics.Filters | None = None, with_models: bool = True
) -> ReportInputs:
    """Compute analytics, segmentation and (optionally) model CV for the figures."""
    f = filters or analytics.Filters()
    scope = analytics.scope_filter(data, f)
    lines = scope.loc[analytics.line_mask(scope, f)]
    seg_lines = scope.loc[analytics.line_mask(scope, analytics.Filters())]
    seg = segmentation.segment_customers(seg_lines)
    ds = cv = None
    if with_models:
        try:
            ds = modeling.build_dataset(data)
            cv = modeling.evaluate_all(ds)
        except ValueError:
            ds = cv = None
    return ReportInputs(lines, scope, data[C.COL_DATE].min(), data[C.COL_DATE].max(), seg, ds, cv)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _axes(title: str, xlabel: str = "", ylabel: str = "", size=(9, 5.2)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.set_xlabel(xlabel, fontsize=11, color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _empty(title: str, message: str = "Not available for this dataset") -> Figure:
    fig, ax = _axes(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED, transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def _money_fmt(cur: str = CUR) -> FuncFormatter:
    def fmt(v, _):
        for size, suffix in ((1e6, "M"), (1e3, "k")):
            if abs(v) >= size:
                return f"{cur}{v / size:g}{suffix}"
        return f"{cur}{v:,.0f}"

    return FuncFormatter(fmt)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def monthly_revenue(r: ReportInputs) -> Figure:
    """Monthly revenue; partial months drawn hollow and dashed."""
    ts = analytics.time_series(r.lines, "Monthly", r.data_end, r.data_start)
    if ts.empty:
        return _empty("Monthly revenue")
    fig, ax = _axes("Monthly revenue (completed sales)", "Month", f"Revenue ({CUR})")
    ax.plot(ts["period_start"], ts["revenue"], color=C.SINGLE_SERIES_COLOR, lw=1.2, ls=":")
    full = ts[~ts["partial"]]
    ax.plot(
        full["period_start"],
        full["revenue"],
        color=C.SINGLE_SERIES_COLOR,
        lw=2,
        marker="o",
        label="Complete month",
    )
    part = ts[ts["partial"]]
    if len(part):
        ax.scatter(
            part["period_start"],
            part["revenue"],
            s=70,
            facecolor="white",
            edgecolor=C.SINGLE_SERIES_COLOR,
            lw=2,
            zorder=3,
            label="Partial month",
        )
    ax.yaxis.set_major_formatter(_money_fmt())
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    return fig


def weekday_hour(r: ReportInputs) -> Figure:
    """Revenue heatmap by weekday × hour; no-trading cells hatched grey."""
    mat = analytics.weekday_hour_matrix(r.lines)
    if mat.empty:
        return _empty("Revenue by weekday and hour")
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    z = np.ma.masked_invalid(mat.to_numpy(dtype=float))
    cmap = plt.get_cmap("Blues").with_extremes(bad="#D9DCE0")
    im = ax.imshow(z, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(mat.columns)), [f"{h:02d}" for h in mat.columns])
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i, day in enumerate(mat.index):
        if mat.loc[day].isna().all():
            ax.text(
                len(mat.columns) / 2,
                i,
                "No trading recorded",
                ha="center",
                va="center",
                fontsize=9,
                color=INK,
            )
    fig.colorbar(im, ax=ax, label=f"Revenue ({CUR})", format=_money_fmt())
    ax.set_title("Revenue by weekday and hour of invoice", loc="left", fontsize=13)
    ax.set_xlabel("Hour of day")
    return fig


def top_products(r: ReportInputs, n: int = 15) -> Figure:
    """Top products by revenue (horizontal bars, value labels)."""
    top = analytics.top_products(r.lines, n)
    if top.empty:
        return _empty("Top products")
    data = top.iloc[::-1]
    fig, ax = _axes(f"Top {len(top)} products by revenue", f"Revenue ({CUR})", "", (10, 6.5))
    names = [
        f"{str(d)[:38]} ({s})" for d, s in zip(data[C.COL_DESC], data[C.COL_STOCK], strict=True)
    ]
    ax.barh(names, data["revenue"], color=C.SINGLE_SERIES_COLOR)
    for y, v in enumerate(data["revenue"]):
        ax.text(v, y, f" {CUR}{v:,.0f}", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(_money_fmt())
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    return fig


def country_share(r: ReportInputs) -> Figure:
    """Donut of revenue share by country (top 5 + Other) with percentage labels."""
    mix = analytics.revenue_share(r.lines)
    if mix.empty:
        return _empty("Revenue by country")
    colors = [
        C.OTHER_COLOR if lbl == "Other" else C.CATEGORICAL_ORDER[i % 5]
        for i, lbl in enumerate(mix["label"])
    ]
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    wedges, _ = ax.pie(
        mix["value"],
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    ax.legend(
        wedges,
        [f"{lbl} · {s:.1%}" for lbl, s in zip(mix["label"], mix["share"], strict=True)],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    ax.text(0, 0, f"{CUR}{mix['value'].sum() / 1e6:.2f}M", ha="center", va="center", fontsize=13)
    ax.set_title("Revenue by country", loc="left", fontsize=13)
    return fig


def rfm_segments(r: ReportInputs) -> Figure:
    """Share of customers vs share of revenue per RFM segment (paired bars)."""
    s = r.seg.summary
    if s.empty:
        return _empty("RFM segments")
    data = s.iloc[::-1]
    fig, ax = _axes("RFM segments: share of customers vs share of revenue", "Share", "", (10, 6))
    y = np.arange(len(data))
    ax.barh(
        y + 0.2,
        data["customer_share"],
        0.4,
        color="#56B4E9",
        hatch="//",
        edgecolor="white",
        label="Customers",
    )
    ax.barh(y - 0.2, data["revenue_share"], 0.4, color=C.SINGLE_SERIES_COLOR, label="Revenue")
    for yi, (c, v) in enumerate(zip(data["customer_share"], data["revenue_share"], strict=True)):
        ax.text(c, yi + 0.2, f" {c:.0%}", va="center", fontsize=8)
        ax.text(v, yi - 0.2, f" {v:.0%}", va="center", fontsize=8)
    ax.set_yticks(y, data["segment"])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    return fig


def k_selection(r: ReportInputs) -> Figure:
    """Inertia (elbow) and silhouette by k, side by side; recommended k dashed."""
    ev = r.seg.evaluation
    if ev.empty:
        return _empty("Choosing k")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, col, title in (
        (a1, "inertia", "Inertia (elbow)"),
        (a2, "silhouette", "Silhouette (higher = better)"),
    ):
        ax.plot(ev["k"], ev[col], marker="o", color=C.SINGLE_SERIES_COLOR, lw=2)
        if r.seg.recommended_k:
            ax.axvline(r.seg.recommended_k, ls="--", color=MUTED, lw=1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(ev["k"])
        ax.grid(color=GRID)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        f"Choosing the number of clusters — elbow at k = {r.seg.recommended_k}",
        x=0.01,
        ha="left",
        fontsize=13,
    )
    return fig


def cluster_profiles(r: ReportInputs) -> Figure:
    """Diverging heatmap of cluster profiles (blue = better on every column)."""
    if r.seg.clusters is None:
        return _empty("Cluster profiles")
    prof = r.seg.clusters.profile
    z = np.column_stack([-prof["z_recency"], prof["z_frequency"], prof["z_monetary"]])
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * len(prof)), constrained_layout=True)
    im = ax.imshow(z, cmap="RdBu", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(3), ["Recentness", "Frequency", "Monetary"])
    ax.set_yticks(range(len(prof)), prof["name"])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(
                j,
                i,
                f"{z[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(z[i, j]) > 1.2 else INK,
            )
    fig.colorbar(im, ax=ax, label="SD from average customer")
    ax.set_title("K-Means cluster profiles (blue = better)", loc="left", fontsize=13)
    return fig


def cluster_scatter(r: ReportInputs) -> Figure:
    """Recency vs monetary (log) per customer, colour + marker by cluster."""
    if r.seg.clusters is None:
        return _empty("Customers by cluster")
    cust, prof = r.seg.customers, r.seg.clusters.profile.set_index("cluster")
    fig, ax = _axes(
        "Customers by cluster",
        "Recency (days since last purchase)",
        f"Monetary value ({CUR}, log scale)",
        (10, 6),
    )
    markers = ["o", "s", "D", "^", "x", "*"]
    for i, cl in enumerate(sorted(cust["cluster"].unique())):
        part = cust[cust["cluster"] == cl]
        ax.scatter(
            part["recency"],
            part["monetary"],
            s=10,
            alpha=0.6,
            color=C.CLUSTER_COLORS[i % 6],
            marker=markers[i % 6],
            label=prof.loc[cl, "name"],
        )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(_money_fmt())
    ax.legend(frameon=False, fontsize=8, markerscale=1.8, loc="upper right")
    return fig


def _cv_or_empty(r: ReportInputs, title: str) -> Figure | None:
    return None if r.cv is not None and r.ds is not None else _empty(title)


def model_comparison(r: ReportInputs) -> Figure:
    """ROC-AUC by model for settings A and B with fold SD and the baseline."""
    if (fig := _cv_or_empty(r, "Model comparison")) is not None:
        return fig
    table = modeling.results_table(r.cv, r.ds.y)
    models = list(C.MODEL_NAMES)
    fig, ax = _axes("ROC-AUC by model and feature setting (± fold SD)", "", "ROC-AUC")
    x = np.arange(len(models))
    for off, setting, hatch in ((-0.2, "A", "//"), (0.2, "B", "")):
        rows = table[table.setting == setting].set_index("model").reindex(models)
        bars = ax.bar(
            x + off,
            rows["roc_auc"],
            0.4,
            yerr=rows["roc_auc_fold_std"],
            color=C.SETTING_COLORS[setting],
            hatch=hatch,
            edgecolor="white",
            ecolor=MUTED,
            capsize=4,
            label=f"Setting {setting}",
        )
        for b, v in zip(bars, rows["roc_auc"], strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center", fontsize=8)
    base = float(table[table.model == modeling.BASELINE]["roc_auc"].iloc[0])
    ax.axhline(base, ls="--", color=MUTED, lw=1)
    ax.text(
        -0.45,
        base + 0.015,
        f"Majority-class baseline {base:.3f}",
        ha="left",
        color=MUTED,
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        zorder=5,
    )
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper left")
    return fig


def _best(r: ReportInputs) -> tuple[str, modeling.CVResult]:
    table = modeling.results_table(r.cv, r.ds.y)
    name = modeling.best_model(table, C.DEFAULT_CHURN_SETTING)
    return name, r.cv[(C.DEFAULT_CHURN_SETTING, name)]


def roc_curves(r: ReportInputs) -> Figure:
    """Out-of-fold ROC curves for every model (setting B)."""
    if (fig := _cv_or_empty(r, "ROC curves")) is not None:
        return fig
    fig, ax = _axes(
        f"ROC curves, setting {C.DEFAULT_CHURN_SETTING} (out-of-fold)",
        "False positive rate",
        "True positive rate",
        (7, 6),
    )
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="Random / baseline (AUC 0.500)")
    styles = {"Logistic Regression": "-", "Decision Tree": ":", "Random Forest": "-."}
    for name in C.MODEL_NAMES:
        res = r.cv[(C.DEFAULT_CHURN_SETTING, name)]
        pts = modeling.roc_points(r.ds.y, res.oof_proba)
        auc = modeling.classification_metrics(r.ds.y.to_numpy(), res.oof_proba.to_numpy())
        ax.plot(
            pts["fpr"],
            pts["tpr"],
            ls=styles[name],
            lw=2,
            color=C.MODEL_COLORS[name],
            label=f"{name} (AUC {auc['roc_auc']:.3f})",
        )
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(color=GRID)
    return fig


def confusion(r: ReportInputs) -> Figure:
    """Confusion matrix of the best setting-B model at threshold 0.5."""
    if (fig := _cv_or_empty(r, "Confusion matrix")) is not None:
        return fig
    name, res = _best(r)
    cm = modeling.confusion(r.ds.y, res.oof_proba, C.DEFAULT_THRESHOLD)
    z = cm.to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    ax.imshow(z, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{z[i, j]:,}\n({z[i, j] / z[i].sum():.0%} of row)",
                ha="center",
                va="center",
                color="white" if z[i, j] > z.max() * 0.6 else INK,
            )
    ax.set_xticks([0, 1], ["Predicted\nretained", "Predicted\nchurned"])
    ax.set_yticks([0, 1], ["Actual retained", "Actual churned"])
    ax.set_title(
        f"Confusion matrix — {name}, threshold {C.DEFAULT_THRESHOLD}", loc="left", fontsize=12
    )
    return fig


def threshold_tradeoff(r: ReportInputs) -> Figure:
    """Churn recall, precision and flagged share across thresholds (best model)."""
    if (fig := _cv_or_empty(r, "Threshold trade-off")) is not None:
        return fig
    name, res = _best(r)
    curve = modeling.threshold_curve(r.ds.y, res.oof_proba)
    fig, ax = _axes(f"Recall vs precision by threshold — {name}", "Decision threshold", "Rate")
    ax.plot(
        curve["threshold"],
        curve["recall"],
        color=C.PALETTE["vermillion"],
        lw=2,
        label="Recall (churners caught)",
    )
    ax.plot(
        curve["threshold"],
        curve["precision"],
        color=C.PALETTE["blue"],
        lw=2,
        ls="--",
        label="Precision (flags that are right)",
    )
    ax.plot(
        curve["threshold"],
        curve["flagged_share"],
        color=MUTED,
        lw=1.5,
        ls=":",
        label="Share flagged",
    )
    ax.axvline(C.DEFAULT_THRESHOLD, color=INK, lw=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    return fig


def coefficients(r: ReportInputs) -> Figure:
    """Drivers of the best setting-B model (coefficients or importances)."""
    if (fig := _cv_or_empty(r, "Model drivers")) is not None:
        return fig
    name, _ = _best(r)
    feats = modeling.feature_list(C.DEFAULT_CHURN_SETTING)
    pipe = modeling.make_model(name).fit(r.ds.X[feats], r.ds.y)
    imp = modeling.feature_importance(pipe, feats).iloc[::-1]
    signed = (imp["value"] < 0).any()
    fig, ax = _axes(
        f"What drives churn risk — {name}",
        "Coefficient (+ raises churn risk)" if signed else "Importance",
        "",
        (9, 5.5),
    )
    colors = (
        [C.PALETTE["vermillion"] if v >= 0 else C.PALETTE["blue"] for v in imp["value"]]
        if signed
        else C.SINGLE_SERIES_COLOR
    )
    ax.barh(imp["label"], imp["value"], color=colors)
    for y, v in enumerate(imp["value"]):
        ax.text(
            v,
            y,
            f" {v:+.2f} " if signed else f" {v:.3f}",
            va="center",
            fontsize=8,
            ha="left" if v >= 0 else "right",
        )
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.2)
    return fig


FIGURES: dict[str, tuple[Callable[[ReportInputs], Figure], str]] = {
    "01_monthly_revenue": (
        monthly_revenue,
        "Monthly completed-sales revenue; partial months marked.",
    ),
    "02_weekday_hour_heatmap": (weekday_hour, "Revenue by weekday × hour; no-trading cells grey."),
    "03_top_products": (top_products, "Top 15 products by revenue."),
    "04_country_share": (country_share, "Revenue share by country (top 5 + Other)."),
    "05_rfm_segments": (rfm_segments, "Share of customers vs revenue per RFM segment."),
    "06_k_selection": (k_selection, "Inertia and silhouette for k = 2–10; elbow marked."),
    "07_cluster_profiles": (cluster_profiles, "K-Means cluster profiles in SD units."),
    "08_cluster_scatter": (cluster_scatter, "Customers by recency and spend, per cluster."),
    "09_model_comparison": (model_comparison, "ROC-AUC per model, settings A and B vs baseline."),
    "10_roc_curves": (roc_curves, "Out-of-fold ROC curves (setting B)."),
    "11_confusion_matrix": (confusion, "Confusion matrix of the best model at 0.5."),
    "12_threshold_tradeoff": (threshold_tradeoff, "Recall/precision trade-off by threshold."),
    "13_churn_drivers": (coefficients, "Coefficients / importances of the best model."),
}


def export_all(
    inputs: ReportInputs, out_dir: Path = config.FIGURES_DIR, dpi: int = config.EXPORT_DPI
) -> list[Path]:
    """Render every report figure to ``out_dir`` as PNG and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context({"font.size": 10, "font.family": "sans-serif"}):
        for slug, (builder, _) in FIGURES.items():
            fig = builder(inputs)
            path = out_dir / f"{slug}.png"
            fig.savefig(path, dpi=dpi, facecolor="white")
            plt.close(fig)
            written.append(path)
    return written
