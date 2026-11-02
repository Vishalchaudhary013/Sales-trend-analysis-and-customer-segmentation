"""Train and evaluate the churn early-warning models from the command line.

Usage::

    python scripts/train.py                       # evaluate A/B, save best model (setting B)
    python scripts/train.py --class-weight balanced
    python scripts/train.py --setting A --model "Decision Tree" --threshold 0.4

Writes ``reports/metrics/metrics.json`` and ``reports/metrics/metrics.md``
(settings A and B, every model plus the majority-class baseline, pooled
out-of-fold metrics and fold mean ± std), saves the selected model to
``models/churn_model.joblib`` (+ ``.json`` metadata) and registers it in SQLite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src import config, ingestion, modeling, storage  # noqa: E402

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision (churn)",
    "recall": "Recall (churn)",
    "f1": "F1 (churn)",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "retained_recall": "Recall (retained)",
}


def markdown_report(
    table: pd.DataFrame,
    ds: modeling.ChurnDataset,
    threshold: float,
    class_weight: str | None,
    chosen: dict[str, str],
) -> str:
    """Render the metrics table and context as Markdown."""
    lines = [
        "# Churn model evaluation",
        "",
        f"- Dataset: {config.DATASET_NAME}; customers: {len(ds.y):,}; "
        f"churn rate: {ds.churn_rate:.1%}",
        f"- Feature cut-off: {ds.cutoff:%Y-%m-%d}; label window: {config.CHURN_HORIZON_DAYS} "
        f"days to {ds.horizon_end:%Y-%m-%d}",
        "- Positive class: churned (no completed purchase in the label window)",
        f"- Validation: StratifiedKFold(n_splits={config.CV_FOLDS}, shuffle={config.CV_SHUFFLE}, "
        f"random_state={config.RANDOM_STATE}); identical folds for all models and settings",
        f"- Decision threshold: {threshold}; class_weight: {class_weight}",
        "- Metrics: pooled out-of-fold predictions; ± = standard deviation across folds",
        "",
    ]
    for setting, desc in config.CHURN_SETTINGS.items():
        lines += [f"## Setting {setting}: {desc}", ""]
        header = "| Model | " + " | ".join(METRIC_LABELS.values()) + " |"
        lines += [header, "|" + "---|" * (len(METRIC_LABELS) + 1)]
        for _, row in table[table["setting"] == setting].iterrows():
            cells = [f"{row[m]:.3f} ± {row[f'{m}_fold_std']:.3f}" for m in METRIC_LABELS]
            lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"Selected model: **{chosen['model']}** (setting {chosen['setting']}).", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--setting", choices=["A", "B"], default=config.DEFAULT_CHURN_SETTING)
    parser.add_argument(
        "--model",
        choices=list(config.MODEL_NAMES),
        help="Model to save (default: best fold-mean ROC-AUC)",
    )
    parser.add_argument("--class-weight", choices=["balanced"], default=None)
    parser.add_argument("--threshold", type=float, default=config.DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    result = ingestion.load_bundled_cached()
    if not result.is_valid or result.data is None:
        print(
            "Dataset missing or invalid; run python scripts/fetch_data.py first.", file=sys.stderr
        )
        return 1
    try:
        ds = modeling.build_dataset(result.data)
    except ValueError as exc:
        print(f"Cannot build the churn dataset: {exc}", file=sys.stderr)
        return 1
    print(
        f"Customers: {len(ds.y):,}  churn rate: {ds.churn_rate:.1%}  cut-off: {ds.cutoff:%Y-%m-%d}"
    )
    results = modeling.evaluate_all(ds, args.class_weight)
    table = modeling.results_table(results, ds.y, args.threshold)
    model_name = args.model or modeling.best_model(table, args.setting)
    chosen = {"model": model_name, "setting": args.setting}

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": {
            "customers": int(len(ds.y)),
            "churn_rate": ds.churn_rate,
            "cutoff": ds.cutoff.isoformat(),
            "horizon_days": config.CHURN_HORIZON_DAYS,
            "threshold": args.threshold,
            "class_weight": args.class_weight,
            "folds": config.CV_FOLDS,
            "random_state": config.RANDOM_STATE,
            "positive_class": "churned",
        },
        "results": table.to_dict(orient="records"),
        "fold_metrics": {
            f"{s}|{m}": r.fold_metrics.to_dict(orient="records") for (s, m), r in results.items()
        },
        "selected": chosen,
    }
    json_path = config.METRICS_DIR / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md = markdown_report(table, ds, args.threshold, args.class_weight, chosen)
    md_path = config.METRICS_DIR / "metrics.md"
    md_path.write_text(md)
    print("\n" + md)

    summary = table[(table["setting"] == args.setting) & (table["model"] == model_name)]
    trained = modeling.train_final(
        ds,
        args.setting,
        model_name,
        args.class_weight,
        args.threshold,
        summary.iloc[0].to_dict(),
        result.checksum,
    )
    path = modeling.save_model(trained, config.MODEL_FILE)
    storage.record_model(
        model_name,
        args.setting,
        str(path.relative_to(config.ROOT_DIR)),
        trained.metadata,
        float(summary["roc_auc"].iloc[0]),
    )
    print(f"Saved {model_name} (setting {args.setting}) to {path}")
    print(f"Metrics: {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
