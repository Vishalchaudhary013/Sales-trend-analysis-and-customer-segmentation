"""Churn early-warning model: dataset construction, cross-validation, persistence.

Problem definition (positive class = **churned**):

* A *cut-off* date splits history: cut-off = last invoice day − horizon
  (:data:`config.CHURN_HORIZON_DAYS`).
* Customers with at least one completed purchase before the cut-off form the
  training population. Features use **only** data before the cut-off.
* ``churned = 1`` if the customer makes no completed purchase in the horizon
  after the cut-off, else 0.

At inference time the same feature code runs with the cut-off at the end of
the data, scoring current customers for churn over the next horizon.

Validation: StratifiedKFold(5, shuffle=True, random_state=42); identical folds
for every model and feature setting. Metrics are computed on out-of-fold (OOF)
predictions; a majority-class baseline is fitted inside each training fold.
Preprocessing lives inside scikit-learn Pipelines, so it is fitted on training
folds only (no leakage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src import analytics, config

C = config
TARGET = "churned"


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #


def completed_customer_sales(data: pd.DataFrame) -> pd.DataFrame:
    """Completed-sales lines of identified customers (base currency)."""
    lines = data.loc[analytics.line_mask(data, analytics.Filters())]
    return lines.loc[lines[C.COL_HAS_CUSTOMER]]


def build_features(data: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Compute customer features from all activity strictly before ``cutoff``.

    Args:
        data: Clean transaction table (all line types; cancellations are used
            for the cancellation rate only).
        cutoff: Features use rows with ``InvoiceDate < cutoff``.

    Returns:
        One row per customer with a completed purchase before ``cutoff``,
        indexed by ``CustomerID``, with the columns of
        :data:`config.CHURN_FEATURES_FULL`. ``mean_days_between_orders`` is NaN
        for single-order customers (imputed inside the model pipeline).
    """
    hist = data.loc[data[C.COL_DATE] < cutoff]
    sales = completed_customer_sales(hist)
    if sales.empty:
        return pd.DataFrame(columns=list(C.CHURN_FEATURES_FULL)).rename_axis(C.COL_CUSTOMER)
    g = sales.groupby(C.COL_CUSTOMER)
    day = cutoff.normalize()
    orders = sales.drop_duplicates(C.COL_INVOICE)[[C.COL_CUSTOMER, C.COL_DATE]]
    feats = pd.DataFrame(
        {
            "recency_days": (day - g[C.COL_DATE].max().dt.normalize()).dt.days,
            "frequency": g[C.COL_INVOICE].nunique(),
            "monetary": g[C.COL_REVENUE].sum(),
            "tenure_days": (day - g[C.COL_DATE].min().dt.normalize()).dt.days,
            "distinct_products": g[C.COL_STOCK].nunique(),
            "lines": g.size(),
        }
    )
    feats["avg_order_value"] = feats["monetary"] / feats["frequency"]
    feats["lines_per_order"] = feats["lines"] / feats["frequency"]
    window_start = cutoff - pd.Timedelta(days=C.CHURN_RECENT_WINDOW_DAYS)
    recent = sales.loc[sales[C.COL_DATE] >= window_start].groupby(C.COL_CUSTOMER)
    feats["orders_last_90d"] = recent[C.COL_INVOICE].nunique().reindex(feats.index, fill_value=0)
    feats["revenue_last_90d"] = recent[C.COL_REVENUE].sum().reindex(feats.index, fill_value=0.0)
    gaps = (
        orders.sort_values(C.COL_DATE)
        .groupby(C.COL_CUSTOMER)[C.COL_DATE]
        .apply(lambda s: s.diff().dt.total_seconds().div(86400).mean() if len(s) > 1 else np.nan)
    )
    feats["mean_days_between_orders"] = gaps.reindex(feats.index)
    invoices = hist.loc[hist[C.COL_HAS_CUSTOMER] & ~hist[C.COL_IS_ADJUST]].drop_duplicates(
        C.COL_INVOICE
    )
    canc = invoices.groupby(C.COL_CUSTOMER)[C.COL_IS_CANCEL].mean()
    feats["cancellation_rate"] = canc.reindex(feats.index).fillna(0.0)
    if C.COL_COUNTRY in sales.columns:
        country = g[C.COL_COUNTRY].agg(lambda s: s.mode().iat[0] if s.notna().any() else "")
        feats["is_uk"] = (country == "United Kingdom").astype(int)
    else:
        feats["is_uk"] = 0
    return feats[list(C.CHURN_FEATURES_FULL)].astype(float)


@dataclass
class ChurnDataset:
    """Training table for the churn model.

    Attributes:
        X: Features (index = CustomerID).
        y: ``churned`` labels aligned to ``X``.
        cutoff: Feature cut-off date.
        horizon_end: End of the label window (last invoice timestamp).
    """

    X: pd.DataFrame  # noqa: N815 - conventional ML naming
    y: pd.Series
    cutoff: pd.Timestamp
    horizon_end: pd.Timestamp

    @property
    def churn_rate(self) -> float:
        """Share of customers labelled churned."""
        return float(self.y.mean()) if len(self.y) else float("nan")


def build_dataset(data: pd.DataFrame, horizon_days: int = C.CHURN_HORIZON_DAYS) -> ChurnDataset:
    """Build features (before the cut-off) and churn labels (after it).

    Raises:
        ValueError: If the history is shorter than twice the horizon, or no
            customers / only one class remain.
    """
    end = data[C.COL_DATE].max()
    start = data[C.COL_DATE].min()
    if (end - start).days < 2 * horizon_days:
        raise ValueError(
            f"History spans {(end - start).days} days; at least {2 * horizon_days} are needed "
            f"for a {horizon_days}-day churn horizon."
        )
    cutoff = end.normalize() - pd.Timedelta(days=horizon_days)
    X = build_features(data, cutoff)  # noqa: N806
    future = completed_customer_sales(data.loc[data[C.COL_DATE] >= cutoff])
    active = set(future[C.COL_CUSTOMER])
    y = pd.Series([int(c not in active) for c in X.index], index=X.index, name=TARGET)
    if len(X) < C.CV_FOLDS * 2 or y.nunique() < 2:
        raise ValueError("Not enough customers or only one outcome class to train a model.")
    return ChurnDataset(X, y, cutoff, end)


def feature_list(setting: str) -> list[str]:
    """Return the feature names of setting ``"A"`` (RFM) or ``"B"`` (full)."""
    if setting == "A":
        return list(C.CHURN_FEATURES_RFM)
    if setting == "B":
        return list(C.CHURN_FEATURES_FULL)
    raise ValueError(f"Unknown feature setting '{setting}'.")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


def make_model(name: str, class_weight: str | None = None) -> Pipeline:
    """Return an unfitted pipeline: impute → log1p [→ scale] → estimator.

    * Median imputation: only ``mean_days_between_orders`` has gaps (customers
      with a single order). It is fitted on training folds only.
    * ``log1p``: all features are non-negative counts, days, money or rates
      with long right tails; the transform stops extreme spenders dominating
      linear models. Trees are invariant to it.
    * Scaling for Logistic Regression only (coefficients are then comparable).
    """
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ]
    if name == "Logistic Regression":
        est: Any = LogisticRegression(
            max_iter=C.LR_MAX_ITER, class_weight=class_weight, random_state=C.RANDOM_STATE
        )
        steps.append(("scale", StandardScaler()))
    elif name == "Decision Tree":
        est = DecisionTreeClassifier(
            max_depth=C.DT_MAX_DEPTH, class_weight=class_weight, random_state=C.RANDOM_STATE
        )
    elif name == "Random Forest":
        est = RandomForestClassifier(
            n_estimators=C.RF_N_ESTIMATORS,
            class_weight=class_weight,
            random_state=C.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model '{name}'.")
    steps.append(("model", est))
    return Pipeline(steps)


def make_folds(y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the shared StratifiedKFold splits (identical for all models/settings)."""
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=C.CV_SHUFFLE, random_state=C.RANDOM_STATE)
    return list(skf.split(np.zeros(len(y)), y))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "retained_recall",
)


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = C.DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Metrics with churned (1) as the positive class.

    ``recall`` is churn-class recall (share of churners caught);
    ``retained_recall`` is the recall of the retained class (specificity).
    ROC-AUC / PR-AUC are NaN when only one class is present.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else float("nan"),
        "retained_recall": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
    }


@dataclass
class CVResult:
    """Cross-validated evaluation of one model under one feature setting.

    Attributes:
        model: Model name (or ``"Baseline (majority class)"``).
        setting: Feature setting ``"A"`` / ``"B"``.
        oof_proba: Out-of-fold churn probabilities aligned to the dataset.
        fold_metrics: One row of metrics per fold (at the default threshold).
    """

    model: str
    setting: str
    oof_proba: pd.Series
    fold_metrics: pd.DataFrame

    def summary(self, y: pd.Series, threshold: float = C.DEFAULT_THRESHOLD) -> dict[str, Any]:
        """Pooled OOF metrics at ``threshold`` plus fold mean ± std at 0.5."""
        pooled = classification_metrics(y.to_numpy(), self.oof_proba.to_numpy(), threshold)
        out: dict[str, Any] = {"model": self.model, "setting": self.setting}
        out.update(pooled)
        for m in METRIC_NAMES:
            out[f"{m}_fold_mean"] = float(self.fold_metrics[m].mean())
            out[f"{m}_fold_std"] = float(self.fold_metrics[m].std(ddof=1))
        return out


BASELINE = "Baseline (majority class)"


def cross_validate(
    ds: ChurnDataset,
    setting: str,
    model: str,
    class_weight: str | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> CVResult:
    """Evaluate ``model`` (or :data:`BASELINE`) with the shared folds.

    The baseline is a ``DummyClassifier(strategy="most_frequent")`` fitted on
    each training fold; its "probability" is the training-fold churn share
    when churn is the majority, else 0, so thresholded predictions equal the
    majority class.
    """
    folds = folds or make_folds(ds.y)
    X = ds.X[feature_list(setting)]  # noqa: N806
    y = ds.y
    oof = pd.Series(np.nan, index=X.index, name="proba")
    rows = []
    for i, (tr, te) in enumerate(folds, start=1):
        if model == BASELINE:
            dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y.iloc[tr])
            majority = int(dummy.predict(X.iloc[te[:1]])[0])
            proba = np.full(len(te), 1.0 if majority == 1 else 0.0)
        else:
            pipe = make_model(model, class_weight).fit(X.iloc[tr], y.iloc[tr])
            proba = pipe.predict_proba(X.iloc[te])[:, 1]
        oof.iloc[te] = proba
        rows.append({"fold": i, **classification_metrics(y.iloc[te].to_numpy(), proba)})
    return CVResult(model, setting, oof, pd.DataFrame(rows))


def evaluate_all(
    ds: ChurnDataset, class_weight: str | None = None, settings: tuple[str, ...] = ("A", "B")
) -> dict[tuple[str, str], CVResult]:
    """Cross-validate the baseline and all models for each setting (same folds)."""
    folds = make_folds(ds.y)
    results = {}
    for setting in settings:
        for model in (BASELINE, *C.MODEL_NAMES):
            results[(setting, model)] = cross_validate(ds, setting, model, class_weight, folds)
    return results


def results_table(
    results: dict[tuple[str, str], CVResult], y: pd.Series, threshold: float = C.DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """One row per (setting, model): pooled OOF metrics and fold mean ± std."""
    return pd.DataFrame([r.summary(y, threshold) for r in results.values()])


def threshold_curve(
    y: pd.Series, proba: pd.Series, thresholds: np.ndarray | None = None
) -> pd.DataFrame:
    """Precision, churn recall and retained recall across decision thresholds.

    Precision is NaN at thresholds where no customer is flagged.
    """
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2) if thresholds is None else thresholds
    rows = []
    for t in ts:
        m = classification_metrics(y.to_numpy(), proba.to_numpy(), float(t))
        if not (proba >= t).any():  # nothing flagged: precision is undefined, not zero
            m["precision"] = float("nan")
        rows.append(
            {
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "retained_recall": m["retained_recall"],
                "flagged_share": float((proba >= t).mean()),
            }
        )
    return pd.DataFrame(rows)


def roc_points(y: pd.Series, proba: pd.Series) -> pd.DataFrame:
    """False/true positive rates for a ROC curve."""
    fpr, tpr, thr = roc_curve(y, proba)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})


def confusion(y: pd.Series, proba: pd.Series, threshold: float) -> pd.DataFrame:
    """2×2 confusion matrix (rows = actual, columns = predicted) as counts."""
    cm = confusion_matrix(y, (proba >= threshold).astype(int), labels=[0, 1])
    labels = ["Retained", "Churned"]
    return pd.DataFrame(
        cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels]
    )


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #


def feature_importance(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    """Return importances (tree models) or standardised coefficients (LR).

    Returns:
        Columns ``feature, label, value, kind``; for LR, ``value`` is the
        coefficient on the standardised ``log1p`` feature (positive = raises
        churn odds).
    """
    est = pipe["model"]
    if hasattr(est, "feature_importances_"):
        values, kind = est.feature_importances_, "importance"
    else:
        values, kind = est.coef_[0], "coefficient (standardised log feature)"
    out = pd.DataFrame({"feature": features, "value": values, "kind": kind})
    out["label"] = out["feature"].map(C.FEATURE_LABELS)
    order = out["value"].abs().sort_values(ascending=False).index
    return out.loc[order, ["feature", "label", "value", "kind"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Final model, persistence and inference
# --------------------------------------------------------------------------- #


@dataclass
class TrainedModel:
    """A model fitted on the full training set, with metadata.

    Attributes:
        pipeline: Fitted pipeline.
        features: Feature names in order.
        metadata: Setting, parameters, metrics, validation strategy, dates,
            library versions and training timestamp.
        reference: Training feature matrix (used for reason codes).
    """

    pipeline: Pipeline
    features: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    reference: pd.DataFrame | None = None


def train_final(
    ds: ChurnDataset,
    setting: str,
    model: str,
    class_weight: str | None,
    threshold: float,
    cv_summary: dict[str, Any] | None = None,
    data_checksum: str | None = None,
) -> TrainedModel:
    """Fit ``model`` on every training customer and attach metadata."""
    features = feature_list(setting)
    pipe = make_model(model, class_weight).fit(ds.X[features], ds.y)
    meta = {
        "model": model,
        "setting": setting,
        "setting_description": C.CHURN_SETTINGS[setting],
        "features": features,
        "params": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in pipe["model"].get_params().items()
        },
        "preprocessing": [name for name, _ in pipe.steps[:-1]],
        "class_weight": class_weight,
        "threshold": threshold,
        "positive_class": "churned (no completed purchase within the horizon)",
        "horizon_days": C.CHURN_HORIZON_DAYS,
        "cutoff": ds.cutoff.isoformat(),
        "label_window_end": ds.horizon_end.isoformat(),
        "n_customers": int(len(ds.y)),
        "churn_rate": ds.churn_rate,
        "validation": {
            "strategy": "StratifiedKFold",
            "n_splits": C.CV_FOLDS,
            "shuffle": C.CV_SHUFFLE,
            "random_state": C.RANDOM_STATE,
            "metrics_from": "out-of-fold predictions",
        },
        "cv_metrics": cv_summary or {},
        "data_sha256": data_checksum,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    return TrainedModel(pipe, features, meta, ds.X[features].copy())


def save_model(model: TrainedModel, path: Path = C.MODEL_FILE) -> Path:
    """Persist the model with joblib plus a JSON copy of its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    path.with_suffix(".json").write_text(json.dumps(model.metadata, indent=2, default=str))
    return path


def model_status(path: Path = C.MODEL_FILE) -> tuple[str, str]:
    """Check a saved model without unpickling it.

    Returns:
        ``("ok", "")``, ``("missing", reason)`` or ``("incompatible", reason)``.
        Pickled scikit-learn models are only reliable with the version that
        saved them, so a version mismatch (from the JSON metadata) is
        reported as incompatible and the model must be retrained.
    """
    if not path.exists():
        return "missing", "No model has been saved yet."
    meta_path = path.with_suffix(".json")
    try:
        saved = json.loads(meta_path.read_text())["versions"]["scikit-learn"]
    except (OSError, ValueError, KeyError, TypeError):
        return "incompatible", "The saved model has no readable metadata; retrain it."
    if saved != sklearn.__version__:
        return "incompatible", (
            f"The saved model was trained with scikit-learn {saved}, but this environment runs "
            f"{sklearn.__version__}. Retrain it here."
        )
    return "ok", ""


def load_model(path: Path = C.MODEL_FILE) -> TrainedModel | None:
    """Load a persisted model, or ``None`` if absent, incompatible or unreadable."""
    if model_status(path)[0] != "ok":
        return None
    try:
        model = joblib.load(path)
    except Exception:  # noqa: BLE001 - stale/corrupt artefacts are treated as missing
        return None
    return model if isinstance(model, TrainedModel) else None


def score_current(model: TrainedModel, data: pd.DataFrame) -> pd.DataFrame:
    """Score every customer as of the end of the data.

    Features are computed with the cut-off = day after the last invoice, so
    each customer's full history is used; probabilities refer to churn over the
    next :data:`config.CHURN_HORIZON_DAYS` days.

    Returns:
        Feature columns plus ``churn_probability`` and ``risk_tier``.
    """
    cutoff = data[C.COL_DATE].max().normalize() + pd.Timedelta(days=1)
    X = build_features(data, cutoff)  # noqa: N806
    if X.empty:
        return X.assign(churn_probability=pd.Series(dtype=float), risk_tier=pd.Series(dtype=str))
    proba = model.pipeline.predict_proba(X[model.features])[:, 1]
    out = X.copy()
    out["churn_probability"] = proba
    out["risk_tier"] = [risk_tier(p) for p in proba]
    return out.sort_values("churn_probability", ascending=False)


def best_model(table: pd.DataFrame, setting: str) -> str:
    """Return the model with the highest fold-mean ROC-AUC for ``setting``."""
    rows = table[(table["setting"] == setting) & (table["model"] != BASELINE)]
    if rows.empty:
        return C.DEFAULT_MODEL
    return str(rows.sort_values("roc_auc_fold_mean", ascending=False)["model"].iloc[0])


def risk_tier(p: float) -> str:
    """High ≥ 0.7, Medium 0.4–0.7, Low < 0.4 (thresholds in config)."""
    if p >= C.RISK_HIGH:
        return "High"
    if p >= C.RISK_MEDIUM:
        return "Medium"
    return "Low"
