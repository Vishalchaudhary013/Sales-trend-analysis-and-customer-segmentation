"""Methodology: data source, pipeline, definitions and limitations."""

import json

import streamlit as st

from src import config, ingestion, ui

ui.page_setup("Methodology", "📖")
active = ui.get_active_dataset()
ui.sidebar_currency()

st.header("Data")
st.markdown(
    f"**Source.** {config.DATASET_NAME}: all transactions of a UK-based online retailer of "
    "giftware (many customers are wholesalers), 1 Dec 2010 – 9 Dec 2011.\n\n"
    f"**Citation.** {config.DATASET_CITATION}\n\n"
    f"**Download.** {config.DATASET_PAGE_URL}. When that host is unreachable the fetch script "
    f"uses a public copy of the same file and says so: {config.DATASET_MIRROR_URL}."
)
if active is not None:
    s = ingestion.data_quality_summary(active.data)
    st.markdown(
        f"**Loaded now ({active.label}).** {s['rows']:,} clean lines, {s['invoices']:,} invoices, "
        f"{s['customers']:,} customers, {s['countries']} countries; "
        f"{s['share_without_customer']:.1%} of lines have no CustomerID (guest sales)."
    )

st.header("Pipeline")
st.markdown(
    "1. **Ingestion & validation** (`src/ingestion.py`): CSV/XLSX read with codes kept as text; "
    "column names mapped only through documented aliases; date format detected (ambiguous "
    "day/month orders are reported); unparseable rows dropped *and counted*; nothing imputed.\n"
    "2. **Cleaning & flags**: exact duplicate rows removed; cancellations (invoice prefix `C`), "
    "accounting adjustments (`A`), zero/negative prices, stock write-offs, postage/fee codes and "
    "guest lines are **flagged, not deleted**. Sales exactly reversed by a later cancellation "
    "(same customer, product and quantity) are flagged `IsReversed`.\n"
    "3. **Analytics** (`src/analytics.py`): *completed sales* = product lines with positive "
    "quantity and price that are not cancellations, adjustments or reversed. 'Net of returns' "
    "nets every line instead. Partial periods at the edges of the data are marked and excluded "
    "from growth rates.\n"
    "4. **Currency** (`src/currency.py`): source amounts are GBP. Other currencies use published "
    f"daily rates — {config.FX_SOURCE} — at each invoice date (or one period-average rate); "
    "custom currencies use a user-supplied rate that is labelled as such.\n"
    "5. **Segmentation** (`src/segmentation.py`): RFM per identified customer from completed "
    "sales; quintile scores; 10 rule-based segments; K-Means on log-transformed, standardised "
    "R, F, M (seed 42) with k chosen by the Kneedle elbow (silhouette and Davies–Bouldin "
    "reported). Segments are always computed in GBP so a currency switch never moves a customer.\n"
    f"6. **Churn model** (`src/modeling.py`): cut-off = last invoice day − "
    f"{config.CHURN_HORIZON_DAYS} days. Features use only data before the cut-off; churned = no "
    "completed purchase in the following window. Setting A = RFM only; setting B = full "
    "behaviour. Logistic Regression, Decision Tree (max_depth 5) and Random Forest (300 trees) in "
    "Pipelines (median imputation → log1p → [scaling]); StratifiedKFold(5, shuffle, seed 42) "
    "with identical folds for every model and setting; metrics from out-of-fold predictions; a "
    "majority-class baseline fitted in each training fold.\n"
    "7. **Early warning** (`src/explain.py`): the saved model scores each customer on their full "
    "history. Reasons use *occlusion*: each feature is set to the training median and the drop in "
    "probability is its contribution; the top three risk-raising features are phrased with the "
    "customer's value and percentile.\n"
    "8. **Persistence** (`src/storage.py`): SQLite stores upload metadata (file name, rows, "
    "SHA-256), saved-model metadata and each prediction run — never transaction rows."
)

st.header("Metric definitions")
st.markdown(
    "Positive class = **churned**.\n\n"
    "| Metric | Meaning |\n|---|---|\n"
    "| Accuracy | Share of all customers classified correctly |\n"
    "| Precision (churn) | Of customers flagged as churners, the share who did churn |\n"
    "| Recall (churn) | Of customers who churned, the share the model flagged |\n"
    "| F1 (churn) | Harmonic mean of churn precision and recall |\n"
    "| Recall (retained) | Of customers who stayed, the share correctly not flagged |\n"
    "| ROC-AUC | Probability a random churner scores above a random retained customer "
    "(0.5 = chance) |\n"
    "| PR-AUC | Average precision across thresholds (baseline = churn rate) |\n"
    "| Silhouette | Cluster separation, −1 to 1 (higher = better) |\n"
    "| Davies–Bouldin | Cluster compactness vs separation (lower = better) |\n\n"
    "All model metrics are compared with the **majority-class baseline**. Risk tiers: High "
    f"≥ {config.RISK_HIGH}, Medium {config.RISK_MEDIUM}–{config.RISK_HIGH}, Low "
    f"< {config.RISK_MEDIUM}."
)
metrics_file = config.METRICS_DIR / "metrics.json"
if metrics_file.exists():
    payload = json.loads(metrics_file.read_text())
    sel = payload.get("selected", {})
    best = [
        r
        for r in payload["results"]
        if r["model"] == sel.get("model") and r["setting"] == sel.get("setting")
    ]
    if best:
        r = best[0]
        st.info(
            f"Latest `scripts/train.py` run: {sel['model']} (setting {sel['setting']}) — "
            f"ROC-AUC {r['roc_auc']:.3f} ± {r['roc_auc_fold_std']:.3f}, churn recall "
            f"{r['recall']:.3f}, precision {r['precision']:.3f} at threshold "
            f"{payload['context']['threshold']}. Full table: `reports/metrics/metrics.md`."
        )

st.header("Limitations")
st.markdown(
    "- **One retailer, one year.** A UK online giftware wholesaler/retailer, Dec 2010 – Dec 2011; "
    "results may not transfer to other businesses or periods. The 2011 Christmas peak is only "
    "partly observed (data end 9 Dec 2011).\n"
    "- **Guest sales** (about a quarter of lines) have no CustomerID and are excluded from "
    "segmentation and churn modelling.\n"
    "- **Churn is defined by inactivity** in a 90-day window, which suits regular buyers but "
    "mislabels seasonal or one-off buyers; a single cut-off means one label window.\n"
    "- **Correlation is not causation.** Feature importances and reasons describe the model's "
    "associations; correlated features share credit.\n"
    "- **Class balance.** About two-fifths of training customers churn; the threshold slider "
    "shows the precision/recall trade-off, and class weighting is available.\n"
    "- **Data quality.** Cancellations are matched to sales only when customer, product and "
    "quantity agree exactly; partial returns are netted only in 'Net of returns' mode.\n"
    "- **Currency.** Conversions use Federal Reserve noon rates; actual settlement rates differ.\n"
    "- **Privacy and fairness.** Customer IDs are pseudonymous but still personal data. The "
    "preview masks IDs by default, SQLite stores only metadata, and exports should be shared on a "
    "need-to-know basis. Country is used as a feature (UK vs other); scores should support, not "
    "replace, human judgement, and outreach should not disadvantage customers by location."
)
