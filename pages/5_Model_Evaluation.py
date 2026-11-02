"""Model Evaluation: churn models for settings A and B against the baseline."""

import pandas as pd
import streamlit as st

from src import config, modeling, storage, ui, visuals

ui.page_setup("Model Evaluation", "🧪")
active = ui.get_active_dataset()
if active is None:
    ui.show_missing_data_help()
    st.stop()
ui.sidebar_currency()  # keep the currency choice visible/persistent on every page

st.caption(
    "Models are trained on the **whole active dataset** (sidebar filters do not apply here) so "
    "that results are reproducible. Positive class = **churned**: no completed purchase in the "
    f"{config.CHURN_HORIZON_DAYS} days after the feature cut-off."
)

key = ui.dataset_key(active)
try:
    ds = ui.churn_dataset(key, active.data)
except ValueError as exc:
    st.error(f"The churn model cannot be trained on this dataset: {exc}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers in training set", f"{len(ds.y):,}")
c2.metric("Churn rate", ui.fmt_pct(ds.churn_rate))
c3.metric("Feature cut-off", f"{ds.cutoff:%d %b %Y}")
c4.metric("Label window ends", f"{ds.horizon_end:%d %b %Y}")

with st.container(border=True):
    k1, k2 = st.columns([1, 2])
    balanced = k1.toggle(
        'class_weight = "balanced"',
        key="me_balanced",
        help="Re-weights classes inversely to their frequency during training.",
    )
    threshold = k2.slider(
        "Decision threshold (predict churn if probability ≥ threshold)",
        0.05,
        0.95,
        config.DEFAULT_THRESHOLD,
        0.01,
        key="me_threshold",
    )
class_weight = "balanced" if balanced else None

with ui.friendly_errors("evaluating the models"):
    results = ui.evaluation(key, class_weight, ds)
    table = modeling.results_table(results, ds.y, threshold)

    st.subheader("Settings A and B side by side")
    st.caption(
        f"Pooled out-of-fold metrics at threshold {threshold:.2f}; ± = standard deviation across "
        f"the {config.CV_FOLDS} folds (at 0.50). Identical folds for every model and setting. "
        "Recall (churn) = share of churners caught; Recall (retained) = share of loyal customers "
        "correctly left alone. The baseline always predicts the training fold's majority class."
    )
    for setting, desc in config.CHURN_SETTINGS.items():
        with st.container():
            st.markdown(f"**Setting {setting}** — {desc}")
            rows = table[table["setting"] == setting].set_index("model")
            show = pd.DataFrame(
                {
                    visuals.METRIC_DISPLAY[m]: [
                        f"{rows.at[i, m]:.3f} ± {rows.at[i, m + '_fold_std']:.3f}"
                        for i in rows.index
                    ]
                    for m in modeling.METRIC_NAMES
                },
                index=rows.index,
            )
            st.dataframe(show, width="stretch")
    metric = st.selectbox(
        "Compare models on",
        list(visuals.METRIC_DISPLAY),
        index=list(visuals.METRIC_DISPLAY).index("roc_auc"),
        format_func=visuals.METRIC_DISPLAY.get,
        key="me_metric",
    )
    st.plotly_chart(
        visuals.fig_model_comparison(table, metric), width="stretch", theme=None, key="me_compare"
    )

    st.divider()
    st.subheader("Inspect one model")
    s1, s2 = st.columns(2)
    setting = s1.radio(
        "Feature setting",
        list(config.CHURN_SETTINGS),
        horizontal=True,
        index=list(config.CHURN_SETTINGS).index(config.DEFAULT_CHURN_SETTING),
        format_func=lambda s: f"{s} — {config.CHURN_SETTINGS[s].split('(')[0]}",
        key="me_setting",
    )
    best = modeling.best_model(table, setting)
    model_name = s2.selectbox(
        "Model",
        list(config.MODEL_NAMES),
        index=list(config.MODEL_NAMES).index(best),
        key="me_model",
        help=f"Default = best fold-mean ROC-AUC ({best}).",
    )
    res = results[(setting, model_name)]
    left, right = st.columns([3, 2])
    with left:
        curve = modeling.threshold_curve(ds.y, res.oof_proba)
        st.plotly_chart(
            visuals.fig_threshold_tradeoff(curve, threshold),
            width="stretch",
            theme=None,
            key="me_tradeoff",
        )
    with right:
        cm = modeling.confusion(ds.y, res.oof_proba, threshold)
        st.plotly_chart(visuals.fig_confusion(cm), width="stretch", theme=None, key="me_cm")
        m = modeling.classification_metrics(ds.y.to_numpy(), res.oof_proba.to_numpy(), threshold)
        st.markdown(
            f"At **{threshold:.2f}**: catches **{m['recall']:.1%}** of churners; "
            f"**{m['precision']:.1%}** of flagged customers actually churn; "
            f"**{m['retained_recall']:.1%}** of retained customers are correctly not flagged."
        )
    curves = {
        n: modeling.roc_points(ds.y, results[(setting, n)].oof_proba) for n in config.MODEL_NAMES
    }
    aucs = {
        n: float(table[(table.setting == setting) & (table.model == n)]["roc_auc"].iloc[0])
        for n in config.MODEL_NAMES
    }
    st.plotly_chart(visuals.fig_roc(curves, aucs), width="stretch", theme=None, key="me_roc")

    st.divider()
    st.subheader("What drives the predictions")
    st.caption(
        "Fitted on all training customers for the selected setting. Associations, not "
        "causes; correlated features share and can swap credit."
    )
    i1, i2 = st.columns(2)
    feats = modeling.feature_list(setting)
    rf = modeling.make_model("Random Forest", class_weight).fit(ds.X[feats], ds.y)
    lr = modeling.make_model("Logistic Regression", class_weight).fit(ds.X[feats], ds.y)
    with i1:
        st.plotly_chart(
            visuals.fig_importance(
                modeling.feature_importance(rf, feats), "Random Forest feature importance"
            ),
            width="stretch",
            theme=None,
            key="me_rf_imp",
        )
    with i2:
        st.plotly_chart(
            visuals.fig_importance(
                modeling.feature_importance(lr, feats),
                "Logistic Regression standardised coefficients",
            ),
            width="stretch",
            theme=None,
            key="me_lr_coef",
        )

    st.divider()
    st.subheader("Save the selected model for Early Warning")
    current = ui.saved_model()
    if current is not None:
        md = current.metadata
        st.caption(
            f"Currently saved: **{md['model']}** (setting {md['setting']}, threshold "
            f"{md['threshold']}, class_weight {md['class_weight']}), trained "
            f"{md['trained_at']}."
        )
    if st.button(
        f"Save {model_name} · setting {setting} · threshold {threshold:.2f}", type="primary"
    ):
        summary = table[(table.setting == setting) & (table.model == model_name)].iloc[0]
        trained = modeling.train_final(
            ds, setting, model_name, class_weight, threshold, summary.to_dict(), active.checksum
        )
        path = modeling.save_model(trained)
        storage.record_model(
            model_name,
            setting,
            str(path.relative_to(config.ROOT_DIR)),
            trained.metadata,
            float(summary["roc_auc"]),
        )
        st.success(
            f"Saved to `{path.relative_to(config.ROOT_DIR)}` with metadata "
            "(setting, parameters, features, preprocessing, validation, metrics, "
            "timestamp)."
        )
    with st.expander("Saved-model history"):
        hist = storage.model_history()
        st.dataframe(
            hist.drop(columns=["metadata_json"]) if not hist.empty else hist,
            hide_index=True,
            width="stretch",
        )
