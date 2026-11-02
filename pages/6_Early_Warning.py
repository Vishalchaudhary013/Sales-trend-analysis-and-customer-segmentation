"""Early Warning: customers predicted to churn, with risk tiers and reasons."""

import pandas as pd
import streamlit as st

from src import config, modeling, storage, ui, visuals

ui.page_setup("Early Warning", "🚨")
page = ui.load_page_data()

model = ui.saved_model()
if model is None:
    st.warning("No churn model has been saved yet.")
    if st.button("Train and save the recommended model now", type="primary"):
        with st.spinner("Training…"):
            key = ui.dataset_key(page.active)
            ds = ui.churn_dataset(key, page.active.data)
            results = ui.evaluation(key, None, ds)
            table = modeling.results_table(results, ds.y)
            name = modeling.best_model(table, config.DEFAULT_CHURN_SETTING)
            summary = table[
                (table.setting == config.DEFAULT_CHURN_SETTING) & (table.model == name)
            ].iloc[0]
            trained = modeling.train_final(
                ds,
                config.DEFAULT_CHURN_SETTING,
                name,
                None,
                config.DEFAULT_THRESHOLD,
                summary.to_dict(),
                page.active.checksum,
            )
            path = modeling.save_model(trained)
            storage.record_model(
                name,
                config.DEFAULT_CHURN_SETTING,
                str(path.relative_to(config.ROOT_DIR)),
                trained.metadata,
                float(summary["roc_auc"]),
            )
        st.rerun()
    st.caption(
        "Or choose and save a model on the **Model Evaluation** page, or run "
        "`python scripts/train.py`."
    )
    st.stop()

md = model.metadata
threshold = float(md.get("threshold", config.DEFAULT_THRESHOLD))
st.caption(
    f"Model: **{md['model']}** · setting {md['setting']} · threshold {threshold:.2f} · trained "
    f"{md['trained_at']} · CV ROC-AUC {md.get('cv_metrics', {}).get('roc_auc', float('nan')):.3f}. "
    f"Probabilities estimate the chance a customer makes **no purchase in the next "
    f"{md['horizon_days']} days**, scored on each customer's full history up to "
    f"{page.active.end:%d %b %Y}."
)
if md.get("data_sha256") and page.active.checksum and md["data_sha256"] != page.active.checksum:
    st.warning(
        "This model was trained on a different dataset than the one now active. Retrain it "
        "on the Model Evaluation page before relying on these scores."
    )

with ui.friendly_errors("scoring customers"):
    missing = [f for f in model.features if f not in config.CHURN_FEATURES_FULL]
    if missing:
        st.error(f"The saved model expects unknown features: {missing}. Retrain the model.")
        st.stop()
    money_key = f"{page.money.code}|{page.money.mode}|{page.money.custom_rate}"
    scored = ui.score_and_explain(
        md["trained_at"],
        ui.dataset_key(page.active),
        money_key,
        model,
        page.active.data,
        page.cur,
        ui.money_factor(page),
    )
    if scored.empty:
        st.info("No identified customers to score.")
        st.stop()

    # Respect the global filters: keep customers present in the filtered scope.
    in_scope = set(page.scope.loc[page.scope[config.COL_HAS_CUSTOMER], config.COL_CUSTOMER])
    scored = scored[scored.index.isin(in_scope)]
    seg = ui.segment(page).customers
    scored = scored.join(seg[["segment"]], how="left")
    scored["predicted_churn"] = scored["churn_probability"] >= threshold
    factor = ui.money_factor(page)
    spend = scored["monetary"] * (
        factor.reindex(scored.index).fillna(1.0) if isinstance(factor, pd.Series) else factor
    )

    tiers = scored["risk_tier"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    run_key = (
        md["trained_at"],
        ui.dataset_key(page.active),
        threshold,
        page.filters.describe(),
        page.money.code,
    )
    if st.session_state.get("_last_logged_run") != run_key:
        storage.record_prediction_run(
            md["model"],
            md["setting"],
            threshold,
            len(scored),
            int(scored["predicted_churn"].sum()),
            tiers.to_dict(),
            page.active.checksum,
            page.money.code,
        )
        st.session_state["_last_logged_run"] = run_key

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers scored", f"{len(scored):,}")
    c2.metric(f"Predicted to churn (p ≥ {threshold:.2f})", f"{int(scored.predicted_churn.sum()):,}")
    c3.metric("🔴 High risk (≥ 0.70)", f"{tiers['High']:,}")
    c4.metric("🟠 Medium risk (0.40–0.70)", f"{tiers['Medium']:,}")
    c5.metric("🔵 Low risk (< 0.40)", f"{tiers['Low']:,}")
    at_risk_spend = spend[scored["predicted_churn"]].sum()
    st.caption(
        f"Customers predicted to churn spent {ui.fmt_money(at_risk_spend, cur=page.cur)} "
        "in total so far (completed sales)."
    )

    left, right = st.columns([1, 2])
    with left:
        mix = pd.DataFrame({"label": [f"{t} risk" for t in tiers.index], "value": tiers.values})
        mix = mix[mix["value"] > 0]
        mix["share"] = mix["value"] / mix["value"].sum()
        colors = [config.RISK_COLORS[t.split()[0]] for t in mix["label"]]
        st.plotly_chart(
            visuals.fig_donut(
                mix,
                "Customers by risk tier",
                "count",
                f"{len(scored):,}",
                colors=colors,
                legend_below=True,
            ),
            width="stretch",
            theme=None,
            key="ew_donut",
        )
    with right:
        st.markdown("**Find customers**")
        f1, f2 = st.columns(2)
        query = f1.text_input("Search CustomerID", key="ew_q").strip()
        tier_pick = f2.multiselect(
            "Risk tier", ["High", "Medium", "Low"], default=["High", "Medium"], key="ew_tiers"
        )
        only_pred = st.toggle(
            f"Only customers predicted to churn (p ≥ {threshold:.2f})", value=True, key="ew_only"
        )
        st.caption(
            "Reasons show the features that raise each customer's score most (the model's "
            "view, not proven causes)."
        )

    view = scored
    if query:
        view = view[view.index.astype(str).str.contains(query, regex=False)]
    if tier_pick:
        view = view[view["risk_tier"].isin(tier_pick)]
    if only_pred:
        view = view[view["predicted_churn"]]
    table = pd.DataFrame(
        {
            "CustomerID": view.index,
            "Churn probability": view["churn_probability"].round(3).to_numpy(),
            "Risk tier": view["risk_tier"].to_numpy(),
            "Reason 1": view["reason_1"].to_numpy(),
            "Reason 2": view["reason_2"].to_numpy(),
            "Reason 3": view["reason_3"].to_numpy(),
            "RFM segment": view["segment"].to_numpy(),
            "Days since last purchase": view["recency_days"].astype(int).to_numpy(),
            "Orders": view["frequency"].astype(int).to_numpy(),
            f"Total spend ({page.money.code})": spend.reindex(view.index).round(2).to_numpy(),
        }
    )
    st.caption(f"{len(table):,} of {len(scored):,} customers shown")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=520,
        column_config={
            "Churn probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )
    st.download_button(
        "Download this list (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="churn_early_warning.csv",
        mime="text/csv",
    )

with st.expander("Prediction run log (SQLite)"):
    st.dataframe(storage.prediction_history(), hide_index=True, width="stretch")
