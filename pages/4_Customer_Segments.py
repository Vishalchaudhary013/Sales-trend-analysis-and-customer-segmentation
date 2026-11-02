"""Customer Segments: RFM scoring, named segments and K-Means clusters."""

import streamlit as st

from src import config, segmentation, ui, visuals


def pct_columns(df):
    """Show *_share columns as percentages for readability."""
    out = df.copy()
    for col in [c for c in out.columns if c.endswith("_share")]:
        out[col] = out[col].map(lambda v: f"{v:.1%}")
    return out


ui.page_setup("Customer Segments", "👥")
page = ui.load_page_data()

base = ui.segment(page)
if base.customers.empty:
    st.info(
        "No identified customers in the current view, so customers cannot be segmented. "
        "Widen the filters or choose 'All customers' / 'Identified only'."
    )
    st.stop()

st.caption(
    f"{len(base.customers):,} identified customers · RFM snapshot date "
    f"{base.snapshot:%d %b %Y} · based on completed sales (returns and fees excluded) "
    "within the selected dates, countries and customer type."
)

result = base  # replaced by the user's k in the clusters tab
prof = base.clusters.profile if base.clusters is not None else None

tab_rfm, tab_km, tab_lookup = st.tabs(["RFM segments", "K-Means clusters", "Customer lookup"])

with tab_rfm, ui.friendly_errors("building the RFM views"):
    c1, c2, c3 = st.columns(3)
    champions = base.summary.set_index("segment")
    if "Champions" in champions.index:
        ch = champions.loc["Champions"]
        c1.metric(
            "Champions",
            f"{int(ch['customers']):,}",
            help="Recent and frequent buyers (R = 5, F ≥ 4).",
        )
        c2.metric("Champions' share of revenue", ui.fmt_pct(ch["revenue_share"]))
    at_risk = [s for s in ("At risk", "Can't lose them") if s in champions.index]
    if at_risk:
        c3.metric(
            "Customers at risk (At risk + Can't lose them)",
            f"{int(champions.loc[at_risk, 'customers'].sum()):,}",
            help="Previously frequent buyers who have not purchased recently.",
        )
    st.plotly_chart(
        visuals.fig_segment_bars(base.summary), width="stretch", theme=None, key="seg_bars"
    )
    st.caption("Segments come from the Recency and Frequency scores (quintiles, 5 = best).")
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(
            visuals.fig_rf_grid(segmentation.rf_grid(base.customers)),
            width="stretch",
            theme=None,
            key="rf_grid",
        )
    with right:
        st.markdown("**Recommended actions**")
        for _, row in base.summary.iterrows():
            st.markdown(f"- **{row['segment']}** ({int(row['customers']):,}): {row['action']}")
    with st.expander("Segment summary table"):
        table = base.summary.rename(
            columns={
                "mean_monetary": f"mean_spend ({page.money.code})",
                "revenue": f"revenue ({page.money.code})",
            }
        )
        st.dataframe(pct_columns(table.round(2)), hide_index=True, width="stretch")

with tab_km, ui.friendly_errors("building the clustering views"):
    lo = config.KMEANS_K_RANGE[0]
    hi = min(config.KMEANS_MAX_DISPLAY_K, int(base.evaluation["k"].max()))
    default = min(base.recommended_k or lo, hi)
    k = st.slider(
        "Number of clusters (k)",
        lo,
        hi,
        default,
        key="seg_k",
        help=f"Default = elbow of the inertia curve (k = {base.recommended_k}). "
        f"Up to {config.KMEANS_MAX_DISPLAY_K} clusters can be shown with distinct "
        "colours and symbols; all k from 2 to 10 are evaluated below.",
    )
    result = base if k == base.recommended_k else ui.segment(page, k)
    ev = result.evaluation.set_index("k")
    if k in ev.index:
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected k", k)
        m2.metric(
            "Silhouette score",
            f"{ev.loc[k, 'silhouette']:.3f}",
            help="−1 to 1; higher means better-separated clusters.",
        )
        m3.metric(
            "Davies–Bouldin index",
            f"{ev.loc[k, 'davies_bouldin']:.3f}",
            help="Lower means more compact, better-separated clusters.",
        )
    st.plotly_chart(
        visuals.fig_k_selection(result.evaluation, result.recommended_k, k),
        width="stretch",
        theme=None,
        key="k_sel",
    )
    st.caption(
        "Dashed line = recommended k; hollow circle = the k you selected. "
        "The elbow is found with the Kneedle method (the point furthest below the line joining "
        "the first and last inertia values). k = 2 has the best silhouette but only separates "
        "high- from low-value customers."
    )
    with st.expander("Evaluation table (all k)"):
        st.dataframe(result.evaluation.round(3), hide_index=True, width="stretch")
    prof = result.clusters.profile
    st.plotly_chart(visuals.fig_cluster_profile(prof), width="stretch", theme=None, key="profile")
    x = st.radio(
        "Horizontal axis",
        ["recency", "frequency"],
        horizontal=True,
        format_func=str.title,
        key="scatter_x",
    )
    st.plotly_chart(
        visuals.fig_cluster_scatter(result.customers, prof, x, cur=page.cur),
        width="stretch",
        theme=None,
        key="scatter",
    )
    left, right = st.columns([2, 3])
    with left:
        mix = prof.rename(columns={"name": "label", "revenue": "value", "revenue_share": "share"})[
            ["label", "value", "share"]
        ]
        colors = [visuals.cluster_style(c)[0] for c in prof["cluster"]]
        st.plotly_chart(
            visuals.fig_donut(
                mix,
                "Revenue by cluster",
                "money",
                visuals.compact_money(mix["value"].sum(), page.cur),
                cur=page.cur,
                colors=colors,
                legend_below=True,
            ),
            width="stretch",
            theme=None,
            key="cluster_donut",
        )
    with right:
        st.markdown("**Cluster summary**")
        st.dataframe(
            pct_columns(
                prof.drop(columns=["z_recency", "z_frequency", "z_monetary"])
                .rename(
                    columns={
                        "revenue": f"revenue ({page.money.code})",
                        "median_monetary": f"median_spend ({page.money.code})",
                    }
                )
                .round(2)
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander("How do clusters relate to RFM segments?"):
        st.dataframe(
            segmentation.segment_cluster_crosstab(result.customers, result.clusters.labels),
            width="stretch",
        )

with tab_lookup, ui.friendly_errors("building the customer table"):
    data = result.customers
    c1, c2, c3 = st.columns([1, 1, 1])
    query = c1.text_input("Search CustomerID", key="cust_q").strip()
    segs = c2.multiselect(
        "RFM segment", list(base.summary["segment"]), key="cust_seg", placeholder="All segments"
    )
    clus = c3.multiselect(
        "Cluster",
        [] if prof is None else list(prof["name"]),
        key="cust_cl",
        placeholder="All clusters",
    )
    view = data
    if query:
        view = view[view.index.astype(str).str.contains(query, regex=False)]
    if segs:
        view = view[view["segment"].isin(segs)]
    if clus:
        view = view[view["cluster_name"].isin(clus)]
    st.caption(f"{len(view):,} of {len(data):,} customers")
    shown = view.rename(columns={"monetary": f"monetary ({page.money.code})"})
    st.dataframe(shown.round({f"monetary ({page.money.code})": 2}), width="stretch")
    st.download_button(
        "Download customers (CSV)",
        shown.to_csv().encode("utf-8"),
        file_name="customer_segments.csv",
        mime="text/csv",
    )
    if st.button("Save segmentation to data/processed/"):
        segmentation.save_customers(data)
        st.success(
            f"Saved {len(data):,} customers to "
            f"`{config.PROCESSED_CUSTOMERS.relative_to(config.ROOT_DIR)}`."
        )
