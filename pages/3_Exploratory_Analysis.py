"""Exploratory Analysis: interactive charts plus one-click PNG export."""

import streamlit as st

from src import analytics, config, ui, visuals

ui.page_setup("Exploratory Analysis", "🔎")
active, data, filters = ui.load_page_data()

with ui.friendly_errors("building the charts"):
    figures = visuals.build_figures(data)

    def show(slug: str, table=None) -> None:
        """Render one figure with its description and an optional data table."""
        st.plotly_chart(figures[slug], width="stretch", theme=None, key=slug)
        st.caption(visuals.FIGURE_DESCRIPTIONS[slug])
        if table is not None and not table.empty:
            with st.expander("View data table"):
                st.dataframe(table, width="stretch", hide_index=True)

    tab_dist, tab_drivers, tab_corr = st.tabs(["Distributions", "Drivers", "Correlations"])
    with tab_dist:
        show("g3_distribution", analytics.grade_counts(data))
        left, right = st.columns(2)
        with left:
            show("g3_by_subject")
        with right:
            show("g3_by_higher")
        show("g2_vs_g3")
    with tab_drivers:
        left, right = st.columns(2)
        with left:
            show("mean_g3_by_studytime", analytics.mean_g3_by(data, "studytime").round(2))
        with right:
            show("mean_g3_by_failures", analytics.mean_g3_by(data, "failures").round(2))
        show(
            "pass_rate_by_absence_band",
            analytics.pass_rate_by(data, config.ABSENCE_BAND_COL).round(3),
        )
    with tab_corr:
        show("correlation_with_g3", analytics.correlations_with_g3(data).round(3))
        show("correlation_heatmap")
        st.caption("Correlation describes association only; it does not imply causation.")

st.divider()
st.subheader("Export")
st.write(
    f"Save all {len(visuals.STATIC_BUILDERS)} charts for the current filters as "
    f"{config.EXPORT_DPI}-dpi PNGs to `reports/figures/`."
)
if st.button("Export all figures", type="primary"):
    with ui.friendly_errors("exporting the figures"), st.spinner("Rendering figures…"):
        paths = visuals.export_static_figures(data)
        target = config.FIGURES_DIR.relative_to(config.ROOT_DIR)
        st.success(f"Exported {len(paths)} figures to `{target}/`.")
        st.write(", ".join(p.name for p in paths))
