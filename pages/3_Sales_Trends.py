"""Sales Trends: time series, growth, weekday × hour, products, countries, customers."""

import streamlit as st

from src import analytics, config, ui, visuals

ui.page_setup("Sales Trends", "📅")
page = ui.load_page_data()


def show(fig, slug: str, table=None, key: str | None = None) -> None:
    """Render a figure with its description and an optional data table."""
    st.plotly_chart(fig, width="stretch", theme=None, key=key or slug)
    st.caption(visuals.FIGURE_DESCRIPTIONS[slug])
    if table is not None and not table.empty:
        with st.expander("View data table"):
            st.dataframe(table, hide_index=True, width="stretch")


tab_time, tab_mix, tab_cust = st.tabs(["Over time", "Products & countries", "Customers"])

with tab_time, ui.friendly_errors("building the time-series charts"):
    c1, c2 = st.columns(2)
    freq = (
        c1.segmented_control(
            "Granularity", list(analytics.FREQ_ALIASES), default="Monthly", key="trend_freq"
        )
        or "Monthly"
    )
    metric = c2.selectbox(
        "Metric",
        list(visuals.METRIC_NAMES),
        format_func=visuals.METRIC_NAMES.get,
        key="trend_metric",
    )
    ts = analytics.time_series(page.lines, freq, page.active.end, page.active.start)
    show(
        visuals.fig_trend(ts, metric, freq, cur=page.cur),
        "trend",
        ts.drop(columns=["period_start"]),
    )
    if freq != "Daily":
        show(visuals.fig_growth(ts, freq), "growth")
    show(
        visuals.fig_weekday_hour(analytics.weekday_hour_matrix(page.lines), cur=page.cur), "heatmap"
    )

with tab_mix, ui.friendly_errors("building the product and country charts"):
    c1, c2 = st.columns([1, 1])
    by = c1.radio(
        "Rank products by",
        ["revenue", "units"],
        horizontal=True,
        format_func=str.title,
        key="prod_by",
    )
    n = c2.slider("Number of products", 5, 30, 15, key="prod_n")
    top = analytics.top_products(page.lines, n, by)
    show(visuals.fig_top_products(top, by, cur=page.cur), "products", top)
    if config.COL_COUNTRY in page.lines.columns:
        countries = analytics.country_summary(page.lines)
        exclude_uk = st.toggle("Exclude United Kingdom (dominant market)", key="ex_uk")
        view = (
            countries[countries[config.COL_COUNTRY] != "United Kingdom"]
            if exclude_uk
            else countries
        )
        show(visuals.fig_countries(view, cur=page.cur), "countries", countries)
    else:
        st.info("Country analysis is unavailable: the dataset has no Country column.")

with tab_cust, ui.friendly_errors("building the customer charts"):
    history = page.active.data.loc[analytics.line_mask(page.active.data, page.filters)]
    nr = analytics.new_vs_returning(page.lines, history, page.active.end)
    show(visuals.fig_new_returning(nr), "new_returning", nr.drop(columns=["period_start"]))
    canc = analytics.cancellation_trend(page.scope)
    show(
        visuals.fig_cancellations(canc, cur=page.cur),
        "cancellations",
        canc.drop(columns=["period_start"]),
    )
