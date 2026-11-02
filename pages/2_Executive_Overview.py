"""Executive Overview: headline KPIs for the filtered dataset."""

import streamlit as st

from src import analytics, ui

ui.page_setup("Executive Overview", "📈")
active, data, filters = ui.load_page_data()

with ui.friendly_errors("computing the KPIs"):
    kpis = analytics.compute_kpis(data)

    st.subheader("Headline indicators")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{kpis['n_records']:,}")
    if kpis["n_students"] is not None:
        c1.caption(
            f"≈ {kpis['n_students']:,} unique students (records are per student *per subject*)"
        )
    c2.metric("Mean final grade (G3)", ui.fmt_num(kpis["mean_g3"]), help="Scale 0–20")
    c3.metric("Median final grade (G3)", ui.fmt_num(kpis["median_g3"], 1), help="Scale 0–20")

    c4, c5, c6 = st.columns(3)
    c4.metric("Pass rate (G3 ≥ 10)", ui.fmt_pct(kpis["pass_rate"]))
    if kpis["n_pass"] is not None:
        c4.caption(f"✔ Pass: {kpis['n_pass']:,} records")
    c5.metric("Fail rate (G3 < 10)", ui.fmt_pct(kpis["fail_rate"]))
    if kpis["n_fail"] is not None:
        c5.caption(f"✖ Fail: {kpis['n_fail']:,} records")
    with c6:
        st.markdown("**Pass rate by subject**")
        if kpis["by_subject"]:
            for subject, s in kpis["by_subject"].items():
                st.markdown(f"- {subject.title()}: **{ui.fmt_pct(s['pass_rate'])}** (n={s['n']:,})")
        else:
            st.caption("No subject information available.")

    if kpis["mean_g3"] is None:
        st.info("This dataset has no G3 column, so grade KPIs are unavailable.")

    st.divider()
    st.subheader("Summary statistics")
    st.dataframe(analytics.describe_numeric(data), width="stretch")
    st.caption(
        "All values are computed live from the filtered data. G3 = 0 records "
        "(likely non-appearance at the final exam) can be excluded in the sidebar."
    )
