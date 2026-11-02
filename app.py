"""Streamlit entry point: the landing (Home) page. Other pages live in ``pages/``."""

import streamlit as st

from src import config, ingestion, ui


def home() -> None:
    """Landing page: project summary, navigation guide and dataset status."""
    ui.page_setup("Home", "🏠", heading=config.APP_TITLE)
    st.markdown(
        "Analyse **sales trends**, segment **customers** by purchasing behaviour and flag "
        "customers at risk of **churning** — all computed live from transaction data."
    )
    st.subheader("Dataset status")
    with ui.friendly_errors("checking the dataset"):
        active = ui.get_active_dataset()
        if active is None:
            ui.show_missing_data_help()
        else:
            data = active.data
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Transaction lines", f"{len(data):,}")
            c2.metric("Invoices", f"{data[config.COL_INVOICE].nunique():,}")
            c3.metric("Customers (identified)", f"{data[config.COL_CUSTOMER].nunique():,}")
            c4.metric("Period", f"{active.start:%b %Y} – {active.end:%b %Y}")
            st.success(
                f"✔ Loaded: **{active.label}** "
                f"({'UCI benchmark' if active.mode == 'bundled' else 'custom upload'})"
                + (f" · SHA-256 `{active.checksum[:12]}…`" if active.checksum else "")
            )
            caps = ingestion.assess_capabilities(data)
            unavailable = {k: v for k, v in caps.items() if v}
            if unavailable:
                st.warning("Some features are unavailable for this dataset:")
                for feature, reasons in unavailable.items():
                    st.markdown(f"- **{feature}**: {'; '.join(reasons)}")

    st.subheader("How to use this dashboard")
    st.markdown(
        "- **Executive Overview**: headline KPIs and the latest complete month versus the "
        "month before.\n"
        "- **Sales Trends**: revenue, orders, customers and order value over time; growth; "
        "weekday × hour pattern; top products and countries; new vs returning customers; "
        "cancellations.\n"
        "- Further pages (data ingestion, customer segments, churn model, early warning, "
        "methodology) are added in later milestones.\n\n"
        "Use the **sidebar filters** (date range, country, customer type, returns, fees). "
        "They apply to every page."
    )


home()
