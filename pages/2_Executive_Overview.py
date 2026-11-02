"""Executive Overview: headline KPIs for the filtered view."""

import streamlit as st

from src import analytics, ui, visuals

ui.page_setup("Executive Overview", "📈")
page = ui.load_page_data()

with ui.friendly_errors("computing the KPIs"):
    k = analytics.compute_kpis(page.lines, page.scope)
    label = "Net revenue" if page.filters.include_returns else "Revenue (completed sales)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(label, ui.fmt_money(k["revenue"], cur=page.cur))
    c2.metric("Orders", ui.fmt_int(k["orders"]), help="Distinct non-cancellation invoices.")
    c3.metric(
        "Average order value", ui.fmt_money(k["aov"], 2, cur=page.cur), help="Revenue ÷ orders."
    )
    c4.metric(
        "Active customers",
        ui.fmt_int(k["customers"]),
        help="Distinct CustomerIDs; guest sales have none.",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Units sold", ui.fmt_int(k["units"]))
    c6.metric(
        "Revenue per customer",
        ui.fmt_money(k["revenue_per_customer"], cur=page.cur),
        help="Identified-customer revenue ÷ identified customers.",
    )
    c7.metric("Revenue from identified customers", ui.fmt_pct(k["identified_revenue_share"]))
    c8.metric(
        "Cancellation rate",
        ui.fmt_pct(k["cancellation_rate"]),
        help="Cancelled invoices ÷ all invoices in the selected scope. Cancelled value: "
        f"{ui.fmt_money(k['cancelled_value'], cur=page.cur)}.",
    )

    st.divider()
    st.subheader("Latest complete month")
    comp = analytics.period_comparison(page.lines, page.active.end)
    if comp is None:
        st.info("At least two complete calendar months are needed for a month-on-month view.")
    else:
        st.caption(
            f"{comp['current']} compared with {comp['previous']}. Partial months at the "
            "edges of the data are excluded."
        )
        cols = st.columns(4)
        names = {
            "revenue": "Revenue",
            "orders": "Orders",
            "customers": "Active customers",
            "aov": "Average order value",
        }
        for col, (key, title) in zip(cols, names.items(), strict=True):
            now, prev, pct = comp["metrics"][key]
            value = (
                ui.fmt_money(now, 2 if key == "aov" else 0, cur=page.cur)
                if key in ("revenue", "aov")
                else ui.fmt_int(now)
            )
            col.metric(
                title, value, delta=ui.fmt_pct(pct, signed=True) if pct is not None else None
            )

    st.divider()
    st.subheader("Revenue and order mix")
    total = visuals.compact_money(k["revenue"], page.cur)
    invoices = analytics.invoice_status_mix(page.scope)
    mixes = [
        ("mix_country", analytics.revenue_share(page.lines), "Revenue by country", "money", total),
        (
            "mix_customer",
            analytics.revenue_by_customer_type(page.lines),
            "Revenue by customer type",
            "money",
            total,
        ),
        (
            "mix_invoices",
            invoices,
            "Invoices by status",
            "count",
            f"{int(invoices['value'].sum()):,}" if not invoices.empty else None,
        ),
    ]
    slots = [*st.columns(2), st.columns(2)[0]]
    for col, (slug, mix, title, kind, centre) in zip(slots, mixes, strict=True):
        with col:
            st.plotly_chart(
                visuals.fig_donut(mix, title, kind, centre, cur=page.cur),
                width="stretch",
                theme=None,
                key=slug,
            )
            st.caption(visuals.FIGURE_DESCRIPTIONS[slug])
            if not mix.empty:
                with st.expander("View data table"):
                    st.dataframe(
                        mix.assign(share=mix["share"].map("{:.1%}".format)),
                        hide_index=True,
                        width="stretch",
                    )

    conc = analytics.revenue_concentration(page.lines)
    if conc:
        st.divider()
        st.subheader("Revenue concentration")
        st.markdown(
            f"The top **1%** of products earn **{conc['top_1pct']:.0%}** of revenue, the top "
            f"**10%** earn **{conc['top_10pct']:.0%}** and the top **20%** earn "
            f"**{conc['top_20pct']:.0%}**."
        )
    big = analytics.large_lines(page.scope)
    if not big.empty:
        with st.expander(f"⚠ {len(big)} unusually large line(s) (|quantity| ≥ 10,000)"):
            st.dataframe(big, hide_index=True, width="stretch")
            st.caption(
                f"Revenue in {page.money.code}. Lines flagged IsReversed were cancelled by a "
                "matching line and are excluded from completed-sales figures."
            )
