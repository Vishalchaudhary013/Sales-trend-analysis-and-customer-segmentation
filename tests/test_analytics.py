from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import analytics as a
from src import ingestion


def _tx(rows):
    """Build a clean table from compact (invoice, stock, qty, date, price, cust, country) rows."""
    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
                                     "UnitPrice", "CustomerID", "Country"])  # fmt: skip
    df["Description"] = "ITEM " + df["StockCode"]
    return ingestion.load_table(df, "t").data


JAN1 = pd.Timestamp("2011-01-01")


@pytest.fixture
def tiny():
    return _tx([
        ("1", "A", 2, "2011-01-03 09:00", 10.0, "100", "UK"),
        ("1", "X", 1, "2011-01-03 09:00", 5.0, "100", "UK"),
        ("2", "A", 1, "2011-01-10 15:00", 10.0, "200", "France"),
        ("C3", "A", -1, "2011-01-11 10:00", 10.0, "200", "France"),   # reverses invoice 2
        ("4", "POST", 1, "2011-02-07 11:00", 15.0, "100", "UK"),      # postage
        ("5", "X", 4, "2011-02-14 12:00", 5.0, None, "UK"),           # guest
        ("6", "A", 3, "2011-03-07 10:00", 10.0, "100", "UK"),
        ("7", "X", 2, "2011-03-31 16:00", 5.0, "300", "Germany"),
    ])  # fmt: skip


def test_reversal_matching(tiny):
    reversed_ = tiny.loc[tiny.IsReversed, "InvoiceNo"].tolist()
    assert reversed_ == ["2"]


def test_line_mask_modes(tiny):
    gross = tiny[a.line_mask(tiny, a.Filters())]
    assert set(gross.InvoiceNo) == {"1", "5", "6", "7"}
    net = tiny[a.line_mask(tiny, a.Filters(include_returns=True))]
    assert {"2", "C3"} <= set(net.InvoiceNo) and "4" not in set(net.InvoiceNo)
    fees = tiny[a.line_mask(tiny, a.Filters(include_non_product=True))]
    assert "4" in set(fees.InvoiceNo)
    assert net.Revenue.sum() == pytest.approx(gross.Revenue.sum())  # reversal nets to zero


def test_scope_filters(tiny):
    f = a.Filters(date_from=date(2011, 2, 1), date_to=date(2011, 3, 7), countries=["UK"])
    s = a.scope_filter(tiny, f)
    assert set(s.InvoiceNo) == {"4", "5", "6"}  # date_to is inclusive of the whole day
    assert set(a.scope_filter(tiny, a.Filters(customer_scope="guest")).InvoiceNo) == {"5"}
    assert "5" not in set(a.scope_filter(tiny, a.Filters(customer_scope="identified")).InvoiceNo)
    assert "countries: UK" in f.describe() and f.to_dict()["date_from"] == "2011-02-01"
    assert "completed sales" in a.Filters().describe()
    assert "net of returns" in a.Filters(include_returns=True).describe()
    assert "guest" in a.Filters(customer_scope="guest").describe()


def test_filter_options(tiny):
    opts = a.filter_options(tiny)
    assert opts["date_range"] == (date(2011, 1, 3), date(2011, 3, 31))
    assert opts["countries"][0] == "UK"  # ordered by revenue
    assert "countries" not in a.filter_options(tiny.drop(columns=["Country"]))


def test_kpis(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    k = a.compute_kpis(lines, a.scope_filter(tiny, a.Filters()))
    assert k["revenue"] == pytest.approx(25 + 20 + 30 + 10)
    assert k["orders"] == 4 and k["customers"] == 2
    assert k["aov"] == pytest.approx(85 / 4)
    assert k["units"] == 2 + 1 + 4 + 3 + 2
    assert k["revenue_per_customer"] == pytest.approx((25 + 30 + 10) / 2)
    assert k["identified_revenue_share"] == pytest.approx(65 / 85)
    assert k["cancellation_rate"] == pytest.approx(1 / 7)  # 1 of 7 invoices
    assert k["cancelled_value"] == pytest.approx(10)
    empty = a.compute_kpis(lines.iloc[0:0])
    assert empty["aov"] is None and empty["cancellation_rate"] is None


def test_time_series_monthly(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    ts = a.time_series(lines, "Monthly", tiny.InvoiceDate.max(), JAN1)
    assert ts["period_label"].tolist() == ["Jan 2011", "Feb 2011", "Mar 2011"]
    assert ts["revenue"].tolist() == pytest.approx([25, 20, 40])
    assert ts["orders"].tolist() == [1, 1, 2]
    assert ts["partial"].tolist() == [False, False, False]
    assert np.isnan(ts["growth"].iloc[0]) and ts["growth"].iloc[2] == pytest.approx(1.0)
    assert a.time_series(lines.iloc[0:0], "Monthly").empty


def test_partial_periods_and_growth_suppressed(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    end = pd.Timestamp("2011-03-15")  # data ends mid-March -> March partial
    ts = a.time_series(lines[lines.InvoiceDate <= end], "Monthly", end, JAN1)
    assert ts["partial"].tolist() == [False, False, True]
    # Data starting after the 1st makes the first month partial too
    first = a.time_series(lines, "Monthly", tiny.InvoiceDate.max(), tiny.InvoiceDate.min())
    assert first["partial"].iloc[0]
    assert np.isnan(ts["growth"].iloc[-1])
    wk = a.time_series(lines, "Weekly", tiny.InvoiceDate.max(), tiny.InvoiceDate.min())
    assert (wk["period_start"].dt.day_of_week == 0).all()  # weeks start Monday
    assert wk["revenue"].sum() == pytest.approx(85)
    assert (wk["revenue"] == 0).any()  # empty weeks zero-filled
    daily = a.time_series(lines, "Daily", tiny.InvoiceDate.max(), tiny.InvoiceDate.min())
    assert not daily["partial"].any()


def test_period_comparison(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    comp = a.period_comparison(lines, tiny.InvoiceDate.max())
    assert comp["current"] == "Mar 2011" and comp["previous"] == "Feb 2011"
    cur, prev, pct = comp["metrics"]["revenue"]
    assert (cur, prev, pct) == (40, 20, 1.0)
    assert a.period_comparison(lines[lines.InvoiceDate < "2011-02-01"], tiny.InvoiceDate.max()) \
        is None  # fmt: skip


def test_weekday_hour_matrix(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    m = a.weekday_hour_matrix(lines)
    assert list(m.index) == list(a.WEEKDAYS)
    assert m.loc["Mon", 9] == pytest.approx(25)
    assert m.loc["Sat"].isna().all()  # no trading is NaN, not 0
    orders = a.weekday_hour_matrix(lines, "orders")
    assert orders.loc["Mon", 9] == 1
    assert a.weekday_hour_matrix(lines.iloc[0:0]).empty


def test_products_countries_concentration(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    top = a.top_products(lines, 5)
    assert top["StockCode"].tolist() == ["A", "X"]
    assert top["revenue"].tolist() == pytest.approx([50, 35])
    assert top["share"].sum() == pytest.approx(1)
    assert a.top_products(lines, 5, "units")["StockCode"].iloc[0] == "X"
    assert a.top_products(lines.drop(columns=["Description"]), 2)["Description"].isna().all()
    assert a.top_products(lines.iloc[0:0]).empty
    cs = a.country_summary(lines)
    assert cs["Country"].iloc[0] == "UK" and cs["share"].sum() == pytest.approx(1)
    assert a.country_summary(lines.drop(columns=["Country"])).empty
    conc = a.revenue_concentration(lines)
    assert conc["top_1pct"] == pytest.approx(50 / 85)
    assert a.revenue_concentration(lines.iloc[0:0]) == {}


def test_large_lines(tiny):
    assert len(a.large_lines(tiny, threshold=4)) == 1
    assert a.large_lines(tiny).empty


def test_new_vs_returning(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    nr = a.new_vs_returning(lines, lines, tiny.InvoiceDate.max())
    rows = nr.set_index("period_label")
    assert rows.loc["Jan 2011", "new"] == 1 and rows.loc["Jan 2011", "censored"]
    assert rows.loc["Mar 2011", "returning"] == 1 and rows.loc["Mar 2011", "new"] == 1
    # Date-filtered view keeps first-purchase month from full history
    march = lines[lines.InvoiceDate >= "2011-03-01"]
    assert a.new_vs_returning(march, lines).set_index("period_label").loc[
        "Mar 2011", "returning"] == 1  # fmt: skip
    assert a.new_vs_returning(lines[~lines.HasCustomer], lines).empty


def test_cancellation_trend(tiny):
    ct = a.cancellation_trend(tiny).set_index("period_label")
    assert ct.loc["Jan 2011", "cancelled"] == 1 and ct.loc["Jan 2011", "invoices"] == 3
    assert ct.loc["Jan 2011", "cancelled_value"] == pytest.approx(10)
    assert a.cancellation_trend(tiny.iloc[0:0]).empty


def test_synthetic_end_to_end(clean):
    lines = a.apply_filters(clean, a.Filters())
    k = a.compute_kpis(lines, clean)
    ts = a.time_series(lines, "Monthly", clean.InvoiceDate.max(), clean.InvoiceDate.min())
    assert ts["revenue"].sum() == pytest.approx(k["revenue"])


def test_part_to_whole_breakdowns(tiny):
    lines = a.apply_filters(tiny, a.Filters())
    share = a.revenue_share(lines, top_n=1)
    assert share["label"].tolist() == ["UK", "Other"]
    assert share["value"].sum() == pytest.approx(85) and share["share"].sum() == pytest.approx(1)
    assert len(a.revenue_share(lines)) == 2  # no "Other" when everything fits
    ct = a.revenue_by_customer_type(lines).set_index("label")
    assert ct.loc["Guest sales", "value"] == pytest.approx(20)
    inv = a.invoice_status_mix(tiny).set_index("label")
    assert inv.loc["Cancelled", "value"] == 1 and inv.loc["Completed", "value"] == 6
    assert a.revenue_share(lines.drop(columns=["Country"])).empty
    assert a.revenue_by_customer_type(lines.iloc[0:0]).empty
    assert a.invoice_status_mix(tiny.iloc[0:0]).empty
