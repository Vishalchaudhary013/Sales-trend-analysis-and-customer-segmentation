"""Headless smoke tests of the Streamlit app (skipped without the bundled dataset)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src import config

pytestmark = pytest.mark.skipif(not config.RAW_FILE.exists(), reason="dataset not downloaded")
APP = str(config.ROOT_DIR / "app.py")


def _metrics(at: AppTest) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in at.metric:
        out.setdefault(m.label, m.value)  # first occurrence (headline KPI row)
    return out


def _app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


def test_home_runs():
    at = _app()
    assert not at.exception and not at.error
    metrics = _metrics(at)
    assert metrics["Transaction lines"] == "536,641"
    assert metrics["Invoices"] == "25,900"


@pytest.mark.parametrize("page", ["pages/2_Executive_Overview.py", "pages/3_Sales_Trends.py"])
def test_pages_run(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception and not at.error


def test_overview_kpis_and_filter():
    at = _app()
    at.switch_page("pages/2_Executive_Overview.py").run()
    m = _metrics(at)
    assert m["Revenue (completed sales)"] == "£9,823,459"
    assert m["Orders"] == "19,648"
    at.toggle[0].set_value(True).run()  # net of returns
    m = _metrics(at)
    assert "Net revenue" in m
    at.radio[0].set_value("guest").run()
    assert _metrics(at)["Active customers"] == "0"


def test_currency_switch_converts_kpis():
    at = _app()
    at.switch_page("pages/2_Executive_Overview.py").run()
    gbp = _metrics(at)["Revenue (completed sales)"]
    at.selectbox[0].set_value("USD").run()
    assert not at.exception and not at.error
    usd = _metrics(at)["Revenue (completed sales)"]
    assert gbp.startswith("£") and usd.startswith("$")
    ratio = float(usd[1:].replace(",", "")) / float(gbp[1:].replace(",", ""))
    assert 1.53 < ratio < 1.67  # within the 2011 GBP/USD range
    assert any("Federal Reserve" in c.value for c in at.caption)


def test_customer_segments_page():
    at = _app()
    at.switch_page("pages/4_Customer_Segments.py").run()
    assert not at.exception and not at.error
    m = _metrics(at)
    assert m["Selected k"] == "5"
    assert m["Champions"] == "630"
    at.slider(key="seg_k").set_value(3).run()
    assert not at.exception and _metrics(at)["Selected k"] == "3"


@pytest.mark.parametrize(
    "page",
    [
        "pages/1_Data_Ingestion.py",
        "pages/5_Model_Evaluation.py",
        "pages/6_Early_Warning.py",
        "pages/7_Methodology.py",
    ],
)
def test_m5_pages_run(page):
    at = _app()
    at.switch_page(page).run()
    assert not at.exception and not at.error


def test_model_evaluation_threshold_and_balance():
    at = _app()
    at.switch_page("pages/5_Model_Evaluation.py").run()
    m = _metrics(at)
    assert m["Customers in training set"] == "3,364"
    at.slider(key="me_threshold").set_value(0.3).run()
    at.toggle(key="me_balanced").set_value(True).run()
    assert not at.exception and not at.error


def test_early_warning_lists_customers():
    at = _app()
    at.switch_page("pages/6_Early_Warning.py").run()
    assert not at.exception and not at.error
    from src import modeling

    if modeling.model_status(config.MODEL_FILE)[0] == "ok":
        m = _metrics(at)
        assert int(m["Customers scored"].replace(",", "")) > 0
        assert len(at.dataframe) >= 1
    else:  # missing or incompatible model: the page explains and offers retraining
        assert any("Train and save" in b.label for b in at.button)
