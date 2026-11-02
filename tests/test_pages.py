"""Headless smoke tests of the Streamlit pages (skipped without the UCI files)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src import config

pytestmark = pytest.mark.skipif(
    not all((config.RAW_DIR / f).exists() for f in config.RAW_FILES.values()),
    reason="UCI files not downloaded",
)


@pytest.mark.parametrize(
    "page", ["app.py", "pages/2_Executive_Overview.py", "pages/3_Exploratory_Analysis.py"]
)
def test_page_runs_without_exceptions(page):
    at = AppTest.from_file(str(config.ROOT_DIR / page), default_timeout=60).run()
    assert not at.exception
    assert not at.error


def test_overview_kpis_and_filters_apply():
    at = AppTest.from_file(
        str(config.ROOT_DIR / "pages/2_Executive_Overview.py"), default_timeout=60
    ).run()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Total records"] == "1,044"
    assert labels["Pass rate (G3 ≥ 10)"] == "78.0%"
    at.toggle[0].set_value(True).run()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Total records"] == "991"
    at.multiselect[0].set_value(["math"]).run()
    assert {m.label: m.value for m in at.metric}["Total records"] == "357"
