from __future__ import annotations

import pandas as pd
import pytest

from src import config, currency


@pytest.fixture
def fx_file(tmp_path):
    """Synthetic rate file in the upstream layout (units per USD); not real rates."""
    rows = []
    for day, gbp, eur, inr in [("2011-01-03", 0.5, 0.75, 45.0), ("2011-01-04", 0.625, 0.75, 50.0),
                               ("2011-01-07", 0.5, 0.8, 40.0)]:  # fmt: skip
        rows += [(day, "United Kingdom", gbp), (day, "Euro", eur), (day, "India", inr)]
    rows.append(("2011-01-05", "United Kingdom", None))  # missing value row is ignored
    path = tmp_path / "fx.csv"
    pd.DataFrame(rows, columns=["Date", "Country", "Exchange rate"]).to_csv(path, index=False)
    return path


@pytest.fixture
def rates(fx_file):
    return currency.load_rates(fx_file)


def test_load_rates_converts_to_per_base(rates):
    row = rates.loc["2011-01-03"]
    assert row["GBP"] == 1.0
    assert row["USD"] == pytest.approx(2.0)  # 1 / 0.5 GBP per USD
    assert row["EUR"] == pytest.approx(1.5)  # 0.75 / 0.5
    assert row["INR"] == pytest.approx(90.0)
    assert currency.available_currencies(rates) == ["GBP", "USD", "EUR", "INR"]
    assert currency.available_currencies(None) == ["GBP"]


def test_load_rates_errors(tmp_path):
    with pytest.raises(currency.CurrencyError, match="not found"):
        currency.load_rates(tmp_path / "missing.csv")
    p = tmp_path / "x.csv"
    pd.DataFrame({"Date": ["2011-01-03"], "Country": ["Euro"], "Exchange rate": [0.7]}).to_csv(p)
    with pytest.raises(currency.CurrencyError, match="base currency"):
        currency.load_rates(p)


def test_daily_rates_forward_fill(rates):
    dates = pd.Series(pd.to_datetime(["2011-01-01 10:00", "2011-01-04 15:00",
                                      "2011-01-06 09:00", "2011-02-01 00:00"]))  # fmt: skip
    got = currency.daily_rates(rates, "USD", dates).tolist()
    # before first -> first; exact day; weekend/gap -> previous; after last -> last
    assert got == pytest.approx([2.0, 1.6, 1.6, 2.0])


def test_convert_daily_average_custom(rates, clean):
    df = clean.assign(InvoiceDate=pd.Timestamp("2011-01-04 12:00"))
    usd = currency.convert(df, currency.CurrencySetting("USD", "daily"), rates)
    assert usd["Revenue"].sum() == pytest.approx(df["Revenue"].sum() * 1.6)
    assert usd["UnitPrice"].iloc[0] == pytest.approx(df["UnitPrice"].iloc[0] * 1.6)
    assert (usd["Quantity"] == df["Quantity"]).all() and (usd["FxRate"] == 1.6).all()
    avg = currency.convert(df, currency.CurrencySetting("USD", "average"), rates)
    assert avg["FxRate"].iloc[0] == pytest.approx(1.6)  # window = that day only
    custom = currency.convert(df, currency.CurrencySetting("AED", "custom", 5.0), None)
    assert custom["Revenue"].sum() == pytest.approx(df["Revenue"].sum() * 5)
    assert currency.convert(df, currency.CurrencySetting(), rates) is df  # base: unchanged
    assert currency.convert(df.iloc[0:0], currency.CurrencySetting("USD", "average"),
                            rates).empty  # fmt: skip


@pytest.mark.parametrize(
    ("setting", "msg"),
    [(currency.CurrencySetting("AED", "custom", 0), "positive"),
     (currency.CurrencySetting("AED", "custom", None), "positive"),
     (currency.CurrencySetting("XYZ", "daily"), "No exchange rates"),
     (currency.CurrencySetting("USD", "weekly"), "Unknown conversion mode")],
)  # fmt: skip
def test_convert_errors(rates, clean, setting, msg):
    with pytest.raises(currency.CurrencyError, match=msg):
        currency.convert(clean, setting, rates)
    if setting.mode != "custom":
        with pytest.raises(currency.CurrencyError):
            currency.convert(clean, setting, None)


def test_symbols_and_descriptions(rates):
    assert currency.CurrencySetting().symbol == "£"
    assert currency.CurrencySetting("USD").symbol == "$"
    assert currency.CurrencySetting("AED", "custom", 3.9).symbol == "AED "
    assert currency.CurrencySetting("AED", "custom", 3.9, "د.إ").symbol == "د.إ"
    start, end = pd.Timestamp("2011-01-03"), pd.Timestamp("2011-01-07")
    assert "source currency" in currency.describe(currency.CurrencySetting(), rates, start, end)
    text = currency.describe(currency.CurrencySetting("USD", "daily"), rates, start, end)
    assert "1 GBP = 1.6000–2.0000 USD" in text and "Federal Reserve" in text
    assert "user-supplied" in currency.describe(
        currency.CurrencySetting("AED", "custom", 3.9), rates, start, end)  # fmt: skip
    late = currency.describe(currency.CurrencySetting("USD", "average"), rates,
                             start, pd.Timestamp("2012-01-01"))  # fmt: skip
    assert "outside the published rates" in late


def test_rate_coverage_gap_window(rates):
    cov = currency.rate_coverage(rates, "USD", pd.Timestamp("2011-01-05"),
                                 pd.Timestamp("2011-01-06"))  # fmt: skip
    assert cov["mean"] == pytest.approx(1.6) and cov["days"] == 1  # nearest earlier rate
    early = currency.rate_coverage(rates, "USD", pd.Timestamp("2010-01-01"),
                                   pd.Timestamp("2010-01-02"))  # fmt: skip
    assert early["mean"] == pytest.approx(2.0) and early["extrapolated"]


@pytest.mark.skipif(not config.FX_FILE.exists(), reason="rates not downloaded")
def test_real_rates_match_market_anchors():
    """Guards the 'units per USD' convention against known 2011 market levels."""
    r = currency.load_rates()
    day = r.loc["2011-01-04"]
    assert 1.5 < day["USD"] < 1.6  # GBP/USD ~1.56 on 4 Jan 2011
    assert 1.1 < day["EUR"] < 1.25  # GBP/EUR ~1.17
    assert 65 < day["INR"] < 75
    year = r.loc["2011-01-01":"2011-12-31", "USD"].mean()
    assert year == pytest.approx(1.604, abs=0.005)  # published 2011 annual average
