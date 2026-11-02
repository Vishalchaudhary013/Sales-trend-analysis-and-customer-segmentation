"""Currency conversion of monetary columns using published daily exchange rates.

Source data is in :data:`config.BASE_CURRENCY` (GBP). Conversion uses US Federal
Reserve daily rates (see :data:`config.FX_SOURCE`); no rate is hard-coded.

* ``daily`` mode converts each line at the rate of its invoice date. Days
  without a published rate (weekends, US holidays) use the most recent earlier
  rate; dates before the first available rate use the first rate.
* ``average`` mode applies one rate: the mean of the daily rates over the
  period covered by the data.
* ``custom`` mode applies a single rate typed in by the user, for currencies
  not in the rate file. It is labelled as user-supplied everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

C = config
MONEY_COLUMNS: tuple[str, ...] = (C.COL_PRICE, C.COL_REVENUE)


class CurrencyError(Exception):
    """Raised when a conversion cannot be performed (missing or invalid rates)."""


@dataclass(frozen=True)
class CurrencySetting:
    """The user's display-currency choice.

    Attributes:
        code: ISO 4217 code (e.g. ``"USD"``).
        mode: ``"daily"``, ``"average"`` or ``"custom"``.
        custom_rate: Units of ``code`` per 1 base-currency unit (custom mode only).
        custom_symbol: Display prefix for custom currencies (defaults to the code).
    """

    code: str = config.BASE_CURRENCY
    mode: str = "daily"
    custom_rate: float | None = None
    custom_symbol: str | None = None

    @property
    def is_base(self) -> bool:
        """Return True when no conversion is needed."""
        return self.code == config.BASE_CURRENCY and self.mode != "custom"

    @property
    def symbol(self) -> str:
        """Display prefix used in labels and formatted values."""
        if self.mode == "custom":
            return self.custom_symbol or f"{self.code} "
        return config.CURRENCIES.get(self.code, ("", f"{self.code} ", None))[1]


def load_rates(path: Path = config.FX_FILE) -> pd.DataFrame:
    """Load daily rates as *units of each currency per 1 base-currency unit*.

    Args:
        path: CSV with columns ``Date, Country, Exchange rate`` (units per USD).

    Returns:
        DataFrame indexed by date (ascending) with one column per supported ISO
        code, including the base currency (always 1.0).

    Raises:
        CurrencyError: If the file is missing or lacks the base currency.
    """
    if not path.exists():
        raise CurrencyError(f"Exchange-rate file not found: {path}")
    raw = pd.read_csv(path, parse_dates=["Date"])
    raw = raw.dropna(subset=["Exchange rate"])
    per_usd = raw.pivot_table(
        index="Date", columns="Country", values="Exchange rate", aggfunc="last"
    ).sort_index()
    base_country = config.CURRENCIES[config.BASE_CURRENCY][2]
    if base_country not in per_usd.columns:
        raise CurrencyError(f"Rate file has no series for the base currency ({base_country}).")
    base_per_usd = per_usd[base_country]
    out = pd.DataFrame(index=per_usd.index)
    for code, (_, _, country) in config.CURRENCIES.items():
        if country is None:  # USD
            out[code] = 1.0 / base_per_usd
        elif country in per_usd.columns:
            out[code] = per_usd[country] / base_per_usd
    out = out[out[config.BASE_CURRENCY].notna()]
    out[config.BASE_CURRENCY] = 1.0
    return out


def available_currencies(rates: pd.DataFrame | None) -> list[str]:
    """Return ISO codes that can be converted (base first)."""
    if rates is None:
        return [config.BASE_CURRENCY]
    return [c for c in config.CURRENCIES if c in rates.columns and rates[c].notna().any()]


def daily_rates(rates: pd.DataFrame, code: str, dates: pd.Series) -> pd.Series:
    """Return the rate applicable to each timestamp in ``dates``.

    Uses the latest published rate on or before each date (backfilling only
    dates earlier than the first published rate).
    """
    series = rates[code].dropna()
    if series.empty:
        raise CurrencyError(f"No rates available for {code}.")
    days = dates.dt.normalize()
    unique = pd.DatetimeIndex(days.unique()).sort_values()
    lookup = series.reindex(series.index.union(unique)).ffill().bfill().reindex(unique)
    return days.map(lookup).astype(float)


def rate_coverage(
    rates: pd.DataFrame, code: str, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, object]:
    """Describe the rates used for ``code`` over ``[start, end]``.

    Returns:
        ``first``, ``last`` (dates of published rates), ``mean``, ``min``,
        ``max`` over the window, ``days`` with a published rate, and
        ``extrapolated`` (True if the window extends beyond published rates).
    """
    series = rates[code].dropna()
    window = series[(series.index >= start.normalize()) & (series.index <= end.normalize())]
    if window.empty:  # window falls in a gap: use the nearest earlier rate
        window = series[series.index <= end.normalize()].tail(1)
        if window.empty:
            window = series.head(1)
    return {
        "first": series.index.min(),
        "last": series.index.max(),
        "mean": float(window.mean()),
        "min": float(window.min()),
        "max": float(window.max()),
        "days": int(len(window)),
        "extrapolated": bool(start < series.index.min() or end > series.index.max()),
    }


def convert(df: pd.DataFrame, setting: CurrencySetting, rates: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy of ``df`` with monetary columns in the chosen currency.

    Adds ``FxRate`` (units of the target currency per base unit, per line).
    Quantities and all non-monetary columns are unchanged.

    Raises:
        CurrencyError: Invalid custom rate, unknown currency or missing rates.
    """
    if setting.is_base:
        return df
    if setting.mode == "custom":
        if (
            setting.custom_rate is None
            or not np.isfinite(setting.custom_rate)
            or setting.custom_rate <= 0
        ):
            raise CurrencyError("Enter a positive exchange rate for the custom currency.")
        factor: pd.Series | float = float(setting.custom_rate)
    else:
        if rates is None or setting.code not in rates.columns:
            raise CurrencyError(f"No exchange rates available for {setting.code}.")
        if setting.mode == "daily":
            factor = daily_rates(rates, setting.code, df[C.COL_DATE])
        elif setting.mode == "average":
            if df.empty:
                factor = 1.0
            else:
                cov = rate_coverage(rates, setting.code, df[C.COL_DATE].min(), df[C.COL_DATE].max())
                factor = float(cov["mean"])
        else:
            raise CurrencyError(f"Unknown conversion mode '{setting.mode}'.")
    out = df.copy()
    for col in MONEY_COLUMNS:
        if col in out.columns:
            out[col] = out[col] * factor
    out["FxRate"] = factor
    return out


def describe(
    setting: CurrencySetting, rates: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp
) -> str:
    """Return a one-line, human-readable description of the conversion applied."""
    base = config.BASE_CURRENCY
    if setting.is_base:
        return f"Amounts in {base} (source currency)."
    if setting.mode == "custom":
        return (
            f"Amounts in {setting.code} at a user-supplied rate of 1 {base} = "
            f"{setting.custom_rate:,.4f} {setting.code} (not from a published source)."
        )
    cov = rate_coverage(rates, setting.code, start, end)  # type: ignore[arg-type]
    how = config.FX_MODES[setting.mode].lower()
    text = (
        f"Amounts converted from {base} to {setting.code} using the {how}; over this "
        f"period 1 {base} = {cov['min']:,.4f}–{cov['max']:,.4f} {setting.code} "
        f"(mean {cov['mean']:,.4f}). Source: {config.FX_SOURCE}."
    )
    if cov["extrapolated"]:
        text += " Some dates fall outside the published rates; the nearest rate was used."
    return text
