"""Shared Streamlit UI glue: cached data access, global filters, page helpers.

Presentation code only: all computation is delegated to :mod:`src.ingestion`
and :mod:`src.analytics`, so pages stay free of business logic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import analytics, config, currency, ingestion

logger = logging.getLogger(__name__)
_W = "_filter_widget_"  # widget-key prefix
SCOPE_LABELS = {"all": "All customers", "identified": "Identified only", "guest": "Guests only"}


# Sidebar menu: (script path relative to the repo root, label, icon).
NAV_PAGES: tuple[tuple[str, str, str], ...] = (
    ("app.py", "Home", "🏠"),
    ("pages/2_Executive_Overview.py", "Executive Overview", "📈"),
    ("pages/3_Sales_Trends.py", "Sales Trends", "📅"),
)


def page_setup(title: str, icon: str, heading: str | None = None) -> None:
    """Configure the page, render the sidebar menu and the page heading."""
    st.set_page_config(page_title=f"{title} · {config.APP_TITLE}", page_icon=icon, layout="wide")
    with st.sidebar:
        for path, label, page_icon in NAV_PAGES:
            st.page_link(path, label=label, icon=page_icon)
        st.divider()
    st.title(heading or title)


@dataclass
class ActiveDataset:
    """The dataset every page works on.

    Attributes:
        data: Clean transaction table (see :func:`src.ingestion.prepare`).
        label: Human-readable source name.
        mode: ``"bundled"`` (UCI benchmark) or ``"custom"`` (user upload).
        checksum: SHA-256 of the source file, if known.
    """

    data: pd.DataFrame
    label: str
    mode: str
    checksum: str | None = None

    @property
    def start(self) -> pd.Timestamp:
        """First invoice timestamp."""
        return self.data[config.COL_DATE].min()

    @property
    def end(self) -> pd.Timestamp:
        """Last invoice timestamp."""
        return self.data[config.COL_DATE].max()


@st.cache_resource(show_spinner="Loading the bundled UCI Online Retail dataset…")
def load_bundled() -> ingestion.LoadResult:
    """Load (and cache for the server's lifetime) the bundled dataset."""
    return ingestion.load_bundled_cached()


def get_active_dataset() -> ActiveDataset | None:
    """Return the session's active dataset, defaulting to the bundled data."""
    active = st.session_state.get(config.DATASET_STATE_KEY)
    if isinstance(active, ActiveDataset):
        return active
    result = load_bundled()
    if not result.is_valid or result.data is None:
        return None
    return ActiveDataset(
        result.data, f"{config.DATASET_NAME} (bundled)", "bundled", result.checksum
    )


def show_missing_data_help() -> None:
    """Explain how to obtain the dataset when it is unavailable."""
    st.error("The bundled dataset is not available or failed validation.")
    st.markdown(
        "Download it from a terminal in the project folder:\n\n"
        "```bash\npython scripts/fetch_data.py\n```\n\n"
        f"Source: [{config.DATASET_PAGE_URL}]({config.DATASET_PAGE_URL}). You can also upload "
        "your own transactions file on the **Data Ingestion** page."
    )


@st.cache_resource(show_spinner=False)
def load_fx_rates() -> pd.DataFrame | None:
    """Load (once per server) the daily exchange rates, or ``None`` if unavailable."""
    try:
        return currency.load_rates()
    except (currency.CurrencyError, ValueError, KeyError) as exc:
        logger.warning("Exchange rates unavailable: %s", exc)
        return None


OTHER_CURRENCY = "OTHER"


def sidebar_currency() -> currency.CurrencySetting:
    """Render the display-currency picker and persist the choice across pages.

    Published rates are offered for every currency in the rate file; any other
    currency can be used with a user-supplied rate, which is labelled as such.
    """
    rates = load_fx_rates()
    codes = currency.available_currencies(rates)
    stored = st.session_state.get(config.CURRENCY_STATE_KEY, currency.CurrencySetting())
    base = config.BASE_CURRENCY

    def label(code: str) -> str:
        if code == OTHER_CURRENCY:
            return "Other (enter rate)"
        name, symbol, _ = config.CURRENCIES[code]
        return f"{code} — {name} ({symbol.strip()})"

    with st.sidebar:
        st.header("Currency")
        initial = OTHER_CURRENCY if stored.mode == "custom" else stored.code
        if initial not in codes and initial != OTHER_CURRENCY:
            initial = base
        code = st.selectbox(
            "Show amounts in",
            [*codes, OTHER_CURRENCY],
            format_func=label,
            key=_seed("currency", initial),
        )
        if rates is None:
            st.caption(
                "Published exchange rates are not installed (run "
                "`python scripts/fetch_data.py`); only custom rates are available."
            )
        setting = currency.CurrencySetting()
        if code == OTHER_CURRENCY:
            custom_code = (
                st.text_input(
                    "Currency code (ISO 4217)",
                    max_chars=3,
                    key=_seed("cur_code", stored.code if stored.mode == "custom" else ""),
                    placeholder="e.g. AED",
                )
                .strip()
                .upper()
            )
            rate = st.number_input(
                f"Rate: 1 {base} =",
                min_value=0.0,
                format="%.6f",
                key=_seed("cur_rate", stored.custom_rate or 0.0),
                help="Units of the chosen currency per 1 GBP. You are "
                "responsible for this rate; it is shown as user-supplied.",
            )
            if len(custom_code) == 3 and custom_code.isalpha() and rate > 0:
                setting = currency.CurrencySetting(custom_code, "custom", float(rate))
            else:
                st.warning(f"Enter a 3-letter code and a positive rate. Showing {base} until then.")
        elif code != base:
            mode = st.radio(
                "Conversion",
                list(config.FX_MODES),
                format_func=config.FX_MODES.get,
                key=_seed("cur_mode", stored.mode if stored.mode != "custom" else "daily"),
            )
            setting = currency.CurrencySetting(code, mode)
        st.divider()
    st.session_state[config.CURRENCY_STATE_KEY] = setting
    return setting


def _stored() -> analytics.Filters:
    stored = st.session_state.get(config.FILTER_STATE_KEY)
    return stored if isinstance(stored, analytics.Filters) else analytics.Filters()


def _seed(name: str, value: object) -> str:
    key = _W + name
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def sidebar_filters(active: ActiveDataset) -> analytics.Filters:
    """Render the global filters and persist them across pages.

    Widget values are mirrored into ``st.session_state[FILTER_STATE_KEY]``
    because Streamlit drops state for widgets not rendered on a run (i.e. when
    switching pages). Values not valid for the current dataset are discarded.
    """
    opts = analytics.filter_options(active.data)
    stored = _stored()
    lo, hi = opts["date_range"]
    new = analytics.Filters()
    with st.sidebar:
        st.header("Filters")
        st.caption("Applied to every page.")

        key = _W + "dates"
        current = st.session_state.get(key, (stored.date_from or lo, stored.date_to or hi))
        if isinstance(current, (tuple, list)) and len(current) == 2:
            st.session_state[key] = (min(max(current[0], lo), hi), max(min(current[1], hi), lo))
        else:
            st.session_state[key] = (lo, hi)
        picked = st.date_input(
            "Invoice date range", min_value=lo, max_value=hi, key=key, format="DD/MM/YYYY"
        )
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
            new.date_from = None if start == lo else start
            new.date_to = None if end == hi else end
        else:  # user is mid-selection: keep previous range
            new.date_from, new.date_to = stored.date_from, stored.date_to

        if "countries" in opts:
            key = _W + "countries"
            if key in st.session_state:
                st.session_state[key] = [c for c in st.session_state[key] if c in opts["countries"]]
            _seed("countries", [c for c in stored.countries if c in opts["countries"]])
            new.countries = st.multiselect(
                "Country", opts["countries"], key=key, placeholder="All countries"
            )

        if opts["customer_scope"]:
            new.customer_scope = st.radio(
                "Customers",
                list(SCOPE_LABELS),
                format_func=SCOPE_LABELS.get,
                key=_seed("scope", stored.customer_scope),
                horizontal=True,
                help="Guest sales have no CustomerID (about a quarter of lines in the bundled "
                "data).",
            )
        new.include_returns = st.toggle(
            "Net of cancellations & returns",
            key=_seed("returns", stored.include_returns),
            help="Off: completed sales only (sales later reversed by a matching cancellation "
            "are excluded). On: every line is netted, including cancellations and write-offs.",
        )
        new.include_non_product = st.toggle(
            "Include postage & fees",
            key=_seed("nonproduct", stored.include_non_product),
            help="Postage, bank charges, Amazon fees and manual adjustments.",
        )
        if st.button("Reset filters", width="stretch"):
            for k in [k for k in st.session_state if k.startswith(_W)]:
                del st.session_state[k]
            st.session_state[config.FILTER_STATE_KEY] = analytics.Filters()
            st.rerun()
    st.session_state[config.FILTER_STATE_KEY] = new
    return new


@dataclass
class PageData:
    """Everything a page needs: dataset, filters and both filtered views."""

    active: ActiveDataset
    filters: analytics.Filters
    lines: pd.DataFrame  # analysed lines (line_mask applied), in the display currency
    scope: pd.DataFrame  # all line types within date/country/customer scope
    money: currency.CurrencySetting  # display currency

    @property
    def cur(self) -> str:
        """Display prefix of the selected currency (e.g. ``"$"``)."""
        return self.money.symbol


def load_page_data() -> PageData:
    """Return the page's data, rendering the sidebar; stops the page if unusable."""
    active = get_active_dataset()
    if active is None:
        show_missing_data_help()
        st.stop()
    money = sidebar_currency()
    filters = sidebar_filters(active)
    with st.sidebar:
        st.divider()
        mode = "UCI benchmark" if active.mode == "bundled" else "Custom upload"
        st.caption(
            f"**Dataset:** {active.label}  \n**Mode:** {mode}  \n"
            f"**Period:** {active.start:%d %b %Y} – {active.end:%d %b %Y}"
        )
    scope = analytics.scope_filter(active.data, filters)
    rates = load_fx_rates()
    try:
        scope = currency.convert(scope, money, rates)
    except currency.CurrencyError as exc:
        st.warning(f"{exc} Showing amounts in {config.BASE_CURRENCY}.")
        money = currency.CurrencySetting()
    lines = scope.loc[analytics.line_mask(scope, filters)]
    st.caption(f"{len(lines):,} of {len(active.data):,} transaction lines · {filters.describe()}")
    if not money.is_base:
        st.caption(
            "💱 "
            + currency.describe(
                money, rates, scope[config.COL_DATE].min(), scope[config.COL_DATE].max()
            )
        )
    if lines.empty:
        st.warning("No transactions match the current filters. Widen or reset them in the sidebar.")
        st.stop()
    return PageData(active, filters, lines, scope, money)


@contextmanager
def friendly_errors(context: str) -> Iterator[None]:
    """Show a user-friendly message instead of a stack trace if the block fails."""
    try:
        yield
    except Exception:  # noqa: BLE001 - last-resort UI guard
        logger.exception("Error while %s", context)
        st.error(
            f"Something went wrong while {context}. Details were written to the server "
            "log. Try resetting the filters or reloading the page."
        )


def fmt_money(value: float | None, decimals: int = 0, cur: str = config.CURRENCY_SYMBOL) -> str:
    """Format a currency amount with prefix ``cur``, or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{cur}{value:,.{decimals}f}"


def fmt_pct(value: float | None, signed: bool = False) -> str:
    """Format a proportion as a percentage, or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def fmt_int(value: float | None) -> str:
    """Format a count with thousands separators, or an em dash when missing."""
    return "—" if value is None or pd.isna(value) else f"{int(value):,}"
