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

from src import analytics, config, ingestion

logger = logging.getLogger(__name__)
_W = "_filter_widget_"  # widget-key prefix
SCOPE_LABELS = {"all": "All customers", "identified": "Identified only", "guest": "Guests only"}


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
    return ActiveDataset(result.data, f"{config.DATASET_NAME} (bundled)", "bundled",
                         result.checksum)  # fmt: skip


def show_missing_data_help() -> None:
    """Explain how to obtain the dataset when it is unavailable."""
    st.error("The bundled dataset is not available or failed validation.")
    st.markdown(
        "Download it from a terminal in the project folder:\n\n"
        "```bash\npython scripts/fetch_data.py\n```\n\n"
        f"Source: [{config.DATASET_PAGE_URL}]({config.DATASET_PAGE_URL}). You can also upload "
        "your own transactions file on the **Data Ingestion** page."
    )


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
        picked = st.date_input("Invoice date range", min_value=lo, max_value=hi, key=key,
                               format="DD/MM/YYYY")  # fmt: skip
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
            new.countries = st.multiselect("Country", opts["countries"], key=key,
                                           placeholder="All countries")  # fmt: skip

        if opts["customer_scope"]:
            new.customer_scope = st.radio(
                "Customers", list(SCOPE_LABELS), format_func=SCOPE_LABELS.get,
                key=_seed("scope", stored.customer_scope), horizontal=True,
                help="Guest sales have no CustomerID (about a quarter of lines in the bundled "
                "data).",
            )  # fmt: skip
        new.include_returns = st.toggle(
            "Net of cancellations & returns", key=_seed("returns", stored.include_returns),
            help="Off: completed sales only (sales later reversed by a matching cancellation "
            "are excluded). On: every line is netted, including cancellations and write-offs.",
        )  # fmt: skip
        new.include_non_product = st.toggle(
            "Include postage & fees", key=_seed("nonproduct", stored.include_non_product),
            help="Postage, bank charges, Amazon fees and manual adjustments.",
        )  # fmt: skip
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
    lines: pd.DataFrame  # analysed lines (line_mask applied)
    scope: pd.DataFrame  # all line types within date/country/customer scope


def load_page_data() -> PageData:
    """Return the page's data, rendering the sidebar; stops the page if unusable."""
    active = get_active_dataset()
    if active is None:
        show_missing_data_help()
        st.stop()
    filters = sidebar_filters(active)
    with st.sidebar:
        st.divider()
        mode = "UCI benchmark" if active.mode == "bundled" else "Custom upload"
        st.caption(f"**Dataset:** {active.label}  \n**Mode:** {mode}  \n"
                   f"**Period:** {active.start:%d %b %Y} – {active.end:%d %b %Y}")  # fmt: skip
    scope = analytics.scope_filter(active.data, filters)
    lines = scope.loc[analytics.line_mask(scope, filters)]
    st.caption(f"{len(lines):,} of {len(active.data):,} transaction lines · {filters.describe()}")
    if lines.empty:
        st.warning("No transactions match the current filters. Widen or reset them in the sidebar.")
        st.stop()
    return PageData(active, filters, lines, scope)


@contextmanager
def friendly_errors(context: str) -> Iterator[None]:
    """Show a user-friendly message instead of a stack trace if the block fails."""
    try:
        yield
    except Exception:  # noqa: BLE001 - last-resort UI guard
        logger.exception("Error while %s", context)
        st.error(f"Something went wrong while {context}. Details were written to the server "
                 "log. Try resetting the filters or reloading the page.")  # fmt: skip


def fmt_money(value: float | None, decimals: int = 0) -> str:
    """Format a currency amount, or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{config.CURRENCY_SYMBOL}{value:,.{decimals}f}"


def fmt_pct(value: float | None, signed: bool = False) -> str:
    """Format a proportion as a percentage, or an em dash when missing."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def fmt_int(value: float | None) -> str:
    """Format a count with thousands separators, or an em dash when missing."""
    return "—" if value is None or pd.isna(value) else f"{int(value):,}"
