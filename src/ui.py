"""Shared Streamlit UI glue: page setup, cached data access, global filters.

This module contains presentation code only. It calls :mod:`src.ingestion` and
:mod:`src.analytics` for all computation so pages stay free of business logic.
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

_WIDGET_PREFIX = "_filter_widget_"


@dataclass
class ActiveDataset:
    """The dataset every page works on.

    Attributes:
        data: Prepared table (see :func:`src.ingestion.prepare`).
        label: Human-readable source name.
        mode: ``"uci"`` (bundled benchmark) or ``"custom"`` (user upload).
    """

    data: pd.DataFrame
    label: str
    mode: str


def page_setup(title: str, icon: str = "📊") -> None:
    """Configure the page and render its title."""
    st.set_page_config(page_title=f"{title} · {config.APP_TITLE}", page_icon=icon, layout="wide")
    st.title(title)


@st.cache_data(show_spinner="Loading the bundled UCI dataset…")
def load_bundled_dataset() -> tuple[pd.DataFrame | None, list[dict]]:
    """Load and cache the bundled UCI dataset and its validation reports."""
    result = ingestion.load_uci_dataset()
    return result.data, [r.to_dict() for r in result.reports]


def get_active_dataset() -> ActiveDataset | None:
    """Return the session's active dataset, defaulting to the bundled UCI data."""
    active = st.session_state.get(config.DATASET_STATE_KEY)
    if isinstance(active, ActiveDataset):
        return active
    data, _ = load_bundled_dataset()
    if data is None:
        return None
    return ActiveDataset(data=data, label="UCI Student Performance (bundled)", mode="uci")


def show_missing_data_help() -> None:
    """Explain how to obtain the dataset when it is not available."""
    st.error("The bundled UCI dataset is not available or failed validation.")
    st.markdown(
        "Download it with:\n\n```bash\npython scripts/fetch_data.py\n```\n\n"
        f"or place `student-mat.csv` and `student-por.csv` in `{config.RAW_DIR.name}/` "
        f"(see [{config.DATASET_PAGE_URL}]({config.DATASET_PAGE_URL})). "
        "You can also upload your own file on the **Data Ingestion** page."
    )


def _stored_filters() -> analytics.Filters:
    stored = st.session_state.get(config.FILTER_STATE_KEY)
    return stored if isinstance(stored, analytics.Filters) else analytics.Filters()


def sidebar_filters(df: pd.DataFrame) -> analytics.Filters:
    """Render the global filters in the sidebar and persist them across pages.

    Widget values are mirrored into ``st.session_state[FILTER_STATE_KEY]`` so
    they survive page switches (Streamlit discards widget state for widgets that
    are not rendered on a run). Stored values unavailable in ``df`` are dropped.
    """
    options = analytics.filter_options(df)
    stored = _stored_filters()

    def seed(name: str, value: object) -> str:
        key = _WIDGET_PREFIX + name
        if key not in st.session_state:
            st.session_state[key] = value
        return key

    labels = {"subjects": "Subject", "schools": "School", "sexes": "Sex", "addresses": "Address"}
    pretty = {
        "subjects": lambda v: config.VALUE_LABELS["subject"].get(v, v),
        "schools": lambda v: {"GP": "Gabriel Pereira (GP)", "MS": "Mousinho da Silveira (MS)"}.get(
            v, v
        ),
        "sexes": lambda v: {"F": "Female", "M": "Male"}.get(v, v),
        "addresses": lambda v: {"U": "Urban", "R": "Rural"}.get(v, v),
    }
    new = analytics.Filters()
    with st.sidebar:
        st.header("Filters")
        st.caption("Applied to every page. Leave a filter empty to include all values.")
        for name, label in labels.items():
            if name not in options:
                continue
            key = _WIDGET_PREFIX + name
            if key in st.session_state:
                st.session_state[key] = [v for v in st.session_state[key] if v in options[name]]
            seed(name, [v for v in getattr(stored, name) if v in options[name]])
            setattr(
                new,
                name,
                st.multiselect(
                    label, options[name], key=key, placeholder="All", format_func=pretty[name]
                ),
            )
        if "age_range" in options:
            lo, hi = options["age_range"]
            key = _WIDGET_PREFIX + "age_range"
            current = st.session_state.get(key, stored.age_range or (lo, hi))
            st.session_state[key] = (max(lo, min(current[0], hi)), min(hi, max(current[1], lo)))
            if lo < hi:
                rng = st.slider("Age range", lo, hi, key=key)
                new.age_range = None if rng == (lo, hi) else (int(rng[0]), int(rng[1]))
        if "exclude_g3_zero" in options:
            new.exclude_g3_zero = st.toggle(
                f"Exclude G3 = 0 ({options['exclude_g3_zero']} records)",
                key=seed("exclude_g3_zero", stored.exclude_g3_zero),
                help="G3 = 0 usually means the student did not sit the final exam.",
            )
        if st.button("Reset filters", use_container_width=True):
            for k in list(st.session_state):
                if k.startswith(_WIDGET_PREFIX):
                    del st.session_state[k]
            st.session_state[config.FILTER_STATE_KEY] = analytics.Filters()
            st.rerun()
    st.session_state[config.FILTER_STATE_KEY] = new
    return new


def load_page_data() -> tuple[ActiveDataset, pd.DataFrame, analytics.Filters]:
    """Return the active dataset, its filtered view and the filters, or stop the page.

    Renders the sidebar filters and a dataset badge. Calls ``st.stop()`` with a
    helpful message when no data is available or the filters exclude everything.
    """
    active = get_active_dataset()
    if active is None:
        show_missing_data_help()
        st.stop()
    filters = sidebar_filters(active.data)
    with st.sidebar:
        st.divider()
        mode = "UCI benchmark" if active.mode == "uci" else "Custom dataset"
        st.caption(f"**Dataset:** {active.label}  \n**Mode:** {mode}")
    filtered = analytics.apply_filters(active.data, filters)
    st.caption(
        f"Showing **{len(filtered):,}** of {len(active.data):,} records · {filters.describe()}"
    )
    if filtered.empty:
        st.warning("No records match the current filters. Widen or reset them in the sidebar.")
        st.stop()
    return active, filtered, filters


@contextmanager
def friendly_errors(context: str) -> Iterator[None]:
    """Show a user-friendly message instead of a stack trace if the block fails."""
    try:
        yield
    except Exception:  # noqa: BLE001 - last-resort UI guard
        logger.exception("Error while %s", context)
        st.error(
            f"Something went wrong while {context}. The details were written to the server "
            "log. Try resetting the filters or reloading the page."
        )


def fmt_pct(value: float | None) -> str:
    """Format a proportion as a percentage, or an em dash when missing."""
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


def fmt_num(value: float | None, digits: int = 2) -> str:
    """Format a number with fixed decimals, or an em dash when missing."""
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"
