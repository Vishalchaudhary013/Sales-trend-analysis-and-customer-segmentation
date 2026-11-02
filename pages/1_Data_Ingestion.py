"""Data Ingestion: bundled dataset or upload, validation report, preview, history."""

import pandas as pd
import streamlit as st

from src import config, ingestion, storage, ui

ui.page_setup("Data Ingestion", "📥")


def show_report(report: ingestion.ValidationReport, log: ingestion.CleaningLog | None) -> None:
    """Render a validation report: status, errors, warnings, missing values."""
    if report.is_valid:
        st.success(
            f"✔ Valid · {report.n_rows:,} rows · {report.n_columns} columns · "
            f"date format `{report.date_format}`"
        )
    else:
        st.error(f"✖ Not usable · {len(report.errors)} blocking problem(s) found.")
    for issue in report.errors:
        st.error(
            f"**{issue.column or 'File'}**: {issue.message}"
            + (f" Example rows: {issue.rows}." if issue.rows else "")
        )
    if report.warnings:
        with st.expander(
            f"{len(report.warnings)} warning(s) — data kept and flagged",
            expanded=not report.is_valid,
        ):
            for issue in report.warnings:
                st.warning(
                    f"**{issue.column or 'File'}**: {issue.message}"
                    + (f" Example rows: {issue.rows}." if issue.rows else "")
                )
    if log is not None:
        st.caption(
            f"Rows read {log.rows_in:,} · unparseable rows dropped {log.dropped_invalid:,} "
            f"· exact duplicates dropped {log.dropped_duplicates:,} · rows kept "
            f"{log.rows_out:,}"
        )
    if report.missing_values:
        miss = pd.DataFrame(
            {"column": list(report.missing_values), "missing": list(report.missing_values.values())}
        )
        miss["share"] = (miss["missing"] / max(report.n_rows, 1)).map("{:.1%}".format)
        with st.expander("Missing values by column"):
            st.dataframe(miss, hide_index=True, width="stretch")


def show_preview(data: pd.DataFrame) -> None:
    """Show the first rows, masking customer IDs unless the user opts in."""
    reveal = st.toggle(
        "Show full customer IDs",
        value=False,
        key="ing_reveal",
        help="Customer IDs are masked by default to limit exposure of customer identifiers.",
    )
    view = data.head(100).copy()
    if not reveal and config.COL_CUSTOMER in view.columns:
        view[config.COL_CUSTOMER] = view[config.COL_CUSTOMER].map(
            lambda c: c if pd.isna(c) else "•••" + str(c)[-2:]
        )
    st.dataframe(view, hide_index=True, width="stretch")


def show_capabilities(data: pd.DataFrame) -> None:
    """List features available / unavailable for the dataset."""
    caps = ingestion.assess_capabilities(data)
    for feature, reasons in caps.items():
        if reasons:
            st.markdown(f"- ✖ **{feature}** — unavailable: {'; '.join(reasons)}")
        else:
            st.markdown(f"- ✔ **{feature}**")


active = ui.get_active_dataset()
current = "Custom upload" if active is not None and active.mode == "custom" else "Bundled"
st.info(
    f"Active dataset: **{active.label if active else 'none'}** ({current}). "
    "All pages use the active dataset."
)

source = st.radio(
    "Data source",
    ["Bundled UCI Online Retail dataset", "Upload my own file"],
    horizontal=True,
    key="ing_source",
)

with ui.friendly_errors("loading the data"):
    if source.startswith("Bundled"):
        result = ui.load_bundled()
        st.subheader("Validation report")
        show_report(result.report, result.log)
        if result.is_valid and result.data is not None:
            if (
                active is not None
                and active.mode == "custom"
                and st.button("Switch back to the bundled dataset", type="primary")
            ):
                st.session_state.pop(config.DATASET_STATE_KEY, None)
                st.session_state.pop(config.FILTER_STATE_KEY, None)
                st.rerun()
            summary = ingestion.data_quality_summary(result.data, result.log)
            st.subheader("Data-quality summary")
            st.code(ingestion.format_quality_summary(summary), language=None)
            st.subheader("Supported features")
            show_capabilities(result.data)
            st.subheader("Preview (first 100 cleaned rows)")
            show_preview(result.data)
        else:
            ui.show_missing_data_help()
    else:
        st.markdown(
            f"Upload a **CSV or XLSX** file of transaction lines (max {config.MAX_UPLOAD_MB} MB). "
            f"Required columns: `{'`, `'.join(config.REQUIRED_COLUMNS)}`; optional: "
            f"`{config.COL_DESC}`, `{config.COL_COUNTRY}`. Common alternative names (e.g. "
            "`Invoice`, `Price`, `Customer ID`) are mapped and reported; nothing is guessed."
        )
        upload = st.file_uploader("Transactions file", type=["csv", "xlsx"], key="ing_file")
        if upload is not None:
            data = upload.getvalue()
            checksum = ingestion.sha256_bytes(data)
            cache = st.session_state.setdefault("_uploads", {})
            if checksum not in cache:
                with st.spinner("Validating…"):
                    stored = ingestion.save_upload(data, upload.name)
                    try:
                        raw = ingestion.read_table_bytes(data, upload.name)
                        res = ingestion.load_table(raw, upload.name, checksum)
                    except ingestion.IngestionError as exc:
                        rep = ingestion.ValidationReport(upload.name, 0, 0)
                        rep.add("error", None, str(exc))
                        res = ingestion.LoadResult(None, rep, None, checksum)
                    storage.record_upload(
                        upload.name,
                        res.report.n_rows,
                        checksum,
                        res.is_valid,
                        "custom",
                        len(res.report.errors),
                        len(res.report.warnings),
                        str(stored.relative_to(config.ROOT_DIR)),
                    )
                cache[checksum] = res
            res = cache[checksum]
            st.caption(f"SHA-256 `{checksum}` · original file preserved unchanged.")
            st.subheader("Validation report")
            show_report(res.report, res.log)
            if res.is_valid and res.data is not None:
                st.subheader("Supported features")
                show_capabilities(res.data)
                st.subheader("Preview (first 100 cleaned rows)")
                show_preview(res.data)
                if st.button("Use this dataset in the dashboard", type="primary"):
                    st.session_state[config.DATASET_STATE_KEY] = ui.ActiveDataset(
                        res.data, f"Upload: {upload.name}", "custom", checksum
                    )
                    st.session_state.pop(config.FILTER_STATE_KEY, None)
                    for k in [k for k in st.session_state if k.startswith("_filter_widget_")]:
                        del st.session_state[k]
                    st.success(
                        "Active dataset switched. Other pages now use this upload; the "
                        "bundled UCI benchmark is unchanged."
                    )
            else:
                st.info("Fix the problems above and upload the file again.")

st.divider()
st.subheader("Upload history")
history = storage.upload_history()
if history.empty:
    st.caption("No files uploaded yet.")
else:
    history["valid"] = history["valid"].map({1: "✔ valid", 0: "✖ invalid"})
    st.dataframe(history.drop(columns=["id"]), hide_index=True, width="stretch")
