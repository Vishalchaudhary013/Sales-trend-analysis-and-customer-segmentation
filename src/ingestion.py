"""Data ingestion: loading, schema validation, cleaning, derivation and merging.

The functions in this module never raise on bad *data*; problems are collected
in a :class:`ValidationReport` so the UI can show actionable messages. They only
raise :class:`IngestionError` when a file cannot be read at all.

Two dataset modes are supported:

* ``"uci"`` - all 33 UCI Student Performance attributes are present.
* ``"custom"`` - a subset of the schema is present; downstream features are
  enabled or disabled according to :func:`assess_capabilities`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# Errors and report structures
# --------------------------------------------------------------------------- #


class IngestionError(Exception):
    """Raised when a file cannot be read or parsed into a table at all."""


@dataclass
class ValidationIssue:
    """A single validation finding.

    Attributes:
        level: ``"error"`` (blocks processing) or ``"warning"`` (informational).
        column: Column the issue relates to, or ``None`` for table-level issues.
        message: Actionable, human-readable description.
        rows: Up to 10 example row numbers (1-based, as in a spreadsheet body).
        count: Total number of affected rows.
    """

    level: str
    column: str | None
    message: str
    rows: list[int] = field(default_factory=list)
    count: int = 0


@dataclass
class ValidationReport:
    """Structured outcome of schema validation for one table.

    Attributes:
        source: File name or label the table came from.
        n_rows: Number of data rows read.
        n_columns: Number of columns read.
        mode: ``"uci"`` when every schema column is present, else ``"custom"``.
        missing_columns: Schema columns not found in the table.
        extra_columns: Columns not in the schema (kept but ignored by analytics).
        issues: All errors and warnings found.
        missing_values: Missing-value count per schema column present.
    """

    source: str
    n_rows: int
    n_columns: int
    mode: str
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return the blocking issues only."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return the non-blocking issues only."""
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when there are no blocking errors."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to plain Python types (for JSON / UI tables)."""
        out = asdict(self)
        out["is_valid"] = self.is_valid
        return out


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of the file at ``path`` (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _sniff_delimiter(text: str) -> str:
    """Guess the delimiter of a CSV sample, preferring the UCI semicolon."""
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        return config.CSV_SEPARATOR if sample.count(";") >= sample.count(",") else ","


def read_table_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame based on the file extension.

    Args:
        data: Raw file content.
        filename: Original file name; its suffix selects the parser.

    Returns:
        The parsed table with original column names (whitespace-stripped).

    Raises:
        IngestionError: If the file type is unsupported, the file is empty or
            it cannot be parsed.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(config.ALLOWED_UPLOAD_SUFFIXES)
        raise IngestionError(f"Unsupported file type '{suffix or 'none'}'. Allowed: {allowed}.")
    if not data:
        raise IngestionError(f"'{filename}' is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestionError(f"'{filename}' exceeds the {config.MAX_UPLOAD_MB} MB limit.")
    try:
        if suffix == ".csv":
            text = data.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text), sep=_sniff_delimiter(text))
        else:
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"'{filename}' is not UTF-8 encoded text.") from exc
    except Exception as exc:  # noqa: BLE001 - parser errors vary by engine
        raise IngestionError(f"Could not parse '{filename}': {exc}") from exc
    if df.shape[1] == 0 or df.empty:
        raise IngestionError(f"'{filename}' contains no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV/XLSX file from disk (see :func:`read_table_bytes`).

    Raises:
        IngestionError: If the file is missing or cannot be parsed.
    """
    if not path.exists():
        raise IngestionError(f"File not found: {path}")
    return read_table_bytes(path.read_bytes(), path.name)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _example_rows(mask: pd.Series) -> list[int]:
    """Return up to 10 1-based row numbers where ``mask`` is true."""
    idx = np.flatnonzero(mask.to_numpy())
    return [int(i) + 1 for i in idx[:10]]


def _normalise_strings(series: pd.Series) -> pd.Series:
    """Strip whitespace and stray quotes from a text column, preserving NaN."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series
    stripped = series.astype("string").str.strip().str.strip('"').str.strip("'")
    return stripped.replace("", pd.NA).astype(object).where(stripped.notna(), np.nan)


def validate(df: pd.DataFrame, source: str = "dataset") -> ValidationReport:
    """Validate a table against the UCI schema without modifying it.

    Checks performed: presence of required columns, numeric dtype and integer
    values, inclusive value ranges, allowed categorical levels and missing values.

    Args:
        df: Table as read from file.
        source: Label used in messages.

    Returns:
        A :class:`ValidationReport`; never raises for data problems.
    """
    present = [c for c in config.REQUIRED_COLUMNS if c in df.columns]
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in config.SCHEMA_BY_NAME and c != config.SUBJECT_COL]
    report = ValidationReport(
        source=source,
        n_rows=len(df),
        n_columns=df.shape[1],
        mode="uci" if not missing else "custom",
        missing_columns=missing,
        extra_columns=extra,
    )

    if not present:
        report.issues.append(
            ValidationIssue(
                "error",
                None,
                "None of the expected Student Performance columns were found. Check the "
                "delimiter and header row (expected e.g. 'school;sex;age;...;G3').",
            )
        )
        return report
    if missing:
        report.issues.append(
            ValidationIssue(
                "warning",
                None,
                f"{len(missing)} schema column(s) missing: {', '.join(missing)}. The dataset "
                "will be treated as a custom dataset with reduced functionality.",
                count=len(missing),
            )
        )
    if extra:
        report.issues.append(
            ValidationIssue(
                "warning",
                None,
                f"Unrecognised column(s) ignored by analytics: {', '.join(extra)}.",
                count=len(extra),
            )
        )

    for name in present:
        spec = config.SCHEMA_BY_NAME[name]
        raw = df[name]
        n_missing = int(raw.isna().sum())
        report.missing_values[name] = n_missing
        if n_missing:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    name,
                    f"{n_missing} missing value(s) in '{name}'. Values are not imputed; "
                    "affected rows are excluded from analyses that need this column.",
                    _example_rows(raw.isna()),
                    n_missing,
                )
            )
        if spec.kind == "numeric":
            _validate_numeric(raw, spec, report)
        else:
            _validate_categorical(raw, spec, report)

    if config.SUBJECT_COL in df.columns:
        n_missing = int(df[config.SUBJECT_COL].isna().sum())
        if n_missing:
            report.issues.append(
                ValidationIssue(
                    "error",
                    config.SUBJECT_COL,
                    f"{n_missing} row(s) have no subject label.",
                    _example_rows(df[config.SUBJECT_COL].isna()),
                    n_missing,
                )
            )
    return report


def _validate_numeric(raw: pd.Series, spec: config.ColumnSpec, report: ValidationReport) -> None:
    """Append dtype, integrality and range issues for a numeric column."""
    values = pd.to_numeric(_normalise_strings(raw), errors="coerce")
    non_numeric = values.isna() & raw.notna()
    if non_numeric.any():
        report.issues.append(
            ValidationIssue(
                "error",
                spec.name,
                f"'{spec.name}' must be numeric; {int(non_numeric.sum())} value(s) are not "
                f"(e.g. {raw[non_numeric].astype(str).head(3).tolist()}).",
                _example_rows(non_numeric),
                int(non_numeric.sum()),
            )
        )
    valid = values.notna()
    non_integer = valid & (values.round() != values)
    if non_integer.any():
        report.issues.append(
            ValidationIssue(
                "error",
                spec.name,
                f"'{spec.name}' must contain whole numbers; "
                f"{int(non_integer.sum())} value(s) have decimals.",
                _example_rows(non_integer),
                int(non_integer.sum()),
            )
        )
    out_of_range = valid & ((values < spec.min_value) | (values > spec.max_value))
    if out_of_range.any():
        report.issues.append(
            ValidationIssue(
                "error",
                spec.name,
                f"'{spec.name}' must be between {spec.min_value:g} and {spec.max_value:g}; "
                f"{int(out_of_range.sum())} value(s) fall outside "
                f"(observed {values[out_of_range].min():g}–{values[out_of_range].max():g}).",
                _example_rows(out_of_range),
                int(out_of_range.sum()),
            )
        )


def _validate_categorical(
    raw: pd.Series, spec: config.ColumnSpec, report: ValidationReport
) -> None:
    """Append invalid-level issues for a categorical column."""
    values = _normalise_strings(raw)
    invalid = values.notna() & ~values.isin(spec.allowed)
    if invalid.any():
        found = sorted(set(values[invalid].astype(str)))[:5]
        report.issues.append(
            ValidationIssue(
                "error",
                spec.name,
                f"'{spec.name}' has {int(invalid.sum())} value(s) outside the allowed set "
                f"{list(spec.allowed)} (found {found}).",
                _example_rows(invalid),
                int(invalid.sum()),
            )
        )


# --------------------------------------------------------------------------- #
# Cleaning and derivation
# --------------------------------------------------------------------------- #


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of a *validated* table.

    Text columns are whitespace/quote-stripped; numeric columns become ``int64``
    (or ``float64`` when they contain missing values, which are left as NaN -
    nothing is imputed). The input frame is not modified.
    """
    out = df.copy()
    for name in out.columns:
        spec = config.SCHEMA_BY_NAME.get(name)
        if spec is None:
            continue
        col = _normalise_strings(out[name])
        if spec.kind == "numeric":
            col = pd.to_numeric(col, errors="coerce")
            col = col.astype("int64") if col.notna().all() else col.astype("float64")
        else:
            col = col.astype(object)
        out[name] = col
    if config.SUBJECT_COL in out.columns:
        out[config.SUBJECT_COL] = _normalise_strings(out[config.SUBJECT_COL]).str.lower()
    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pass``, ``absence_band`` and ``g3_zero`` where source columns exist.

    * ``pass`` = 1 if G3 >= 10 else 0 (NaN where G3 is missing).
    * ``absence_band`` - one of 0, 1–5, 6–10, 11–20, 20+.
    * ``g3_zero`` - True where G3 == 0 (likely non-appearance at the final exam).
    """
    out = df.copy()
    if config.OUTCOME_COL in out.columns:
        g3 = out[config.OUTCOME_COL]
        passed = (g3 >= config.PASS_THRESHOLD).astype("float64").where(g3.notna())
        out[config.TARGET_COL] = passed.astype("int64") if passed.notna().all() else passed
        out["g3_zero"] = g3.eq(0)
    if "absences" in out.columns:
        out[config.ABSENCE_BAND_COL] = pd.cut(
            out["absences"],
            bins=list(config.ABSENCE_BINS),
            labels=list(config.ABSENCE_LABELS),
            ordered=True,
        )
    return out


def student_keys(df: pd.DataFrame) -> pd.Series:
    """Return a short pseudonymous key built from the demographic match columns.

    The UCI files contain no student ID; Cortez & Silva match records across
    subjects on 13 demographic attributes. The key is a truncated SHA-256 of
    those values, so it is not reversible to the raw attributes at a glance.
    Missing key columns yield an all-NaN series.
    """
    keys = [k for k in config.STUDENT_MATCH_KEYS if k in df.columns]
    if len(keys) != len(config.STUDENT_MATCH_KEYS):
        return pd.Series(np.nan, index=df.index, dtype=object)
    joined = df[keys].astype(str).agg("|".join, axis=1)
    return joined.map(lambda s: hashlib.sha256(s.encode()).hexdigest()[:12])


def flag_cross_subject_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``student_key`` and ``in_both_subjects`` columns.

    ``in_both_subjects`` is True when the same demographic key appears in more
    than one subject, i.e. the record probably belongs to a student counted
    once per subject. This is approximate (keys are not true identifiers).
    """
    out = df.copy()
    out["student_key"] = student_keys(out)
    if out["student_key"].isna().all() or config.SUBJECT_COL not in out.columns:
        out["in_both_subjects"] = False
        return out
    n_subjects = out.groupby("student_key")[config.SUBJECT_COL].transform("nunique")
    out["in_both_subjects"] = n_subjects > 1
    return out


def merge_subjects(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-subject tables, adding a ``subject`` column.

    Args:
        frames: Mapping of subject name (e.g. ``"math"``) to its table.

    Returns:
        One table with a fresh RangeIndex, subjects in the given order.
    """
    parts = [f.assign(**{config.SUBJECT_COL: name}) for name, f in frames.items()]
    return pd.concat(parts, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Run cleaning, derivation and duplicate flagging on a validated table."""
    return flag_cross_subject_duplicates(add_derived_columns(clean(df)))


# --------------------------------------------------------------------------- #
# High-level entry points
# --------------------------------------------------------------------------- #


@dataclass
class LoadResult:
    """Outcome of loading a dataset.

    Attributes:
        data: Prepared table, or ``None`` if validation failed.
        reports: One validation report per source file.
        checksums: SHA-256 per source file name.
    """

    data: pd.DataFrame | None
    reports: list[ValidationReport]
    checksums: dict[str, str]

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when every source passed validation."""
        return self.data is not None and all(r.is_valid for r in self.reports)


def load_uci_dataset(raw_dir: Path = config.RAW_DIR) -> LoadResult:
    """Load, validate and merge the bundled UCI math and Portuguese files.

    Args:
        raw_dir: Folder containing ``student-mat.csv`` and ``student-por.csv``.

    Returns:
        A :class:`LoadResult`; ``data`` is ``None`` if any file is missing or invalid.
    """
    frames: dict[str, pd.DataFrame] = {}
    reports: list[ValidationReport] = []
    checksums: dict[str, str] = {}
    for subject, filename in config.RAW_FILES.items():
        path = raw_dir / filename
        try:
            df = read_table(path)
        except IngestionError as exc:
            reports.append(
                ValidationReport(
                    filename, 0, 0, "uci", issues=[ValidationIssue("error", None, str(exc))]
                )
            )
            continue
        checksums[filename] = sha256_file(path)
        report = validate(df, source=filename)
        expected = config.EXPECTED_ROWS[subject]
        if len(df) != expected:
            report.issues.append(
                ValidationIssue(
                    "warning",
                    None,
                    f"Expected {expected} rows for the UCI {subject} file, found {len(df)}.",
                )
            )
        if report.mode != "uci":
            report.issues.append(
                ValidationIssue("error", None, "Bundled UCI file is missing schema columns.")
            )
        reports.append(report)
        frames[subject] = df
    if len(frames) != len(config.RAW_FILES) or not all(r.is_valid for r in reports):
        return LoadResult(None, reports, checksums)
    return LoadResult(prepare(merge_subjects(frames)), reports, checksums)


def load_custom_table(
    df: pd.DataFrame, source: str, subject: str | None = None
) -> tuple[pd.DataFrame | None, ValidationReport]:
    """Validate and prepare a user-supplied table.

    Args:
        df: Raw uploaded table.
        source: File name for messages.
        subject: Subject label to assign when the table has no ``subject`` column.
            When omitted, rows are labelled ``"unspecified"`` and a warning is added.

    Returns:
        ``(prepared_table, report)``; the table is ``None`` if validation failed.
    """
    report = validate(df, source=source)
    if not report.is_valid:
        return None, report
    work = df.copy()
    if config.SUBJECT_COL not in work.columns:
        label = subject.strip().lower() if subject and subject.strip() else "unspecified"
        if label == "unspecified":
            report.issues.append(
                ValidationIssue(
                    "warning",
                    config.SUBJECT_COL,
                    "No 'subject' column found; all rows are labelled 'unspecified'.",
                )
            )
        work[config.SUBJECT_COL] = label
    return prepare(work), report


def save_upload(data: bytes, filename: str, uploads_dir: Path = config.UPLOADS_DIR) -> Path:
    """Persist the original uploaded bytes unchanged, named by checksum.

    Returns:
        Path of the stored copy (``<sha256[:16]>_<safe-name>``).
    """
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    target = uploads_dir / f"{sha256_bytes(data)[:16]}_{safe}"
    if not target.exists():
        target.write_bytes(data)
    return target


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_FILE) -> Path:
    """Write the prepared dataset to ``data/processed`` as CSV and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Capabilities (custom-dataset compatibility) and quality summary
# --------------------------------------------------------------------------- #

# Feature -> columns it requires.
CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "KPIs and grade distribution": ("G3",),
    "Pass rate by absence band": ("G3", "absences"),
    "Mean G3 by study time": ("G3", "studytime"),
    "Mean G3 by past failures": ("G3", "failures"),
    "G2 vs G3 scatter": ("G2", "G3"),
    "G3 by higher-education aspiration": ("G3", "higher"),
    "Model setting A (with G1, G2)": config.REQUIRED_COLUMNS,
    "Model setting B (early warning)": tuple(
        c for c in config.REQUIRED_COLUMNS if c not in config.PRIOR_GRADE_COLUMNS
    ),
}


def assess_capabilities(columns: list[str] | pd.Index) -> dict[str, list[str]]:
    """Map each dashboard feature to the required columns that are missing.

    An empty list means the feature is available. Columns are never substituted.
    """
    cols = set(columns)
    return {
        feature: [c for c in needed if c not in cols]
        for feature, needed in CAPABILITY_REQUIREMENTS.items()
    }


def data_quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Compute data-quality statistics for a prepared dataset.

    Returns:
        A dictionary with record counts per subject, approximate unique students,
        cross-subject duplicates, G3 == 0 counts, missing values and pass rates.
        Every figure is computed from ``df``.
    """
    summary: dict[str, Any] = {"n_records": int(len(df))}
    if config.SUBJECT_COL in df.columns:
        summary["records_by_subject"] = {
            str(k): int(v) for k, v in df[config.SUBJECT_COL].value_counts(sort=False).items()
        }
    if "student_key" in df.columns and df["student_key"].notna().any():
        summary["approx_unique_students"] = int(df["student_key"].nunique())
        summary["records_in_both_subjects"] = int(df["in_both_subjects"].sum())
        summary["students_in_both_subjects"] = int(
            df.loc[df["in_both_subjects"], "student_key"].nunique()
        )
        if config.SUBJECT_COL in df.columns:
            counts = df.groupby(["student_key", config.SUBJECT_COL]).size().unstack(fill_value=0)
            summary["cross_subject_match_pairs"] = int(
                counts[(counts > 0).sum(axis=1) > 1].prod(axis=1).sum()
            )
            summary["keys_repeated_within_subject"] = int((counts > 1).any(axis=1).sum())
    if "g3_zero" in df.columns:
        summary["g3_zero_records"] = int(df["g3_zero"].sum())
        if config.SUBJECT_COL in df.columns:
            summary["g3_zero_by_subject"] = {
                str(k): int(v)
                for k, v in df.groupby(config.SUBJECT_COL, sort=False)["g3_zero"].sum().items()
            }
    if config.TARGET_COL in df.columns:
        summary["pass_rate"] = float(df[config.TARGET_COL].mean())
        if config.SUBJECT_COL in df.columns:
            summary["pass_rate_by_subject"] = {
                str(k): float(v)
                for k, v in df.groupby(config.SUBJECT_COL, sort=False)[config.TARGET_COL]
                .mean()
                .items()
            }
    schema_cols = [c for c in config.REQUIRED_COLUMNS if c in df.columns]
    summary["missing_values_total"] = int(df[schema_cols].isna().sum().sum())
    summary["exact_duplicate_rows"] = int(df.duplicated(subset=schema_cols + ["subject"]).sum())
    return summary


def format_quality_summary(summary: dict[str, Any]) -> str:
    """Render :func:`data_quality_summary` output as a plain-text block."""
    lines = ["Data-quality summary", "=" * 20, f"Records: {summary['n_records']}"]
    for subj, n in summary.get("records_by_subject", {}).items():
        lines.append(f"  - {subj}: {n}")
    if "approx_unique_students" in summary:
        lines += [
            f"Approx. unique students (13-attribute match): {summary['approx_unique_students']}",
            f"Students appearing in both subjects: {summary['students_in_both_subjects']} "
            f"({summary['records_in_both_subjects']} records)",
        ]
        if "cross_subject_match_pairs" in summary:
            lines += [
                f"Cross-subject match pairs (Cortez & Silva merge): "
                f"{summary['cross_subject_match_pairs']}",
                f"Match keys repeated within one subject (ambiguous): "
                f"{summary['keys_repeated_within_subject']}",
            ]
    if "g3_zero_records" in summary:
        by = ", ".join(f"{k}: {v}" for k, v in summary.get("g3_zero_by_subject", {}).items())
        lines.append(
            f"G3 == 0 records (likely non-appearance): {summary['g3_zero_records']} ({by})"
        )
    if "pass_rate" in summary:
        by = ", ".join(f"{k}: {v:.1%}" for k, v in summary.get("pass_rate_by_subject", {}).items())
        lines.append(f"Pass rate (G3 >= 10): {summary['pass_rate']:.1%} ({by})")
    lines += [
        f"Missing values (schema columns): {summary['missing_values_total']}",
        f"Exact duplicate rows within a subject: {summary['exact_duplicate_rows']}",
    ]
    return "\n".join(lines)
