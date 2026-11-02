"""Transaction data ingestion: reading, schema validation, cleaning and flagging.

Design rules:

* Bad *data* never raises; findings are collected in a :class:`ValidationReport`
  with actionable messages. :class:`IngestionError` is raised only when a file
  cannot be read as a table at all.
* Nothing is imputed or silently substituted. Column renames come only from
  the explicit aliases in :mod:`src.config` and are listed in the report.
  Rows that cannot be parsed are dropped *and counted*.
* Questionable but real rows (cancellations, zero prices, non-product codes,
  missing customer IDs) are kept and flagged so analyses can choose.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config

C = config  # short alias for column-name constants


# --------------------------------------------------------------------------- #
# Errors and reports
# --------------------------------------------------------------------------- #


class IngestionError(Exception):
    """Raised when a file cannot be read or parsed into a table at all."""


@dataclass
class ValidationIssue:
    """One validation finding.

    Attributes:
        level: ``"error"`` (blocks processing) or ``"warning"`` (informational).
        column: Related column, or ``None`` for table-level findings.
        message: Actionable, human-readable description.
        rows: Up to 10 example row numbers (1-based, data rows only).
        count: Total number of affected rows.
    """

    level: str
    column: str | None
    message: str
    rows: list[int] = field(default_factory=list)
    count: int = 0


@dataclass
class ValidationReport:
    """Structured outcome of validating one table.

    Attributes:
        source: File name or label.
        n_rows: Data rows read.
        n_columns: Columns read.
        column_mapping: Source name -> canonical name for alias renames.
        missing_required: Required columns that are absent.
        missing_optional: Optional columns that are absent.
        extra_columns: Columns outside the schema (ignored).
        date_format: Date format detected for ``InvoiceDate`` (or ``None``).
        invalid_rows: Rows with unparseable required values (dropped on prepare).
        issues: All errors and warnings.
        missing_values: Missing-value count per schema column present.
    """

    source: str
    n_rows: int
    n_columns: int
    column_mapping: dict[str, str] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    date_format: str | None = None
    invalid_rows: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Return blocking issues."""
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Return non-blocking issues."""
        return [i for i in self.issues if i.level == "warning"]

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when there are no blocking errors."""
        return not self.errors

    def add(self, level: str, column: str | None, message: str, mask: pd.Series | None = None):
        """Append an issue, deriving example rows and count from ``mask``."""
        rows, count = ([], 0) if mask is None else (_example_rows(mask), int(mask.sum()))
        self.issues.append(ValidationIssue(level, column, message, rows, count))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain Python types (JSON / UI tables)."""
        out = asdict(self)
        out["is_valid"] = self.is_valid
        return out


def _example_rows(mask: pd.Series) -> list[int]:
    return [int(i) + 1 for i in np.flatnonzero(mask.to_numpy())[:10]]


# --------------------------------------------------------------------------- #
# Hashing and reading
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(data: bytes) -> str:
    for encoding in config.CSV_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise IngestionError("File is not valid UTF-8 or Latin-1 text.")  # pragma: no cover


def _sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:16384], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_table_bytes(data: bytes, filename: str) -> pd.DataFrame:
    """Parse CSV/XLSX bytes into a DataFrame of raw (string-preserving) values.

    Identifier-like columns are read as text so codes such as ``"85123A"`` or
    leading zeros survive. Column names are whitespace-stripped.

    Raises:
        IngestionError: Unsupported type, empty/oversized file, or parse failure.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_UPLOAD_SUFFIXES:
        allowed = ", ".join(config.ALLOWED_UPLOAD_SUFFIXES)
        raise IngestionError(f"Unsupported file type '{suffix or 'none'}'. Allowed: {allowed}.")
    if not data:
        raise IngestionError(f"'{filename}' is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise IngestionError(f"'{filename}' exceeds the {config.MAX_UPLOAD_MB} MB limit.")
    text_cols = {c: str for spec in config.SCHEMA if spec.kind == "string"
                 for c in (spec.name, *spec.aliases)}  # fmt: skip
    try:
        if suffix == ".csv":
            text = _decode(data)
            df = pd.read_csv(
                io.StringIO(text), sep=_sniff_delimiter(text), dtype=text_cols, low_memory=False
            )
        else:
            df = pd.read_excel(io.BytesIO(data), engine="openpyxl", dtype=text_cols)
    except IngestionError:
        raise
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
# Column resolution and parsing helpers
# --------------------------------------------------------------------------- #


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to canonical names using exact names and declared aliases.

    Matching ignores case, spaces and punctuation (``"Customer ID"`` ->
    ``CustomerID``). Only names listed in the schema are mapped; nothing is guessed.

    Returns:
        ``(renamed_copy, mapping)`` where ``mapping`` holds only actual renames.
    """
    lookup: dict[str, str] = {}
    for spec in config.SCHEMA:
        for name in (spec.name, *spec.aliases):
            lookup.setdefault(_norm(name), spec.name)
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for col in df.columns:
        target = lookup.get(_norm(col))
        if target and target not in taken:
            taken.add(target)
            if col != target:
                mapping[col] = target
    return df.rename(columns=mapping), mapping


def _strip_text(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out == "")


def detect_date_format(series: pd.Series) -> tuple[str | None, list[str]]:
    """Pick the configured format that parses the most non-null values.

    Returns:
        ``(format, ambiguous_alternatives)``. ``format`` is ``None`` when the best
        candidate parses less than :data:`config.DATE_PARSE_MIN_SHARE` of values.
        Alternatives are other formats parsing equally well (e.g. day/month vs
        month/day); the first in :data:`config.DATE_FORMATS` order wins.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return "native", []
    values = _strip_text(series).dropna()
    if values.empty:
        return None, []
    sample = values.sample(min(len(values), 20_000), random_state=0)
    shares = {
        fmt: float(pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean())
        for fmt in config.DATE_FORMATS
    }
    best = max(shares.values())
    if best < config.DATE_PARSE_MIN_SHARE:
        return None, []
    tied = [fmt for fmt, share in shares.items() if share >= best - 1e-9]
    return tied[0], tied[1:]


def parse_dates(series: pd.Series, fmt: str | None) -> pd.Series:
    """Parse ``series`` with a detected format (NaT where unparseable)."""
    if fmt == "native" or pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    if fmt is None:
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    return pd.to_datetime(_strip_text(series), format=fmt, errors="coerce")


def normalise_customer_ids(series: pd.Series) -> pd.Series:
    """Return customer IDs as clean strings (``"17850.0"`` -> ``"17850"``)."""
    text = _strip_text(series)
    numeric = pd.to_numeric(text, errors="coerce")
    integral = numeric.notna() & (numeric == numeric.round())
    as_int = numeric.where(integral).astype("Int64").astype("string")
    return as_int.where(integral, text)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(df: pd.DataFrame, source: str = "dataset") -> ValidationReport:
    """Validate a raw table against the transaction schema without modifying it.

    Checks: required/optional columns (after alias mapping), parseability of
    dates/quantities/prices, missing values, and business-rule anomalies
    (cancellations, adjustments, non-positive prices, negative quantities on
    normal invoices, duplicates, non-product codes).

    Args:
        df: Table as read from file.
        source: Label used in messages.

    Returns:
        A :class:`ValidationReport`; never raises for data problems.
    """
    work, mapping = resolve_columns(df)
    report = ValidationReport(source=source, n_rows=len(work), n_columns=work.shape[1],
                              column_mapping=mapping)  # fmt: skip
    report.missing_required = [c for c in config.REQUIRED_COLUMNS if c not in work.columns]
    report.missing_optional = [
        c for c in config.ALL_COLUMNS if c not in work.columns and c not in config.REQUIRED_COLUMNS
    ]
    report.extra_columns = [c for c in work.columns if c not in config.SCHEMA_BY_NAME]
    if mapping:
        pairs = ", ".join(f"'{k}' → '{v}'" for k, v in mapping.items())
        report.add("warning", None, f"Columns renamed via documented aliases: {pairs}.")
    if report.missing_required:
        report.add(
            "error",
            None,
            f"Missing required column(s): {', '.join(report.missing_required)}. Expected "
            f"{', '.join(config.REQUIRED_COLUMNS)} (see Methodology for accepted aliases).",
        )
        return report
    if report.missing_optional:
        absent = ", ".join(report.missing_optional)
        report.add("warning", None, f"Optional column(s) absent: {absent}; related views "
                   "are disabled.")  # fmt: skip
    if report.extra_columns:
        report.add("warning", None, f"Unrecognised column(s) ignored: "
                   f"{', '.join(report.extra_columns)}.")  # fmt: skip

    for name in config.ALL_COLUMNS:
        if name in work.columns:
            report.missing_values[name] = int(work[name].isna().sum())

    invalid = pd.Series(False, index=work.index)
    for name in (C.COL_INVOICE, C.COL_STOCK):
        miss = _strip_text(work[name]).isna()
        if miss.any():
            report.add("warning", name, f"{int(miss.sum())} row(s) have no {name}; they will "
                       "be dropped.", miss)  # fmt: skip
        invalid |= miss

    fmt, alternatives = detect_date_format(work[C.COL_DATE])
    report.date_format = fmt
    if fmt is None:
        report.add("error", C.COL_DATE, f"Could not recognise the {C.COL_DATE} format. "
                   f"Supported: {', '.join(config.DATE_FORMATS)}.")  # fmt: skip
        return report
    if alternatives:
        report.add("warning", C.COL_DATE, f"Dates are ambiguous: '{fmt}' was used, but "
                   f"{alternatives} would also parse. Check day/month order.")  # fmt: skip
    dates = parse_dates(work[C.COL_DATE], fmt)
    bad_date = dates.isna()
    if bad_date.any():
        report.add("warning", C.COL_DATE, f"{int(bad_date.sum())} {C.COL_DATE} value(s) "
                   "missing or unparseable; rows will be dropped.", bad_date)  # fmt: skip
    invalid |= bad_date

    qty = pd.to_numeric(work[C.COL_QTY], errors="coerce")
    bad_qty = qty.isna() | (qty.notna() & (qty != qty.round()))
    if bad_qty.any():
        report.add("warning", C.COL_QTY, f"{int(bad_qty.sum())} {C.COL_QTY} value(s) missing, "
                   "non-numeric or fractional; rows will be dropped.", bad_qty)  # fmt: skip
    invalid |= bad_qty
    price = pd.to_numeric(work[C.COL_PRICE], errors="coerce")
    bad_price = price.isna()
    if bad_price.any():
        report.add("warning", C.COL_PRICE, f"{int(bad_price.sum())} {C.COL_PRICE} value(s) "
                   "missing or non-numeric; rows will be dropped.", bad_price)  # fmt: skip
    invalid |= bad_price

    report.invalid_rows = int(invalid.sum())
    if len(work) and report.invalid_rows / len(work) > config.MAX_INVALID_ROW_SHARE:
        report.add("error", None, f"{report.invalid_rows / len(work):.1%} of rows are invalid "
                   f"(limit {config.MAX_INVALID_ROW_SHARE:.0%}). Check that columns hold the "
                   "expected values.")  # fmt: skip

    _business_rule_warnings(work, qty, price, report)
    return report


def _business_rule_warnings(
    work: pd.DataFrame, qty: pd.Series, price: pd.Series, report: ValidationReport
) -> None:
    """Append informational findings that are kept and flagged, not dropped."""
    invoice = _strip_text(work[C.COL_INVOICE]).fillna("")
    cancel = invoice.str.startswith(config.CANCELLATION_PREFIX)
    adjust = invoice.str.startswith(config.ADJUSTMENT_PREFIX)
    customers = normalise_customer_ids(work[C.COL_CUSTOMER])
    checks = [
        (cancel, C.COL_INVOICE, "cancellation line(s) (invoice prefix 'C'); flagged and "
         "excluded from gross sales by default."),
        (adjust, C.COL_INVOICE, "accounting adjustment line(s) (invoice prefix 'A'); flagged."),
        (customers.isna(), C.COL_CUSTOMER, "line(s) without a CustomerID (guest sales); kept for "
         "sales totals, excluded from customer analytics."),
        (price == 0, C.COL_PRICE, "line(s) with zero unit price; flagged."),
        (price < 0, C.COL_PRICE, "line(s) with negative unit price; flagged."),
        ((qty < 0) & ~cancel, C.COL_QTY, "negative-quantity line(s) on non-cancellation "
         "invoices (stock write-offs); flagged."),
        (qty == 0, C.COL_QTY, "zero-quantity line(s); flagged."),
        (is_non_product(work[C.COL_STOCK]), C.COL_STOCK, "line(s) with non-product codes "
         "(postage, fees, adjustments); flagged."),
        (work.duplicated(keep="first"), None, "exact duplicate row(s); removed during cleaning."),
    ]  # fmt: skip
    for mask, column, text in checks:
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            report.add("warning", column, f"{int(mask.sum()):,} {text}", mask)
    if C.COL_DESC in work.columns:
        miss = _strip_text(work[C.COL_DESC]).isna()
        if miss.any():
            report.add("warning", C.COL_DESC, f"{int(miss.sum()):,} line(s) without a product "
                       "description; not imputed.", miss)  # fmt: skip


def is_non_product(stock_codes: pd.Series) -> pd.Series:
    """Return True where a stock code is a fee/adjustment rather than a product."""
    codes = _strip_text(stock_codes).fillna("")
    prefixed = np.zeros(len(codes), dtype=bool)
    for prefix in config.NON_PRODUCT_PREFIXES:
        prefixed |= codes.str.startswith(prefix).to_numpy(dtype=bool)
    return pd.Series(codes.isin(config.NON_PRODUCT_STOCK_CODES).to_numpy(dtype=bool) | prefixed,
                     index=stock_codes.index)  # fmt: skip


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


@dataclass
class CleaningLog:
    """Row accounting for :func:`prepare` (every dropped row is counted)."""

    rows_in: int = 0
    dropped_invalid: int = 0
    dropped_duplicates: int = 0
    rows_out: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return the log as a plain dictionary."""
        return asdict(self)


def prepare(df: pd.DataFrame, date_format: str | None) -> tuple[pd.DataFrame, CleaningLog]:
    """Type, clean and flag a *validated* table.

    Steps: alias renames, text stripping, typed parsing, dropping rows with
    unparseable required values, dropping exact duplicate rows, then derived
    columns ``Revenue``, ``IsCancellation``, ``IsAdjustment``, ``HasCustomer``,
    ``IsProduct`` and ``IsReversed``. Nothing is imputed.

    Returns:
        ``(clean_table, cleaning_log)``; the input frame is not modified.
    """
    work, _ = resolve_columns(df)
    work = work[[c for c in config.ALL_COLUMNS if c in work.columns]].copy()
    log = CleaningLog(rows_in=len(work))
    dup = work.duplicated(keep="first")

    out = pd.DataFrame(index=work.index)
    out[C.COL_INVOICE] = _strip_text(work[C.COL_INVOICE])
    out[C.COL_STOCK] = _strip_text(work[C.COL_STOCK])
    if C.COL_DESC in work.columns:
        out[C.COL_DESC] = _strip_text(work[C.COL_DESC])
    qty = pd.to_numeric(work[C.COL_QTY], errors="coerce")
    out[C.COL_DATE] = parse_dates(work[C.COL_DATE], date_format)
    out[C.COL_PRICE] = pd.to_numeric(work[C.COL_PRICE], errors="coerce").astype("float64")
    out[C.COL_CUSTOMER] = normalise_customer_ids(work[C.COL_CUSTOMER])
    if C.COL_COUNTRY in work.columns:
        out[C.COL_COUNTRY] = _strip_text(work[C.COL_COUNTRY])

    valid = (
        out[C.COL_INVOICE].notna() & out[C.COL_STOCK].notna() & out[C.COL_DATE].notna()
        & qty.notna() & (qty == qty.round()) & out[C.COL_PRICE].notna()
    )  # fmt: skip
    log.dropped_invalid = int((~valid).sum())
    keep = valid & ~dup
    log.dropped_duplicates = int((valid & dup).sum())
    out[C.COL_QTY] = qty
    out = out.loc[keep].copy()
    out[C.COL_QTY] = out[C.COL_QTY].astype("int64")

    out[C.COL_REVENUE] = out[C.COL_QTY] * out[C.COL_PRICE]
    out[C.COL_IS_CANCEL] = out[C.COL_INVOICE].str.startswith(config.CANCELLATION_PREFIX)
    out["IsAdjustment"] = out[C.COL_INVOICE].str.startswith(config.ADJUSTMENT_PREFIX)
    out[C.COL_HAS_CUSTOMER] = out[C.COL_CUSTOMER].notna()
    out[C.COL_IS_PRODUCT] = ~is_non_product(out[C.COL_STOCK])
    for col in (C.COL_IS_CANCEL, "IsAdjustment"):
        out[col] = out[col].astype(bool)
    ordered = [c for c in (*config.ALL_COLUMNS, C.COL_REVENUE, C.COL_IS_CANCEL, "IsAdjustment",
                           C.COL_HAS_CUSTOMER, C.COL_IS_PRODUCT) if c in out.columns]  # fmt: skip
    out = out[ordered].sort_values(C.COL_DATE, kind="stable").reset_index(drop=True)
    out[C.COL_IS_REVERSED] = mark_reversed_sales(out)
    log.rows_out = len(out)
    return out, log


def mark_reversed_sales(df: pd.DataFrame) -> pd.Series:
    """Flag sale lines that a later cancellation line exactly reverses.

    A cancellation reverses a sale when both have the same customer, stock code
    and absolute quantity, and the sale is not later than the cancellation.
    Each cancellation is matched to at most one sale (the most recent eligible
    one) and each sale to at most one cancellation. Guest lines (no customer)
    cannot be matched reliably and are never flagged.

    Returns:
        Boolean Series aligned to ``df`` (True = sale later reversed).
    """
    flag = pd.Series(False, index=df.index)
    key = [C.COL_CUSTOMER, C.COL_STOCK, "_absqty"]
    base = df[df[C.COL_HAS_CUSTOMER]].assign(_absqty=lambda d: d[C.COL_QTY].abs())
    cancels = base.loc[base[C.COL_IS_CANCEL] & (base[C.COL_QTY] < 0), [*key, C.COL_DATE]]
    sales = base.loc[~base[C.COL_IS_CANCEL] & (base[C.COL_QTY] > 0), [*key, C.COL_DATE]]
    if cancels.empty or sales.empty:
        return flag
    pairs = cancels.reset_index(names="cancel_idx").merge(
        sales.reset_index(names="sale_idx"), on=key, suffixes=("_c", "_s")
    )
    pairs = pairs[pairs[f"{C.COL_DATE}_s"] <= pairs[f"{C.COL_DATE}_c"]]
    pairs = pairs.sort_values(["cancel_idx", f"{C.COL_DATE}_s"], ascending=[True, False])
    pairs = pairs.drop_duplicates("cancel_idx").drop_duplicates("sale_idx")
    flag.loc[pairs["sale_idx"].to_numpy()] = True
    return flag


def sales_lines(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of genuine product sales lines.

    A sales line is not a cancellation or adjustment, has positive quantity and
    price, is a product (not postage/fees) and was not later reversed by a
    matching cancellation (see :func:`mark_reversed_sales`).
    """
    return (
        ~df[C.COL_IS_CANCEL] & ~df["IsAdjustment"] & df[C.COL_IS_PRODUCT]
        & (df[C.COL_QTY] > 0) & (df[C.COL_PRICE] > 0) & ~df[C.COL_IS_REVERSED]
    )  # fmt: skip


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


@dataclass
class LoadResult:
    """Outcome of loading a dataset.

    Attributes:
        data: Clean table, or ``None`` if validation failed.
        report: Validation report.
        log: Row accounting, or ``None`` if not prepared.
        checksum: SHA-256 of the source file (``None`` for in-memory tables).
    """

    data: pd.DataFrame | None
    report: ValidationReport
    log: CleaningLog | None = None
    checksum: str | None = None

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when data was produced without blocking errors."""
        return self.data is not None and self.report.is_valid


def load_table(df: pd.DataFrame, source: str, checksum: str | None = None) -> LoadResult:
    """Validate and, if valid, prepare an in-memory raw table."""
    report = validate(df, source=source)
    if not report.is_valid:
        return LoadResult(None, report, None, checksum)
    data, log = prepare(df, report.date_format)
    if data.empty:
        report.add("error", None, "No valid rows remain after cleaning.")
        return LoadResult(None, report, log, checksum)
    return LoadResult(data, report, log, checksum)


def load_file(path: Path) -> LoadResult:
    """Read, validate and prepare a CSV/XLSX file from disk."""
    try:
        raw = read_table(path)
    except IngestionError as exc:
        report = ValidationReport(path.name, 0, 0)
        report.add("error", None, str(exc))
        return LoadResult(None, report)
    return load_table(raw, path.name, sha256_file(path))


def load_bundled_dataset(path: Path = config.RAW_FILE) -> LoadResult:
    """Load the bundled UCI Online Retail file, warning on unexpected row counts."""
    result = load_file(path)
    if result.report.n_rows and result.report.n_rows != config.EXPECTED_ROWS:
        result.report.add("warning", None, f"Expected {config.EXPECTED_ROWS:,} rows for "
                          f"{config.DATASET_NAME}, found {result.report.n_rows:,}.")  # fmt: skip
    return result


CACHE_VERSION = 2  # bump when prepare() output changes


def load_bundled_cached(
    path: Path = config.RAW_FILE, cache: Path = config.PROCESSED_CACHE
) -> LoadResult:
    """Load the bundled dataset, reusing a joblib cache keyed by the raw checksum.

    The cache is rebuilt whenever the raw file's SHA-256 or :data:`CACHE_VERSION`
    changes, so cached results always match a fresh :func:`load_bundled_dataset`.
    """
    if not path.exists():
        return load_bundled_dataset(path)
    checksum = sha256_file(path)
    if cache.exists():
        try:
            payload = joblib.load(cache)
            if payload.get("checksum") == checksum and payload.get("version") == CACHE_VERSION:
                return payload["result"]
        except Exception:  # noqa: BLE001 - a corrupt cache is simply rebuilt
            pass
    result = load_bundled_dataset(path)
    if result.is_valid:
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"checksum": checksum, "version": CACHE_VERSION, "result": result}, cache)
    return result


def save_upload(data: bytes, filename: str, uploads_dir: Path = config.UPLOADS_DIR) -> Path:
    """Persist original uploaded bytes unchanged, named by checksum prefix."""
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    target = uploads_dir / f"{sha256_bytes(data)[:16]}_{safe}"
    if not target.exists():
        target.write_bytes(data)
    return target


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_TRANSACTIONS) -> Path:
    """Write the clean table to CSV (ISO dates) and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path


# --------------------------------------------------------------------------- #
# Capabilities and quality summary
# --------------------------------------------------------------------------- #

CAPABILITY_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "Sales KPIs and trends": (C.COL_INVOICE, C.COL_DATE, C.COL_QTY, C.COL_PRICE),
    "Product analysis": (C.COL_STOCK, C.COL_DESC),
    "Country analysis": (C.COL_COUNTRY,),
    "RFM and K-Means segmentation": (C.COL_CUSTOMER, C.COL_INVOICE, C.COL_DATE),
    "Churn early-warning model": (C.COL_CUSTOMER, C.COL_INVOICE, C.COL_DATE),
}


def assess_capabilities(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map each feature to the reasons it is unavailable (empty list = available).

    Considers both missing columns and data sufficiency (e.g. no customer IDs,
    or too short a history for the churn horizon). Columns are never substituted.
    """
    out: dict[str, list[str]] = {}
    for feature, needed in CAPABILITY_REQUIREMENTS.items():
        reasons = [f"missing column {c}" for c in needed if c not in df.columns]
        out[feature] = reasons
    if C.COL_CUSTOMER in df.columns and not df[C.COL_CUSTOMER].notna().any():
        for feature in ("RFM and K-Means segmentation", "Churn early-warning model"):
            out[feature].append("no customer IDs present")
    if C.COL_DATE in df.columns and df[C.COL_DATE].notna().any():
        span = (df[C.COL_DATE].max() - df[C.COL_DATE].min()).days
        if span < 2 * config.CHURN_HORIZON_DAYS:
            out["Churn early-warning model"].append(
                f"history spans {span} days; needs at least {2 * config.CHURN_HORIZON_DAYS}"
            )
    return out


def data_quality_summary(df: pd.DataFrame, log: CleaningLog | None = None) -> dict[str, Any]:
    """Compute data-quality figures for a clean table (all values from ``df``)."""
    sales = sales_lines(df)
    s: dict[str, Any] = {
        "rows": int(len(df)),
        "date_min": str(df[C.COL_DATE].min()),
        "date_max": str(df[C.COL_DATE].max()),
        "invoices": int(df[C.COL_INVOICE].nunique()),
        "customers": int(df[C.COL_CUSTOMER].nunique()),
        "products": int(df.loc[df[C.COL_IS_PRODUCT], C.COL_STOCK].nunique()),
        "countries": int(df[C.COL_COUNTRY].nunique()) if C.COL_COUNTRY in df.columns else None,
        "cancellation_lines": int(df[C.COL_IS_CANCEL].sum()),
        "cancelled_invoices": int(df.loc[df[C.COL_IS_CANCEL], C.COL_INVOICE].nunique()),
        "adjustment_lines": int(df["IsAdjustment"].sum()),
        "lines_without_customer": int((~df[C.COL_HAS_CUSTOMER]).sum()),
        "share_without_customer": float((~df[C.COL_HAS_CUSTOMER]).mean()),
        "zero_price_lines": int((df[C.COL_PRICE] == 0).sum()),
        "negative_price_lines": int((df[C.COL_PRICE] < 0).sum()),
        "negative_qty_non_cancel_lines": int(((df[C.COL_QTY] < 0) & ~df[C.COL_IS_CANCEL]).sum()),
        "non_product_lines": int((~df[C.COL_IS_PRODUCT]).sum()),
        "reversed_sale_lines": int(df[C.COL_IS_REVERSED].sum()),
        "reversed_sale_value": float(df.loc[df[C.COL_IS_REVERSED], C.COL_REVENUE].sum()),
        "sales_lines": int(sales.sum()),
        "gross_product_revenue": float(df.loc[sales, C.COL_REVENUE].sum()),
        "cancelled_revenue": float(df.loc[df[C.COL_IS_CANCEL], C.COL_REVENUE].sum()),
        "net_revenue_all_lines": float(df[C.COL_REVENUE].sum()),
        "customer_share_of_gross_revenue": float(
            df.loc[sales & df[C.COL_HAS_CUSTOMER], C.COL_REVENUE].sum()
            / max(df.loc[sales, C.COL_REVENUE].sum(), 1e-9)
        ),
    }
    if log is not None:
        s["cleaning"] = log.to_dict()
    return s


def format_quality_summary(s: dict[str, Any]) -> str:
    """Render :func:`data_quality_summary` output as plain text."""
    cur = config.CURRENCY_SYMBOL
    lines = ["Data-quality summary", "=" * 20]
    if "cleaning" in s:
        c = s["cleaning"]
        lines.append(f"Rows read: {c['rows_in']:,} | invalid dropped: {c['dropped_invalid']:,} | "
                     f"exact duplicates dropped: {c['dropped_duplicates']:,} | "
                     f"rows kept: {c['rows_out']:,}")  # fmt: skip
    lines += [
        f"Period: {s['date_min']} to {s['date_max']}",
        f"Invoices: {s['invoices']:,} | customers: {s['customers']:,} | products: "
        f"{s['products']:,} | countries: {s['countries']}",
        f"Cancellation lines: {s['cancellation_lines']:,} "
        f"({s['cancelled_invoices']:,} invoices, {cur}{s['cancelled_revenue']:,.2f})",
        f"Adjustment lines: {s['adjustment_lines']:,}",
        f"Lines without CustomerID: {s['lines_without_customer']:,} "
        f"({s['share_without_customer']:.1%})",
        f"Zero-price lines: {s['zero_price_lines']:,} | negative-price lines: "
        f"{s['negative_price_lines']:,}",
        f"Negative-quantity lines on normal invoices: {s['negative_qty_non_cancel_lines']:,}",
        f"Non-product lines (postage, fees, ...): {s['non_product_lines']:,}",
        f"Sales later reversed by a matching cancellation: {s['reversed_sale_lines']:,} lines "
        f"({cur}{s['reversed_sale_value']:,.2f}); excluded from gross sales",
        f"Product sales lines: {s['sales_lines']:,} | gross product revenue: "
        f"{cur}{s['gross_product_revenue']:,.2f} "
        f"({s['customer_share_of_gross_revenue']:.1%} from identified customers)",
        "Net revenue, all lines incl. cancellations & fees: "
        f"{cur}{s['net_revenue_all_lines']:,.2f}",
    ]
    return "\n".join(lines)
