"""Download the UCI Online Retail dataset into ``data/raw/``.

Usage::

    python scripts/fetch_data.py                  # UCI first, then the public CSV copy
    python scripts/fetch_data.py --no-mirror      # UCI only
    python scripts/fetch_data.py --source PATH    # manually downloaded .zip / .xlsx / .csv
    python scripts/fetch_data.py --force          # re-acquire even if the file exists

The original download is kept unchanged in ``data/raw/original/``; a canonical
CSV is written to ``data/raw/online_retail.csv``. The script then validates and
cleans the data, writes ``data/processed/transactions_clean.csv`` and prints a
data-quality summary. A joblib cache of the clean table is written alongside
for fast dashboard start-up.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingestion  # noqa: E402

TIMEOUT_S = 120
ORIGINAL_DIR = config.RAW_DIR / "original"


def http_get(url: str) -> bytes:
    """Download ``url`` and return the body.

    Raises:
        urllib.error.URLError: On network failure.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "sales-analytics/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
        return response.read()


def extract_from_zip(data: bytes) -> tuple[str, bytes]:
    """Return ``(name, bytes)`` of the first .xlsx/.csv member (nested zips searched).

    Only base names are used and nothing is written here, so archive paths
    cannot escape the target directory.

    Raises:
        zipfile.BadZipFile: If ``data`` is not a zip archive.
        FileNotFoundError: If no data file is found.
    """
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = sorted(archive.infolist(), key=lambda i: Path(i.filename).name
                         != config.DATASET_ZIP_MEMBER)  # fmt: skip
        for info in members:
            name = Path(info.filename).name
            if name.lower().endswith((".xlsx", ".csv")):
                return name, archive.read(info)
            if name.lower().endswith(".zip"):
                try:
                    return extract_from_zip(archive.read(info))
                except FileNotFoundError:
                    continue
    raise FileNotFoundError("No .xlsx or .csv file found in the archive.")


def acquire_from_source(source: Path) -> tuple[str, bytes]:
    """Return ``(name, bytes)`` from a local .zip/.xlsx/.csv file."""
    data = source.read_bytes()
    if source.suffix.lower() == ".zip":
        return extract_from_zip(data)
    return source.name, data


def write_raw(name: str, data: bytes, raw_file: Path = config.RAW_FILE) -> Path:
    """Keep the original bytes and write the canonical CSV used by the app.

    CSV input is copied byte-for-byte; XLSX is converted with all identifier
    columns kept as text (no values are altered).
    """
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    original = ORIGINAL_DIR / Path(name).name
    original.write_bytes(data)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    if name.lower().endswith(".csv"):
        raw_file.write_bytes(data)
    else:
        ingestion.read_table_bytes(data, name).to_csv(raw_file, index=False)
    return original


def manual_instructions() -> str:
    """Return step-by-step manual download instructions."""
    return (
        "\nAutomatic download failed. To install the dataset manually:\n"
        f"  1. Open {config.DATASET_PAGE_URL} and click 'Download' (online+retail.zip),\n"
        f"     or download the CSV copy: {config.DATASET_MIRROR_URL}\n"
        "  2. Run:\n"
        "       python scripts/fetch_data.py --source path/to/online+retail.zip\n"
        "     (an extracted 'Online Retail.xlsx' or the CSV also works), or copy a CSV to:\n"
        f"       {config.RAW_FILE}\n"
    )


def acquire(use_mirror: bool) -> tuple[str, bytes, str]:
    """Download from UCI, falling back to the public CSV copy if allowed.

    Returns:
        ``(file_name, bytes, origin_url)``.

    Raises:
        RuntimeError: When every source failed (message lists each failure).
    """
    failures = []
    try:
        print(f"Downloading {config.DATASET_ZIP_URL} ...")
        name, data = extract_from_zip(http_get(config.DATASET_ZIP_URL))
        return name, data, config.DATASET_ZIP_URL
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
        failures.append(f"UCI: {exc}")
        print(f"  UCI download failed: {exc}")
    if use_mirror:
        try:
            print(f"Falling back to the public CSV copy:\n  {config.DATASET_MIRROR_URL}")
            return (
                "online-retail-dataset.csv",
                http_get(config.DATASET_MIRROR_URL),
                (config.DATASET_MIRROR_URL),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append(f"mirror: {exc}")
            print(f"  Mirror download failed: {exc}")
    raise RuntimeError("; ".join(failures))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--source", type=Path, help="Local .zip, .xlsx or .csv file")
    parser.add_argument("--force", action="store_true", help="Re-acquire even if present")
    parser.add_argument("--no-mirror", action="store_true", help="Do not use the CSV copy")
    args = parser.parse_args(argv)
    config.ensure_directories()

    if config.RAW_FILE.exists() and not args.force and args.source is None:
        print(f"Raw file already present: {config.RAW_FILE} (use --force to re-acquire).")
    else:
        try:
            if args.source is not None:
                print(f"Reading {args.source} ...")
                name, data = acquire_from_source(args.source)
                origin = str(args.source)
            else:
                name, data, origin = acquire(use_mirror=not args.no_mirror)
        except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(manual_instructions(), file=sys.stderr)
            return 1
        try:
            original = write_raw(name, data)
        except ingestion.IngestionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"  origin:   {origin}\n  original: {original}  sha256={ingestion.sha256_bytes(data)}"
              f"\n  raw CSV:  {config.RAW_FILE}")  # fmt: skip

    print("\nValidating and cleaning ...")
    config.PROCESSED_CACHE.unlink(missing_ok=True)  # always rebuild from the raw file
    result = ingestion.load_bundled_cached()
    report = result.report
    print(f"Validation [{'OK' if report.is_valid else 'FAILED'}] {report.source}: "
          f"{report.n_rows:,} rows, {report.n_columns} columns, date format "
          f"{report.date_format!r}, sha256={result.checksum}")  # fmt: skip
    for issue in report.issues:
        print(f"  - {issue.level.upper()}: {issue.message}")
    if not result.is_valid or result.data is None:
        print("\nDataset failed validation; processed file not written.", file=sys.stderr)
        return 2
    out = ingestion.save_processed(result.data)
    print(f"\nProcessed dataset written to {out}\n")
    print(ingestion.format_quality_summary(ingestion.data_quality_summary(result.data, result.log)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
