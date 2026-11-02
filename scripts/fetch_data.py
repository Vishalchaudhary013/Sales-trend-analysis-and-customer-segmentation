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
import sys
import urllib.error
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import acquire as acquire_mod  # noqa: E402
from src import config, currency, ingestion  # noqa: E402
from src.acquire import (  # noqa: E402,F401 - re-exported for backwards compatibility
    ORIGINAL_DIR,
    acquire,
    acquire_from_source,
    extract_from_zip,
    http_get,
    manual_instructions,
    write_raw,
)


def fetch_rates(force: bool = False, target: Path | None = None) -> bool:
    """Download the daily exchange-rate file (used for currency conversion).

    Failure is not fatal: the dashboard then offers the base currency and a
    user-supplied custom rate only.

    Returns:
        True if a valid rate file is present afterwards.
    """
    target = target or config.FX_FILE
    if target.exists() and not force:
        print(f"Exchange rates already present: {target}")
    else:
        try:
            print(f"Downloading exchange rates:\n  {config.FX_URL}")
            acquire_mod.download_rates(target)
            print(f"  saved {target}  sha256={ingestion.sha256_file(target)}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(
                f"WARNING: exchange rates unavailable ({exc}). Currency conversion will be "
                "limited to user-supplied rates.",
                file=sys.stderr,
            )
            return False
    try:
        rates = currency.load_rates(target)
    except (currency.CurrencyError, ValueError, KeyError) as exc:
        print(f"WARNING: exchange-rate file invalid ({exc}).", file=sys.stderr)
        return False
    codes = currency.available_currencies(rates)
    print(
        f"  {len(codes)} currencies, {rates.index.min():%Y-%m-%d} to "
        f"{rates.index.max():%Y-%m-%d}: {', '.join(codes)}"
    )
    return True


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
        print(
            f"  origin:   {origin}\n  original: {original}  sha256={ingestion.sha256_bytes(data)}"
            f"\n  raw CSV:  {config.RAW_FILE}"
        )

    print()
    fetch_rates(force=args.force)
    print("\nValidating and cleaning ...")
    config.PROCESSED_CACHE.unlink(missing_ok=True)  # always rebuild from the raw file
    result = ingestion.load_bundled_cached()
    report = result.report
    print(
        f"Validation [{'OK' if report.is_valid else 'FAILED'}] {report.source}: "
        f"{report.n_rows:,} rows, {report.n_columns} columns, date format "
        f"{report.date_format!r}, sha256={result.checksum}"
    )
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
