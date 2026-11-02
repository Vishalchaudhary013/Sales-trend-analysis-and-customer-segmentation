"""Download the UCI Student Performance dataset into ``data/raw/``.

Usage::

    python scripts/fetch_data.py                 # download from UCI
    python scripts/fetch_data.py --source PATH   # use a manually downloaded .zip or folder
    python scripts/fetch_data.py --force         # re-download even if files exist

After the files are in place the script validates them, writes the cleaned,
merged dataset to ``data/processed/`` and prints a data-quality summary.
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

TIMEOUT_S = 60


def _extract_members(archive: zipfile.ZipFile, wanted: set[str]) -> dict[str, bytes]:
    """Return the bytes of ``wanted`` file names, searching nested zips too.

    Only the base name of each member is matched and nothing is written to disk
    here, so archive paths cannot escape the target directory.
    """
    found: dict[str, bytes] = {}
    for info in archive.infolist():
        name = Path(info.filename).name
        if name in wanted and name not in found:
            found[name] = archive.read(info)
        elif name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(info))) as inner:
                for k, v in _extract_members(inner, wanted - set(found)).items():
                    found.setdefault(k, v)
    return found


def download_zip(url: str = config.DATASET_ZIP_URL) -> bytes:
    """Download the dataset archive and return its bytes.

    Raises:
        urllib.error.URLError: On network failure.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "education-analytics/1.0"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310
        return response.read()


def files_from_source(source: Path, wanted: set[str]) -> dict[str, bytes]:
    """Collect the dataset CSVs from a local ``.zip`` or a folder."""
    if source.is_dir():
        return {n: (source / n).read_bytes() for n in wanted if (source / n).exists()}
    with zipfile.ZipFile(source) as archive:
        return _extract_members(archive, wanted)


def manual_instructions(target: Path) -> str:
    """Return step-by-step manual download instructions."""
    files = " and ".join(config.RAW_FILES.values())
    return (
        "\nAutomatic download failed. To install the dataset manually:\n"
        f"  1. Open {config.DATASET_PAGE_URL}\n"
        "  2. Click 'Download' to get 'student+performance.zip'.\n"
        "  3. Either run:\n"
        "       python scripts/fetch_data.py --source path/to/student+performance.zip\n"
        f"     or extract the inner 'student.zip' and copy {files} into:\n"
        f"       {target}\n"
        "  4. Re-run: python scripts/fetch_data.py\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="Local .zip or folder with the CSVs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw files")
    args = parser.parse_args(argv)

    config.ensure_directories()
    wanted = set(config.RAW_FILES.values())
    target = config.RAW_DIR
    existing = {n for n in wanted if (target / n).exists()}

    if existing == wanted and not args.force and args.source is None:
        print(f"Raw files already present in {target} (use --force to re-download).")
    else:
        try:
            if args.source is not None:
                print(f"Reading dataset from {args.source} ...")
                files = files_from_source(args.source, wanted)
            else:
                print(f"Downloading {config.DATASET_ZIP_URL} ...")
                with zipfile.ZipFile(io.BytesIO(download_zip())) as archive:
                    files = _extract_members(archive, wanted)
        except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(manual_instructions(target), file=sys.stderr)
            return 1
        missing = wanted - set(files)
        if missing:
            print(f"ERROR: archive is missing {sorted(missing)}", file=sys.stderr)
            print(manual_instructions(target), file=sys.stderr)
            return 1
        for name, data in files.items():
            (target / name).write_bytes(data)
            print(f"  saved {target / name}  sha256={ingestion.sha256_bytes(data)}")

    result = ingestion.load_uci_dataset(target)
    for report in result.reports:
        status = "OK" if report.is_valid else "FAILED"
        print(f"\nValidation [{status}] {report.source}: {report.n_rows} rows, "
              f"{report.n_columns} columns, mode={report.mode}")  # fmt: skip
        for issue in report.issues:
            print(f"  - {issue.level.upper()}: {issue.message}")
    if not result.is_valid or result.data is None:
        print("\nDataset failed validation; processed file not written.", file=sys.stderr)
        return 2

    out = ingestion.save_processed(result.data)
    print(f"\nProcessed dataset written to {out}\n")
    print(ingestion.format_quality_summary(ingestion.data_quality_summary(result.data)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
