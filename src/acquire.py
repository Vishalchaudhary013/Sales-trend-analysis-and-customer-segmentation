"""Dataset and exchange-rate acquisition (download, extract, store).

Shared by ``scripts/fetch_data.py`` and the dashboard, which calls
:func:`bootstrap` on first start when the data files are missing (e.g. on a
fresh Streamlit Community Cloud deployment, where ``data/`` is not in git).
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from src import config, ingestion

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
        members = sorted(
            archive.infolist(), key=lambda i: Path(i.filename).name != config.DATASET_ZIP_MEMBER
        )
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


def write_raw(name: str, data: bytes, raw_file: Path | None = None) -> Path:
    """Keep the original bytes and write the canonical CSV used by the app.

    CSV input is copied byte-for-byte; XLSX is converted with all identifier
    columns kept as text (no values are altered).
    """
    raw_file = raw_file or config.RAW_FILE  # resolved at call time (testable, patchable)
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


def acquire(use_mirror: bool, log: Callable[[str], None] = print) -> tuple[str, bytes, str]:
    """Download from UCI, falling back to the public CSV copy if allowed.

    Returns:
        ``(file_name, bytes, origin_url)``.

    Raises:
        RuntimeError: When every source failed (message lists each failure).
    """
    failures = []
    try:
        log(f"Downloading {config.DATASET_ZIP_URL} ...")
        name, data = extract_from_zip(http_get(config.DATASET_ZIP_URL))
        return name, data, config.DATASET_ZIP_URL
    except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile) as exc:
        failures.append(f"UCI: {exc}")
        log(f"  UCI download failed: {exc}")
    if use_mirror:
        try:
            log(f"Falling back to the public CSV copy:\n  {config.DATASET_MIRROR_URL}")
            return (
                "online-retail-dataset.csv",
                http_get(config.DATASET_MIRROR_URL),
                (config.DATASET_MIRROR_URL),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            failures.append(f"mirror: {exc}")
            log(f"  Mirror download failed: {exc}")
    raise RuntimeError("; ".join(failures))


def download_rates(target: Path | None = None) -> None:
    """Download the daily exchange-rate file to ``target``.

    Raises:
        urllib.error.URLError, TimeoutError, OSError: On network/file failure.
    """
    target = target or config.FX_FILE
    data = http_get(config.FX_URL)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def bootstrap(use_mirror: bool = True, log: Callable[[str], None] = print) -> dict[str, bool]:
    """Fetch whatever data files are missing (dataset and exchange rates).

    Never raises for network problems; returns which files are present
    afterwards so callers can explain what is unavailable.
    """
    config.ensure_directories()
    if not config.RAW_FILE.exists():
        try:
            name, data, origin = acquire(use_mirror, log)
            write_raw(name, data)
            log(f"Dataset saved from {origin}")
        except (RuntimeError, OSError, zipfile.BadZipFile, ingestion.IngestionError) as exc:
            log(f"Dataset download failed: {exc}")
    if not config.FX_FILE.exists():
        try:
            download_rates()
            log("Exchange rates saved")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log(f"Exchange-rate download failed: {exc}")
    return {"dataset": config.RAW_FILE.exists(), "rates": config.FX_FILE.exists()}
