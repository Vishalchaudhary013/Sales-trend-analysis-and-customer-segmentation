"""SQLite repository: upload history, saved models and prediction runs.

Uses only the standard-library :mod:`sqlite3`. Every public function opens its
own short-lived connection, so it is safe to call from Streamlit's threads.
Only metadata is stored (file names, counts, checksums, settings); no
transaction rows or customer details are written to the database.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uploaded_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    stored_path TEXT,
    row_count INTEGER NOT NULL,
    valid INTEGER NOT NULL,
    mode TEXT,
    sha256 TEXT NOT NULL,
    n_errors INTEGER NOT NULL DEFAULT 0,
    n_warnings INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    saved_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    setting TEXT NOT NULL,
    path TEXT NOT NULL,
    roc_auc REAL,
    metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    setting TEXT NOT NULL,
    threshold REAL NOT NULL,
    dataset_sha256 TEXT,
    n_scored INTEGER NOT NULL,
    n_flagged INTEGER NOT NULL,
    n_high INTEGER NOT NULL,
    n_medium INTEGER NOT NULL,
    n_low INTEGER NOT NULL,
    currency TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: UP017 - py3.10


@contextmanager
def connect(db_path: Path = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a connection with the schema ensured; commits on success."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_upload(
    filename: str,
    row_count: int,
    sha256: str,
    valid: bool,
    mode: str | None,
    n_errors: int = 0,
    n_warnings: int = 0,
    stored_path: str | None = None,
    db_path: Path = config.DB_PATH,
) -> int:
    """Register an uploaded file and return its id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO uploads (uploaded_at, filename, stored_path, row_count, valid, mode, "
            "sha256, n_errors, n_warnings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                filename,
                stored_path,
                int(row_count),
                int(valid),
                mode,
                sha256,
                int(n_errors),
                int(n_warnings),
            ),
        )
        return int(cur.lastrowid)


def record_model(
    model_name: str,
    setting: str,
    path: str,
    metadata: dict[str, Any],
    roc_auc: float | None = None,
    db_path: Path = config.DB_PATH,
) -> int:
    """Register a saved model (metadata stored as JSON) and return its id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO models (saved_at, model_name, setting, path, roc_auc, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), model_name, setting, path, roc_auc, json.dumps(metadata, default=str)),
        )
        return int(cur.lastrowid)


def record_prediction_run(
    model_name: str,
    setting: str,
    threshold: float,
    n_scored: int,
    n_flagged: int,
    tiers: dict[str, int],
    dataset_sha256: str | None,
    currency: str | None = None,
    db_path: Path = config.DB_PATH,
) -> int:
    """Log one early-warning scoring run and return its id."""
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO prediction_runs (run_at, model_name, setting, threshold, dataset_sha256, "
            "n_scored, n_flagged, n_high, n_medium, n_low, currency) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                model_name,
                setting,
                float(threshold),
                dataset_sha256,
                int(n_scored),
                int(n_flagged),
                int(tiers.get("High", 0)),
                int(tiers.get("Medium", 0)),
                int(tiers.get("Low", 0)),
                currency,
            ),
        )
        return int(cur.lastrowid)


def _read(table: str, order: str, limit: int, db_path: Path) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {table} ORDER BY {order} DESC LIMIT ?",  # noqa: S608
            conn,
            params=(int(limit),),
        )


def upload_history(limit: int = 50, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Return the most recent uploads (newest first)."""
    return _read("uploads", "id", limit, db_path)


def model_history(limit: int = 50, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Return the most recently saved models (newest first)."""
    return _read("models", "id", limit, db_path)


def prediction_history(limit: int = 50, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    """Return the most recent prediction runs (newest first)."""
    return _read("prediction_runs", "id", limit, db_path)
