from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_transactions(n_invoices: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw table in the UCI Online Retail CSV layout (random, not real data)."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2010-12-01 08:00")
    for i in range(n_invoices):
        cust = None if i % 7 == 0 else f"{12000 + i % 15}.0"
        ts = start + pd.Timedelta(days=int(rng.integers(0, 365)), minutes=int(rng.integers(0, 600)))
        for _ in range(int(rng.integers(1, 4))):
            rows.append({
                "InvoiceNo": str(536000 + i),
                "StockCode": str(rng.choice(["85123A", "71053", "22423", "84029G"])),
                "Description": "ITEM",
                "Quantity": int(rng.integers(1, 12)),
                "InvoiceDate": f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}",
                "UnitPrice": float(rng.choice([1.25, 2.55, 3.39, 12.75])),
                "CustomerID": cust,
                "Country": str(rng.choice(["United Kingdom", "France", "Germany"])),
            })  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture
def clean():
    from src import ingestion

    return ingestion.load_table(make_transactions(120, seed=3), "synthetic").data
