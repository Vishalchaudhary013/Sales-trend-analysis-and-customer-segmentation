from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


def make_students(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic table satisfying the UCI schema (values are random, not real data)."""
    rng = np.random.default_rng(seed)
    cols: dict[str, object] = {}
    for spec in config.SCHEMA:
        if spec.kind == "numeric":
            cols[spec.name] = rng.integers(int(spec.min_value), int(spec.max_value) + 1, n)
        else:
            cols[spec.name] = rng.choice(spec.allowed, n)
    return pd.DataFrame(cols)


@pytest.fixture
def students() -> pd.DataFrame:
    return make_students()
