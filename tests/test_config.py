import re

from src import config


def test_schema_columns_unique_and_required_subset():
    names = [c.name for c in config.SCHEMA]
    assert len(names) == len(set(names)) == 8
    assert set(config.REQUIRED_COLUMNS) <= set(names)
    assert config.COL_CUSTOMER in config.REQUIRED_COLUMNS


def test_aliases_do_not_collide():
    seen = set(config.ALL_COLUMNS)
    for spec in config.SCHEMA:
        for alias in spec.aliases:
            assert alias not in seen, alias
            seen.add(alias)


def test_rfm_rules_cover_every_score_pair():
    for r in range(1, 6):
        for f in range(1, 6):
            assert any(re.fullmatch(p, f"{r}{f}") for p, _ in config.RFM_SEGMENT_RULES), (r, f)


def test_modelling_constants():
    assert config.RANDOM_STATE == 42 and config.CV_FOLDS == 5 and config.CV_SHUFFLE
    assert config.DT_MAX_DEPTH == 5 and config.RF_N_ESTIMATORS == 300
    assert 0 < config.RISK_MEDIUM < config.RISK_HIGH < 1
    lo, hi = config.KMEANS_K_RANGE
    assert 2 <= lo < hi


def test_paths_are_relative_to_repo():
    assert config.DATA_DIR.parent == config.ROOT_DIR
    assert (config.ROOT_DIR / "src" / "config.py").exists()
