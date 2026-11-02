from src import config


def test_schema_has_33_attributes():
    assert len(config.SCHEMA) == 33
    assert len(set(config.REQUIRED_COLUMNS)) == 33


def test_grade_ranges():
    for g in ("G1", "G2", "G3"):
        spec = config.SCHEMA_BY_NAME[g]
        assert (spec.min_value, spec.max_value) == (0, 20)


def test_absence_bands_align():
    assert len(config.ABSENCE_BINS) == len(config.ABSENCE_LABELS) + 1


def test_paths_are_relative_to_repo():
    assert config.DATA_DIR.parent == config.ROOT_DIR
    assert (config.ROOT_DIR / "src" / "config.py").exists()
