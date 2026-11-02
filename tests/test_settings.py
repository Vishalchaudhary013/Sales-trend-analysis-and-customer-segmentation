from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import config, settings

ROOT = Path(config.ROOT_DIR)


def test_parse_dotenv_formats():
    text = (
        "# comment\n\nA=1\nexport B = two \nC='quoted # not a comment'\n"
        'D="x y"\nE=val # trailing comment\nA=override\n'
    )
    assert settings.parse_dotenv(text) == {
        "A": "override",
        "B": "two",
        "C": "quoted # not a comment",
        "D": "x y",
        "E": "val",
    }
    with pytest.raises(settings.SettingsError, match="line 2"):
        settings.parse_dotenv("A=1\nnot a setting\n")


def test_load_dotenv_respects_existing_and_ignore_flag(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ZZ_NEW=from_file\nZZ_SET=from_file\n")
    monkeypatch.delenv(settings.IGNORE_FLAG, raising=False)
    monkeypatch.setenv("ZZ_SET", "from_shell")
    monkeypatch.delenv("ZZ_NEW", raising=False)
    applied = settings.load_dotenv(env)
    assert applied == {"ZZ_NEW": "from_file"}
    assert os.environ["ZZ_SET"] == "from_shell"  # real environment wins
    monkeypatch.delenv("ZZ_NEW")
    monkeypatch.setenv(settings.IGNORE_FLAG, "1")
    assert settings.load_dotenv(env) == {} and "ZZ_NEW" not in os.environ
    assert settings.load_dotenv(tmp_path / "missing.env") == {}


def test_typed_readers(monkeypatch, tmp_path):
    monkeypatch.setenv("ZZ_INT", "60")
    monkeypatch.setenv("ZZ_FLOAT", "0.65")
    monkeypatch.setenv("ZZ_PATH", "sub/dir")
    monkeypatch.setenv("ZZ_CHOICE", "inr")
    monkeypatch.setenv("ZZ_EMPTY", "")
    assert settings.env_int("ZZ_INT", 90, 30, 180) == 60
    assert settings.env_int("ZZ_EMPTY", 90) == 90  # empty = default
    assert settings.env_float("ZZ_FLOAT", 0.5, 0.0, 1.0) == 0.65
    assert settings.env_path("ZZ_PATH", tmp_path, tmp_path) == tmp_path / "sub" / "dir"
    assert settings.env_path("ZZ_UNSET_PATH", tmp_path / "d", tmp_path) == tmp_path / "d"
    assert settings.env_choice("ZZ_CHOICE", "GBP", ("GBP", "INR")) == "INR"


@pytest.mark.parametrize(
    ("reader", "value", "message"),
    [
        (lambda: settings.env_int("ZZ_BAD", 1, 30, 180), "ten", "whole number"),
        (lambda: settings.env_int("ZZ_BAD", 1, 30, 180), "500", "between 30 and 180"),
        (lambda: settings.env_float("ZZ_BAD", 0.5, 0.0, 1.0), "high", "must be a number"),
        (lambda: settings.env_float("ZZ_BAD", 0.5, 0.0, 1.0), "1.0", "between 0.0 and 1.0"),
        (lambda: settings.env_choice("ZZ_BAD", "GBP", ("GBP",)), "XYZ", "must be one of"),
    ],
)
def test_invalid_values_name_the_variable(monkeypatch, reader, value, message):
    monkeypatch.setenv("ZZ_BAD", value)
    with pytest.raises(settings.SettingsError, match=message) as err:
        reader()
    assert "ZZ_BAD" in str(err.value)


def _config_in_subprocess(extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != settings.IGNORE_FLAG}
    env.update(extra_env)
    code = (
        "from src import config as c; print(c.CHURN_HORIZON_DAYS, c.RISK_HIGH, "
        "c.DEFAULT_DISPLAY_CURRENCY, c.DB_PATH.name, c.MODEL_FILE.parent.name)"
    )
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=60
    )


def test_config_picks_up_overrides():
    out = _config_in_subprocess(
        {
            "CHURN_HORIZON_DAYS": "60",
            "RISK_HIGH": "0.8",
            "DEFAULT_DISPLAY_CURRENCY": "eur",
            "DB_PATH": "x/other.db",
            "MODELS_DIR": "alt_models",
        }
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["60", "0.8", "EUR", "other.db", "alt_models"]


def test_invalid_settings_fall_back_with_a_reported_error():
    out = _config_in_subprocess(
        {"RISK_HIGH": "0.3", "RISK_MEDIUM": "0.4", "CHURN_HORIZON_DAYS": "ten"}
    )
    assert out.returncode == 0, out.stderr
    horizon, high = out.stdout.split()[:2]
    assert (horizon, high) == ("90", "0.7")  # defaults used
    assert "RISK_MEDIUM (0.4) must be lower than RISK_HIGH (0.3)" in out.stderr
    assert "CHURN_HORIZON_DAYS must be a whole number" in out.stderr


def test_safe_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ERRORS", [])
    monkeypatch.setenv("ZZ_SAFE", "abc")
    with pytest.warns(UserWarning, match="ZZ_SAFE"):
        assert settings.safe(settings.env_int, "ZZ_SAFE", 7) == 7
    bad = tmp_path / ".env"
    bad.write_text("this is not valid\n")
    monkeypatch.delenv(settings.IGNORE_FLAG, raising=False)
    with pytest.warns(UserWarning, match="not applied"):
        assert settings.safe_load_dotenv(bad) == {}
    assert len(settings.ERRORS) == 2


def test_env_example_lists_every_supported_setting():
    text = (ROOT / ".env.example").read_text()
    for name in (
        "DATA_DIR",
        "MODELS_DIR",
        "REPORTS_DIR",
        "DB_PATH",
        "DEFAULT_DISPLAY_CURRENCY",
        "LARGE_LINE_QUANTITY",
        "EXPORT_DPI",
        "CHURN_HORIZON_DAYS",
        "DEFAULT_THRESHOLD",
        "RISK_HIGH",
        "RISK_MEDIUM",
        "TOP_N_REASONS",
        "RANDOM_STATE",
    ):
        assert f"# {name}=" in text, name
    parsed = settings.parse_dotenv(text)
    assert parsed == {}  # everything commented out by default
