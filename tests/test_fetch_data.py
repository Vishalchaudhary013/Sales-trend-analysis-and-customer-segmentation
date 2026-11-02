from __future__ import annotations

import io
import urllib.error
import zipfile

import pytest

from scripts import fetch_data


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_extract_prefers_named_member_and_nested():
    data = _zip({"readme.txt": b"x", "other.csv": b"o", "Online Retail.xlsx": b"x1"})
    assert fetch_data.extract_from_zip(data) == ("Online Retail.xlsx", b"x1")
    nested = _zip({"inner.zip": _zip({"a.csv": b"a"})})
    assert fetch_data.extract_from_zip(nested) == ("a.csv", b"a")
    with pytest.raises(FileNotFoundError):
        fetch_data.extract_from_zip(_zip({"readme.txt": b"x"}))


def test_acquire_from_source(tmp_path):
    z = tmp_path / "d.zip"
    z.write_bytes(_zip({"x.csv": b"1"}))
    assert fetch_data.acquire_from_source(z) == ("x.csv", b"1")
    c = tmp_path / "y.csv"
    c.write_bytes(b"2")
    assert fetch_data.acquire_from_source(c) == ("y.csv", b"2")


def test_write_raw_keeps_original(tmp_path, monkeypatch, raw):
    monkeypatch.setattr(fetch_data, "ORIGINAL_DIR", tmp_path / "orig")
    data = raw.to_csv(index=False).encode()
    original = fetch_data.write_raw("src.csv", data, tmp_path / "raw.csv")
    assert original.read_bytes() == data and (tmp_path / "raw.csv").read_bytes() == data
    buf = io.BytesIO()
    raw.to_excel(buf, index=False)
    fetch_data.write_raw("src.xlsx", buf.getvalue(), tmp_path / "raw2.csv")
    assert (tmp_path / "raw2.csv").read_text().startswith("InvoiceNo,")


def test_acquire_falls_back_to_mirror(monkeypatch):
    def fake_get(url):
        if "uci.edu" in url:
            raise urllib.error.URLError("blocked")
        return b"csv-bytes"

    monkeypatch.setattr(fetch_data, "http_get", fake_get)
    name, data, origin = fetch_data.acquire(use_mirror=True)
    assert data == b"csv-bytes" and "githubusercontent" in origin
    with pytest.raises(RuntimeError, match="UCI"):
        fetch_data.acquire(use_mirror=False)


def test_main_reports_manual_instructions(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(fetch_data.config, "RAW_FILE", tmp_path / "none.csv")
    monkeypatch.setattr(fetch_data, "ORIGINAL_DIR", tmp_path / "orig")
    monkeypatch.setattr(fetch_data.config, "ensure_directories", lambda: None)

    def fail(url):
        raise urllib.error.URLError("blocked")

    monkeypatch.setattr(fetch_data, "http_get", fail)
    assert fetch_data.main([]) == 1
    err = capsys.readouterr().err
    assert "--source" in err and "manually" in err
