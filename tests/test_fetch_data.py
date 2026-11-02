from __future__ import annotations

import io
import zipfile

from scripts import fetch_data


def _nested_zip() -> bytes:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("student-mat.csv", "a")
        z.writestr("student-por.csv", "b")
        z.writestr("student.txt", "ignored")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("student.zip", inner.getvalue())
    return outer.getvalue()


def test_extract_nested_members():
    with zipfile.ZipFile(io.BytesIO(_nested_zip())) as z:
        files = fetch_data._extract_members(z, {"student-mat.csv", "student-por.csv"})
    assert files == {"student-mat.csv": b"a", "student-por.csv": b"b"}


def test_files_from_source_zip_and_dir(tmp_path):
    zp = tmp_path / "d.zip"
    zp.write_bytes(_nested_zip())
    assert len(fetch_data.files_from_source(zp, {"student-mat.csv"})) == 1
    (tmp_path / "student-por.csv").write_text("x")
    assert fetch_data.files_from_source(tmp_path, {"student-por.csv"}) == {"student-por.csv": b"x"}


def test_manual_instructions_mentions_target(tmp_path):
    text = fetch_data.manual_instructions(tmp_path)
    assert str(tmp_path) in text and "--source" in text
