from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import config, ingestion
from tests.conftest import make_students


def _csv_bytes(df: pd.DataFrame, sep: str = ";") -> bytes:
    return df.to_csv(index=False, sep=sep).encode()


# ---- reading ------------------------------------------------------------------
@pytest.mark.parametrize("sep", [";", ","])
def test_read_csv_sniffs_delimiter(students, sep):
    df = ingestion.read_table_bytes(_csv_bytes(students, sep), "x.csv")
    assert list(df.columns) == list(students.columns)
    assert len(df) == len(students)


def test_read_xlsx_roundtrip(students):
    buf = io.BytesIO()
    students.to_excel(buf, index=False, engine="openpyxl")
    df = ingestion.read_table_bytes(buf.getvalue(), "x.xlsx")
    assert df.shape == students.shape


@pytest.mark.parametrize(
    ("data", "name", "msg"),
    [
        (b"a,b\n1,2", "x.txt", "Unsupported"),
        (b"", "x.csv", "empty"),
        (b"a;b\n", "x.csv", "no data"),
    ],
)
def test_read_rejects_bad_files(data, name, msg):
    with pytest.raises(ingestion.IngestionError, match=msg):
        ingestion.read_table_bytes(data, name)


def test_read_rejects_non_utf8():
    with pytest.raises(ingestion.IngestionError, match="UTF-8"):
        ingestion.read_table_bytes(b"a;b\n\xff\xfe;1", "x.csv")


def test_read_table_missing_file(tmp_path):
    with pytest.raises(ingestion.IngestionError, match="not found"):
        ingestion.read_table(tmp_path / "nope.csv")


# ---- validation ---------------------------------------------------------------
def test_valid_table_is_uci_mode(students):
    report = ingestion.validate(students)
    assert report.is_valid
    assert report.mode == "uci"
    assert report.missing_columns == []
    assert report.to_dict()["is_valid"] is True


def test_quoted_strings_are_accepted(students):
    quoted = students.copy()
    quoted["school"] = '"' + quoted["school"] + '"'
    assert ingestion.validate(quoted).is_valid


def test_out_of_range_grade_is_error(students):
    bad = students.copy()
    bad.loc[3, "G3"] = 25
    report = ingestion.validate(bad)
    assert not report.is_valid
    (issue,) = [i for i in report.errors if i.column == "G3"]
    assert issue.rows == [4] and issue.count == 1


def test_non_numeric_and_decimal_are_errors(students):
    bad = students.astype({"age": object, "absences": float})
    bad.loc[0, "age"] = "seventeen"
    bad.loc[1, "absences"] = 2.5
    cols = {i.column for i in ingestion.validate(bad).errors}
    assert {"age", "absences"} <= cols


def test_invalid_category_is_error(students):
    bad = students.copy()
    bad.loc[0, "sex"] = "X"
    (issue,) = ingestion.validate(bad).errors
    assert issue.column == "sex" and "X" in issue.message


def test_missing_values_are_warnings_not_imputed(students):
    bad = students.astype({"G2": float})
    bad.loc[[0, 5], "G2"] = np.nan
    report = ingestion.validate(bad)
    assert report.is_valid
    assert report.missing_values["G2"] == 2
    cleaned = ingestion.clean(bad)
    assert cleaned["G2"].isna().sum() == 2  # never fabricated


def test_missing_columns_switch_to_custom_mode(students):
    report = ingestion.validate(students.drop(columns=["G1", "G2"]).assign(extra=1))
    assert report.is_valid
    assert report.mode == "custom"
    assert report.missing_columns == ["G1", "G2"]
    assert report.extra_columns == ["extra"]


def test_no_schema_columns_is_error():
    report = ingestion.validate(pd.DataFrame({"foo": [1], "bar": [2]}))
    assert not report.is_valid and report.mode == "custom"


def test_missing_subject_label_is_error(students):
    bad = students.assign(subject="math")
    bad.loc[0, "subject"] = None
    assert any(i.column == "subject" for i in ingestion.validate(bad).errors)


# ---- derivation ---------------------------------------------------------------
def test_derived_columns(students):
    df = students.copy()
    df["G3"] = [0, 9, 10, 20] * 10
    df["absences"] = [0, 5, 6, 21] * 10
    out = ingestion.add_derived_columns(ingestion.clean(df))
    assert out["pass"].tolist()[:4] == [0, 0, 1, 1]
    assert out["g3_zero"].tolist()[:4] == [True, False, False, False]
    assert out["absence_band"].astype(str).tolist()[:4] == ["0", "1–5", "6–10", "20+"]


@pytest.mark.parametrize(("absences", "band"), [(10, "6–10"), (11, "11–20"), (20, "11–20")])
def test_absence_band_edges(students, absences, band):
    df = students.assign(absences=absences)
    assert (ingestion.add_derived_columns(df)["absence_band"].astype(str) == band).all()


def test_pass_is_nan_where_g3_missing(students):
    df = students.astype({"G3": float})
    df.loc[0, "G3"] = np.nan
    out = ingestion.add_derived_columns(ingestion.clean(df))
    assert np.isnan(out.loc[0, "pass"])


def test_merge_and_cross_subject_flag(students):
    other = students.copy()
    other.loc[0, "age"] = 22 if other.loc[0, "age"] != 22 else 15  # break one match
    merged = ingestion.merge_subjects({"math": students, "portuguese": other})
    out = ingestion.prepare(merged)
    assert out["subject"].value_counts().to_dict() == {"math": 40, "portuguese": 40}
    assert not out.loc[0, "in_both_subjects"]
    summary = ingestion.data_quality_summary(out)
    assert summary["n_records"] == 80
    assert summary["students_in_both_subjects"] >= 39
    assert summary["cross_subject_match_pairs"] >= 39
    assert "Data-quality summary" in ingestion.format_quality_summary(summary)


def test_student_keys_need_all_match_columns(students):
    assert ingestion.student_keys(students.drop(columns=["school"])).isna().all()


# ---- loading entry points -----------------------------------------------------
def _write_uci(tmp_path, n_mat=395, n_por=649):
    for (subject, name), n in zip(config.RAW_FILES.items(), (n_mat, n_por), strict=True):
        (tmp_path / name).write_bytes(_csv_bytes(make_students(n, seed=len(subject))))


def test_load_uci_dataset_from_folder(tmp_path):
    _write_uci(tmp_path)
    result = ingestion.load_uci_dataset(tmp_path)
    assert result.is_valid
    assert len(result.data) == 1044
    assert set(result.checksums) == set(config.RAW_FILES.values())
    assert all(len(v) == 64 for v in result.checksums.values())


def test_load_uci_dataset_row_count_warning(tmp_path):
    _write_uci(tmp_path, n_mat=10)
    result = ingestion.load_uci_dataset(tmp_path)
    assert result.is_valid
    assert any("Expected 395" in i.message for i in result.reports[0].warnings)


def test_load_uci_dataset_missing_file(tmp_path):
    result = ingestion.load_uci_dataset(tmp_path)
    assert not result.is_valid and result.data is None


def test_load_uci_dataset_rejects_partial_schema(tmp_path):
    _write_uci(tmp_path)
    p = tmp_path / config.RAW_FILES["math"]
    p.write_bytes(_csv_bytes(make_students(395).drop(columns=["G1"])))
    assert not ingestion.load_uci_dataset(tmp_path).is_valid


def test_load_custom_table(students):
    df, report = ingestion.load_custom_table(students.drop(columns=["G1"]), "u.csv")
    assert report.mode == "custom" and df is not None
    assert (df["subject"] == "unspecified").all()
    df2, _ = ingestion.load_custom_table(students, "u.csv", subject=" Math ")
    assert (df2["subject"] == "math").all()
    bad = students.assign(G3=99)
    df3, report3 = ingestion.load_custom_table(bad, "u.csv")
    assert df3 is None and not report3.is_valid


def test_capabilities():
    caps = ingestion.assess_capabilities(["G3", "absences"])
    assert caps["KPIs and grade distribution"] == []
    assert caps["G2 vs G3 scatter"] == ["G2"]
    assert caps["Model setting B (early warning)"]
    full = ingestion.assess_capabilities(list(config.REQUIRED_COLUMNS))
    assert all(not v for v in full.values())


def test_save_upload_preserves_bytes_and_dedupes(tmp_path):
    data = b"school;sex\nGP;F\n"
    p1 = ingestion.save_upload(data, "../../evil name.csv", tmp_path)
    p2 = ingestion.save_upload(data, "../../evil name.csv", tmp_path)
    assert p1 == p2 and p1.parent == tmp_path
    assert p1.read_bytes() == data
    assert p1.name.endswith("evil_name.csv")
    assert ingestion.sha256_file(p1) == ingestion.sha256_bytes(data)


def test_save_processed(tmp_path, students):
    out = ingestion.save_processed(students, tmp_path / "sub" / "x.csv")
    assert pd.read_csv(out).shape == students.shape


# ---- real dataset (skipped when not downloaded) ------------------------------
@pytest.mark.skipif(
    not all((config.RAW_DIR / f).exists() for f in config.RAW_FILES.values()),
    reason="UCI files not downloaded",
)
def test_real_uci_dataset_baseline():
    result = ingestion.load_uci_dataset()
    assert result.is_valid
    s = ingestion.data_quality_summary(result.data)
    assert s["records_by_subject"] == {"math": 395, "portuguese": 649}
    assert s["cross_subject_match_pairs"] == 382  # matches Cortez & Silva's merge
    assert s["g3_zero_by_subject"] == {"math": 38, "portuguese": 15}
