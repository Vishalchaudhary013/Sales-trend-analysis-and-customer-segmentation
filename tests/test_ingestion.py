from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import config, ingestion


def _csv(df: pd.DataFrame, sep: str = ",", encoding: str = "utf-8") -> bytes:
    return df.to_csv(index=False, sep=sep).encode(encoding)


# ---- reading -------------------------------------------------------------------
@pytest.mark.parametrize("sep", [",", ";"])
def test_read_csv_keeps_codes_as_text(raw, sep):
    raw = raw.copy()
    raw.loc[0, "StockCode"] = "00123"
    df = ingestion.read_table_bytes(_csv(raw, sep), "x.csv")
    assert df.loc[0, "StockCode"] == "00123"
    assert len(df) == len(raw)


def test_read_latin1(raw):
    raw = raw.copy()
    raw.loc[0, "Description"] = "CAFÉ MUG"
    df = ingestion.read_table_bytes(_csv(raw, encoding="latin-1"), "x.csv")
    assert df.loc[0, "Description"] == "CAFÉ MUG"


def test_read_xlsx(raw):
    buf = io.BytesIO()
    raw.to_excel(buf, index=False, engine="openpyxl")
    assert ingestion.read_table_bytes(buf.getvalue(), "x.xlsx").shape == raw.shape


@pytest.mark.parametrize(
    ("data", "name", "msg"),
    [(b"a,b\n1,2", "x.txt", "Unsupported"), (b"", "x.csv", "empty"),
     (b"a,b\n", "x.csv", "no data"), (b"not an excel file", "x.xlsx", "Could not parse")],
)  # fmt: skip
def test_read_rejects_bad_files(data, name, msg):
    with pytest.raises(ingestion.IngestionError, match=msg):
        ingestion.read_table_bytes(data, name)


def test_read_missing_file(tmp_path):
    with pytest.raises(ingestion.IngestionError, match="not found"):
        ingestion.read_table(tmp_path / "nope.csv")


# ---- column resolution ---------------------------------------------------------
def test_aliases_are_mapped_and_reported(raw):
    renamed = raw.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price",
                                  "CustomerID": "Customer ID"})  # fmt: skip
    df, mapping = ingestion.resolve_columns(renamed)
    assert mapping == {"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"}
    assert set(config.ALL_COLUMNS) <= set(df.columns)
    report = ingestion.validate(renamed)
    assert report.is_valid and report.column_mapping == mapping
    assert any("aliases" in i.message for i in report.warnings)


def test_unrelated_columns_are_not_substituted(raw):
    df = raw.drop(columns=["UnitPrice"]).assign(Cost=1.0)
    report = ingestion.validate(df)
    assert not report.is_valid
    assert report.missing_required == ["UnitPrice"]


def test_optional_and_extra_columns(raw):
    report = ingestion.validate(raw.drop(columns=["Country", "Description"]).assign(Notes="x"))
    assert report.is_valid
    assert report.missing_optional == ["Description", "Country"]
    assert report.extra_columns == ["Notes"]


# ---- dates ---------------------------------------------------------------------
def test_detect_us_dates_and_ambiguity():
    s = pd.Series(["12/1/2010 8:26", "1/13/2011 10:00"])
    assert ingestion.detect_date_format(s) == ("%m/%d/%Y %H:%M", [])
    fmt, alts = ingestion.detect_date_format(pd.Series(["1/2/2011 8:00", "3/4/2011 9:00"]))
    assert fmt == "%m/%d/%Y %H:%M" and "%d/%m/%Y %H:%M" in alts


def test_detect_iso_native_and_unknown():
    assert ingestion.detect_date_format(pd.Series(["2011-01-02 08:00:00"]))[0] == (
        "%Y-%m-%d %H:%M:%S"
    )
    assert ingestion.detect_date_format(pd.to_datetime(pd.Series(["2011-01-02"])))[0] == "native"
    assert ingestion.detect_date_format(pd.Series(["yesterday"])) == (None, [])
    assert ingestion.detect_date_format(pd.Series([None, None])) == (None, [])
    assert ingestion.parse_dates(pd.Series(["x"]), None).isna().all()


def test_ambiguous_dates_warn(raw):
    df = raw.assign(InvoiceDate="1/2/2011 8:00")
    report = ingestion.validate(df)
    assert any("ambiguous" in i.message for i in report.warnings)


def test_unrecognised_dates_error(raw):
    report = ingestion.validate(raw.assign(InvoiceDate="last tuesday"))
    assert not report.is_valid and report.date_format is None


# ---- validation of values -----------------------------------------------------
def test_invalid_rows_are_counted_and_dropped(raw):
    df = raw.copy()
    df["Quantity"] = df["Quantity"].astype(object)
    df.loc[0, "Quantity"] = "two"
    df.loc[1, "UnitPrice"] = np.nan
    df.loc[2, "InvoiceDate"] = "garbage"
    df.loc[3, "InvoiceNo"] = None
    report = ingestion.validate(df)
    assert report.is_valid
    assert report.invalid_rows == 4
    result = ingestion.load_table(df, "x")
    assert result.log.dropped_invalid == 4
    assert result.log.rows_out == len(df) - 4 - result.log.dropped_duplicates


def test_too_many_invalid_rows_is_error(raw):
    df = raw.assign(UnitPrice="n/a")
    assert not ingestion.validate(df).is_valid


def test_business_rule_flags(raw):
    df = raw.copy()
    df.loc[0, "InvoiceNo"] = "C536000"
    df.loc[0, "Quantity"] = -3
    df.loc[1, "InvoiceNo"] = "A563186"
    df.loc[1, "UnitPrice"] = -100.0
    df.loc[2, "UnitPrice"] = 0.0
    df.loc[3, "StockCode"] = "POST"
    df.loc[4, "Quantity"] = -5
    df.loc[5, "Description"] = None
    df = pd.concat([df, df.iloc[[6]]], ignore_index=True)  # exact duplicate
    msgs = " | ".join(i.message for i in ingestion.validate(df).warnings)
    for needle in ("cancellation", "adjustment", "zero unit price", "negative unit price",
                   "non-cancellation", "non-product", "duplicate", "description",
                   "without a CustomerID"):  # fmt: skip
        assert needle in msgs, needle


# ---- cleaning ------------------------------------------------------------------
def test_prepare_types_and_derived_columns(raw):
    df = raw.copy()
    df.loc[0, "InvoiceNo"] = "C536000"
    df.loc[0, "Quantity"] = -2
    df.loc[1, "StockCode"] = "gift_0001_20"
    result = ingestion.load_table(df, "x")
    data = result.data
    assert result.is_valid
    assert pd.api.types.is_datetime64_any_dtype(data["InvoiceDate"])
    assert data["Quantity"].dtype == "int64"
    assert (data["Revenue"] == data["Quantity"] * data["UnitPrice"]).all()
    assert data["IsCancellation"].sum() == 1
    assert not data.loc[data.StockCode == "gift_0001_20", "IsProduct"].any()
    assert data["CustomerID"].dropna().str.fullmatch(r"\d+").all()  # "12000.0" -> "12000"
    assert data["HasCustomer"].sum() == data["CustomerID"].notna().sum()
    assert data["InvoiceDate"].is_monotonic_increasing
    mask = ingestion.sales_lines(data)
    assert not data.loc[mask, "IsCancellation"].any() and (data.loc[mask, "Quantity"] > 0).all()


def test_prepare_drops_exact_duplicates(raw):
    df = pd.concat([raw, raw.iloc[:5]], ignore_index=True)
    result = ingestion.load_table(df, "x")
    assert result.log.dropped_duplicates == 5
    assert result.log.rows_out == len(raw)


def test_customer_id_normalisation():
    s = pd.Series(["17850.0", " 12346 ", None, "", "ABC-1"])
    out = ingestion.normalise_customer_ids(s).tolist()
    assert out[0] == "17850" and out[1] == "12346" and out[4] == "ABC-1"
    assert pd.isna(out[2]) and pd.isna(out[3])


def test_load_table_rejects_invalid(raw):
    result = ingestion.load_table(raw.drop(columns=["InvoiceDate"]), "x")
    assert not result.is_valid and result.data is None


# ---- files, persistence --------------------------------------------------------
def test_load_file_and_checksum(tmp_path, raw):
    p = tmp_path / "t.csv"
    p.write_bytes(_csv(raw))
    result = ingestion.load_file(p)
    assert result.is_valid and result.checksum == ingestion.sha256_file(p)
    assert len(result.checksum) == 64


def test_load_file_unreadable(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("x")
    result = ingestion.load_file(p)
    assert not result.is_valid and "Unsupported" in result.report.errors[0].message


def test_bundled_row_count_warning(tmp_path, raw):
    p = tmp_path / "t.csv"
    p.write_bytes(_csv(raw))
    result = ingestion.load_bundled_dataset(p)
    assert any("Expected 541,909" in i.message for i in result.report.warnings)


def test_save_upload_and_processed(tmp_path, raw):
    data = _csv(raw)
    p1 = ingestion.save_upload(data, "../evil name.csv", tmp_path)
    assert p1 == ingestion.save_upload(data, "../evil name.csv", tmp_path)
    assert p1.parent == tmp_path and p1.read_bytes() == data and p1.name.endswith("evil_name.csv")
    clean = ingestion.load_table(raw, "x").data
    out = ingestion.save_processed(clean, tmp_path / "sub" / "c.csv")
    back = pd.read_csv(out)
    assert len(back) == len(clean)


# ---- capabilities and summary --------------------------------------------------
def test_capabilities(raw):
    clean = ingestion.load_table(raw, "x").data
    caps = ingestion.assess_capabilities(clean)
    assert caps["Sales KPIs and trends"] == []
    assert caps["RFM and K-Means segmentation"] == []
    no_country = ingestion.assess_capabilities(clean.drop(columns=["Country"]))
    assert no_country["Country analysis"] == ["missing column Country"]
    guests = ingestion.assess_capabilities(clean.assign(CustomerID=pd.NA))
    assert "no customer IDs present" in guests["RFM and K-Means segmentation"]
    short = clean[clean.InvoiceDate < clean.InvoiceDate.min() + pd.Timedelta(days=30)]
    assert any("history spans" in r for r in ingestion.assess_capabilities(short)[
        "Churn early-warning model"])  # fmt: skip


def test_quality_summary(raw):
    result = ingestion.load_table(raw, "x")
    s = ingestion.data_quality_summary(result.data, result.log)
    assert s["rows"] == len(result.data)
    assert s["customers"] == result.data["CustomerID"].nunique()
    mask = ingestion.sales_lines(result.data)
    assert s["gross_product_revenue"] == pytest.approx(result.data.loc[mask, "Revenue"].sum())
    assert 0 <= s["customer_share_of_gross_revenue"] <= 1
    assert "Data-quality summary" in ingestion.format_quality_summary(s)


# ---- real dataset (skipped when not downloaded) --------------------------------
@pytest.mark.skipif(not config.RAW_FILE.exists(), reason="dataset not downloaded")
def test_real_dataset_reference_figures():
    result = ingestion.load_bundled_dataset()
    assert result.is_valid
    assert result.report.n_rows == 541_909
    assert result.log.dropped_duplicates == 5_268
    s = ingestion.data_quality_summary(result.data)
    assert s["invoices"] == 25_900
    assert s["customers"] == 4_372
    assert s["countries"] == 38
