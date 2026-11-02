from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analytics, ingestion
from tests.conftest import make_students


@pytest.fixture
def prepared() -> pd.DataFrame:
    df = ingestion.merge_subjects(
        {"math": make_students(30, 1), "portuguese": make_students(50, 2)}
    )
    return ingestion.prepare(df)


def test_filter_options_and_apply(prepared):
    opts = analytics.filter_options(prepared)
    assert opts["subjects"] == ["math", "portuguese"]
    assert set(opts) >= {"schools", "sexes", "addresses", "age_range", "exclude_g3_zero"}
    f = analytics.Filters(subjects=["math"], age_range=(15, 18), exclude_g3_zero=True)
    out = analytics.apply_filters(prepared, f)
    assert (out["subject"] == "math").all()
    assert out["age"].between(15, 18).all()
    assert not out["g3_zero"].any()
    assert "subject = math" in f.describe() and "G3 = 0 excluded" in f.describe()
    assert analytics.Filters().describe() == "No filters applied"
    assert f.to_dict()["subjects"] == ["math"]


def test_filters_ignore_missing_columns(prepared):
    df = prepared.drop(columns=["school", "age", "g3_zero"])
    f = analytics.Filters(schools=["GP"], age_range=(15, 16), exclude_g3_zero=True)
    assert len(analytics.apply_filters(df, f)) == len(df)
    assert "schools" not in analytics.filter_options(df)


def test_kpis_match_manual_computation(prepared):
    k = analytics.compute_kpis(prepared)
    assert k["n_records"] == 80
    assert k["mean_g3"] == pytest.approx(prepared["G3"].mean())
    assert k["median_g3"] == pytest.approx(prepared["G3"].median())
    assert k["pass_rate"] == pytest.approx((prepared["G3"] >= 10).mean())
    assert k["pass_rate"] + k["fail_rate"] == pytest.approx(1)
    assert k["n_pass"] + k["n_fail"] == 80
    assert k["by_subject"]["math"]["n"] == 30
    exp = (prepared.loc[prepared.subject == "math", "G3"] >= 10).mean()
    assert k["by_subject"]["math"]["pass_rate"] == pytest.approx(exp)


def test_kpis_empty_and_without_g3(prepared):
    assert analytics.compute_kpis(prepared.iloc[0:0])["mean_g3"] is None
    k = analytics.compute_kpis(prepared.drop(columns=["G3", "pass"]))
    assert k["pass_rate"] is None and k["n_records"] == 80


def test_mean_g3_by_with_ci():
    df = pd.DataFrame({"studytime": [1, 1, 2, 2, 2], "G3": [10, 12, 5, 7, 9]})
    out = analytics.mean_g3_by(df, "studytime")
    assert out["n"].tolist() == [2, 3]
    assert out["mean_g3"].tolist() == [11, 7]
    assert (out["ci_low"] < out["mean_g3"]).all() and (out["ci_high"] > out["mean_g3"]).all()
    assert analytics.mean_g3_by(df, "nope").empty


def test_pass_rate_by_keeps_empty_bands(prepared):
    df = prepared.assign(absences=0)
    df = ingestion.add_derived_columns(df)
    out = analytics.pass_rate_by(df, "absence_band")
    assert len(out) == 5
    assert out.loc[0, "n"] == 80 and out["n"].iloc[1:].eq(0).all()
    assert 0 <= out.loc[0, "ci_low"] <= out.loc[0, "pass_rate"] <= out.loc[0, "ci_high"] <= 1
    assert analytics.pass_rate_by(df.drop(columns=["pass"]), "absence_band").empty


def test_correlations(prepared):
    corr = analytics.correlations_with_g3(prepared)
    assert "G3" not in corr["feature"].tolist()
    assert corr["r"].is_monotonic_increasing
    r = np.corrcoef(prepared["G2"], prepared["G3"])[0, 1]
    assert corr.set_index("feature").loc["G2", "r"] == pytest.approx(r)
    const = prepared.assign(age=16)
    assert "age" not in analytics.correlations_with_g3(const)["feature"].tolist()
    assert analytics.correlations_with_g3(prepared.drop(columns=["G3"])).empty
    m = analytics.correlation_matrix(prepared)
    assert m.shape[0] == m.shape[1] and "G3" in m.columns


def test_grade_counts_and_pairs(prepared):
    counts = analytics.grade_counts(prepared)
    assert len(counts) == 21 and counts["n"].sum() == 80
    assert set(counts.loc[counts.G3 < 10, "outcome"]) == {"Fail"}
    pairs = analytics.grade_pair_counts(prepared)
    assert pairs["n"].sum() == 80
    assert analytics.grade_pair_counts(prepared.drop(columns=["G2"])).empty
    assert analytics.grade_counts(prepared.drop(columns=["G3"])).empty


def test_outcome_labels_and_describe(prepared):
    labels = analytics.outcome_labels(prepared)
    assert set(labels) <= {"Pass", "Fail"}
    assert "G3" in analytics.describe_numeric(prepared).index
    assert analytics.describe_numeric(pd.DataFrame({"x": [1]})).empty
