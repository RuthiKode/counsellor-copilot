import numpy as np
import pandas as pd
import pytest

from src.clean import (ID_COLS, LEAKY_COLS, LeadCleaner, load_raw, merge_rare,
                       missingness_is_predictive, yes_no_to_int)


def toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return pd.DataFrame({
        "Prospect ID": [f"id{i}" for i in range(n)],
        "Lead Number": range(n),
        "Tags": rng.choice(["Will revert", "Closed"], n),
        "Lead Source": rng.choice(["Google", "google", "Direct", "Rare"], n, p=[.45, .05, .495, .005]),
        "Do Not Email": rng.choice(["Yes", "No"], n),
        "Magazine": ["No"] * n,
        "TotalVisits": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 10, n)).astype(float),
        "Blanky": np.where(y == 1, None, "x"),   # 50% missing, perfectly predictive
        "Noise": np.where(rng.random(n) < 0.6, None, "x"),  # >40% missing, not predictive
        "Converted": y,
    })


def test_load_raw_treats_select_as_missing(tmp_path):
    p = tmp_path / "leads.csv"
    p.write_text("A,Converted\nSelect,1\nfoo,0\n")
    df = load_raw(p)
    assert df["A"].isna().tolist() == [True, False]


def test_load_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")


def test_merge_rare():
    s = pd.Series(["a"] * 98 + ["b", None])
    out = merge_rare(s, min_share=0.02)
    assert out.iloc[98] == "Other" and pd.isna(out.iloc[99])


def test_yes_no_to_int():
    assert yes_no_to_int(pd.Series(["Yes", "no", " YES ", None])).tolist()[:3] == [1, 0, 1]


def test_missingness_is_predictive():
    y = pd.Series([1] * 50 + [0] * 50)
    assert missingness_is_predictive(pd.Series([True] * 50 + [False] * 50), y)
    assert not missingness_is_predictive(pd.Series([True, False] * 50), y)


def test_cleaner_drops_ids_leaky_constant_and_uninformative_missing():
    df = toy()
    cl = LeadCleaner().fit(df)
    X = cl.transform(df)
    cols = " ".join(X.columns)
    for c in ID_COLS + ["Tags", "Magazine", "Noise", "Converted"]:
        assert c not in cols
    assert any(c.startswith("Blanky=") for c in X.columns)   # predictive missingness kept
    assert "Blanky=Missing" in X.columns
    assert set(LEAKY_COLS) & set(cl.drop_cols) == set(LEAKY_COLS)


def test_cleaner_encodes_and_caps():
    df = toy()
    cl = LeadCleaner(cap_quantile=0.5).fit(df)
    X = cl.transform(df)
    assert X["TotalVisits"].max() <= cl.caps_["TotalVisits"]
    assert X["TotalVisits"].isna().sum() == 0
    assert X["TotalVisits__is_missing"].sum() == df["TotalVisits"].isna().sum()
    assert set(X["Do Not Email"].unique()) <= {0, 1}
    # google merged into Google; rare level folded into Other
    assert "Lead Source=google" not in X.columns
    assert "Lead Source=Rare" not in X.columns


def test_transform_is_stable_on_new_data():
    df = toy()
    cl = LeadCleaner().fit(df)
    new = toy(20, seed=1)
    new.loc[0, "Lead Source"] = "Never seen"
    new = new.drop(columns=["Converted", "Tags"])  # new leads have no target / leaky cols
    X = cl.transform(new)
    assert list(X.columns) == cl.feature_names_
    assert X.loc[0, "Lead Source=Other"] == 1


def test_transform_requires_columns():
    cl = LeadCleaner().fit(toy())
    with pytest.raises(ValueError):
        cl.transform(toy().drop(columns=["TotalVisits"]))
