import numpy as np
import pandas as pd

from src import model as M


def test_score_is_probability_times_100():
    assert M.to_score([0.0, 0.1234, 1.0]).tolist() == [0.0, 12.3, 100.0]


def test_band_cutoffs_and_assignment():
    s = np.arange(100)  # 0..99
    cut = M.band_cutoffs(s, hot=0.2, warm=0.3)
    bands = M.assign_band(s, cut)
    assert (bands == "Hot").sum() == 20
    assert (bands == "Warm").sum() == 30
    assert (bands == "Cold").sum() == 50
    assert bands[99] == "Hot" and bands[0] == "Cold"


def test_plain_labels():
    assert M.plain_label("Total Time Spent on Website") == "Time on website"
    assert M.plain_label("What is your current occupation=Missing") == "Occupation left blank"
    assert M.plain_label("Lead Origin=Lead Add Form") == "Lead origin: Lead Add Form"
    assert M.plain_label("TotalVisits__is_missing") == "Website visits not recorded"
    assert M.raw_column("City=Thane & Outskirts") == "City"


def test_group_by_raw_sums_one_hot_levels():
    feats = ["City=A", "City=B", "TotalVisits", "TotalVisits__is_missing"]
    sv = np.array([[0.1, -0.3, 0.5, 0.2]])
    g = M.group_by_raw(sv, feats)
    assert g.loc[0, "City"] == np.float64(0.1 - 0.3)
    assert np.isclose(g.loc[0, "TotalVisits"], 0.7)


def test_top_drivers_orders_by_abs_effect():
    grouped = pd.DataFrame({"City": [-0.9], "TotalVisits": [0.2], "Total Time Spent on Website": [0.5]})
    raw = pd.DataFrame({"City": ["Mumbai"], "TotalVisits": [3.0], "Total Time Spent on Website": [600.0]})
    d = M.top_drivers(grouped, raw, n=2)[0]
    assert [x["feature"] for x in d] == ["City", "Total Time Spent on Website"]
    assert d[0]["effect"] == "lowers" and d[1]["value"] == "10 min"


def test_reliability_table_bins():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    t = M.reliability_table(y, p, n_bins=5)
    assert len(t) == 5 and t["leads"].sum() == 100
