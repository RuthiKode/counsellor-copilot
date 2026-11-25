"""Toy-example tests for ranking metrics. Expected values worked out by hand."""
import math

import numpy as np
import pytest

from src import metrics

# 10 leads, 4 converters. Scores rank them 0..9 (index 0 = highest score).
Y = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 0])
S = np.linspace(1.0, 0.1, 10)


def test_n_top_rounds_up():
    assert metrics.n_top(10, 0.2) == 2
    assert metrics.n_top(10, 0.25) == 3
    assert metrics.n_top(3, 0.1) == 1
    with pytest.raises(ValueError):
        metrics.n_top(10, 0)


def test_precision_recall_at_k():
    # top 20% = leads 0,1 -> both convert
    assert metrics.precision_at_k(Y, S, 0.2) == 1.0
    assert metrics.recall_at_k(Y, S, 0.2) == 0.5
    # top 40% = leads 0..3 -> 3 converters
    assert metrics.precision_at_k(Y, S, 0.4) == 0.75
    assert metrics.recall_at_k(Y, S, 0.4) == 0.75
    assert metrics.recall_at_k(Y, S, 1.0) == 1.0


def test_ranking_uses_score_not_order():
    perm = np.array([9, 3, 0, 7, 1, 2, 4, 5, 6, 8])
    assert metrics.precision_at_k(Y[perm], S[perm], 0.4) == 0.75


def test_lift_and_calls_per_enrolment():
    assert metrics.lift_at_k(Y, S, 0.2) == pytest.approx(1.0 / 0.4)
    assert metrics.calls_per_enrolment(Y, S, 0.4) == pytest.approx(1 / 0.75)
    assert metrics.random_calls_per_enrolment(Y) == pytest.approx(2.5)


def test_lift_by_decile():
    t = metrics.lift_by_decile(Y, S)
    assert len(t) == 10 and t["leads"].sum() == 10
    assert list(t["conversions"]) == list(Y)
    assert t["cum_recall"].iloc[-1] == pytest.approx(1.0)
    assert t["lift"].iloc[0] == pytest.approx(2.5)


def test_gains_curve():
    g = metrics.gains_curve(Y, S)
    assert g.iloc[0].tolist() == [0.0, 0.0, 0.0]
    assert g["frac_captured"].iloc[-1] == pytest.approx(1.0)
    assert g["frac_captured"].iloc[2] == pytest.approx(0.5)
    assert (np.diff(g["frac_captured"]) >= 0).all()


def test_perfect_and_random_models():
    y = np.array([1] * 3 + [0] * 7)
    assert metrics.precision_at_k(y, -np.arange(10), 0.3) == 1.0
    assert math.isinf(metrics.calls_per_enrolment(np.zeros(5), np.arange(5), 0.2))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.precision_at_k([1, 0], [0.5], 0.5)
