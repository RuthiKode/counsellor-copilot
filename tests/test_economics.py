import numpy as np
import pytest

from src import economics as E

A = E.Assumptions(course_price=10_000, gross_margin=0.5, cost_per_call=500,
                  counsellors=2, calls_per_counsellor_per_day=10, days_to_work_batch=1)
# value per enrolment = 5,000 -> break-even probability = 0.1
P = np.array([0.9, 0.5, 0.2, 0.1, 0.05])
Y = np.array([1, 0, 1, 0, 0])


def test_break_even_and_ev():
    assert A.value_per_enrolment == 5_000
    assert A.break_even_probability == pytest.approx(0.1)
    assert E.ev_per_call([0.9, 0.1], A).tolist() == pytest.approx([4_000, 0])


def test_optimal_calls_stops_when_ev_not_positive():
    assert E.optimal_calls(P, A) == 3          # 0.9, 0.5, 0.2 have EV > 0; 0.1 is break-even
    assert E.optimal_calls(P, A, capacity=2) == 2
    assert A.batch_capacity == 20


def test_simulate_uses_score_order_and_observed_outcomes():
    perm = np.array([3, 0, 4, 2, 1])
    r = E.simulate(Y[perm], P[perm], 3, A)
    assert r["enrolments"] == 2
    assert r["profit"] == 2 * 5_000 - 3 * 500
    assert r["pct_enrolments_captured"] == 1.0
    assert r["expected_profit"] == pytest.approx((0.9 + 0.5 + 0.2) * 5_000 - 1_500)


def test_everyone_and_random():
    assert E.simulate_all(Y, A)["profit"] == 2 * 5_000 - 5 * 500
    r = E.simulate_random(Y, 2, A)
    assert r["enrolments"] == pytest.approx(0.8)
    assert r["profit"] == pytest.approx(0.8 * 5_000 - 1_000)


def test_profit_curve_matches_simulate():
    c = E.profit_curve(Y, P, A)
    assert c.loc[2, "profit"] == E.simulate(Y, P, 3, A)["profit"]
    assert c["calls"].iloc[-1] == 5


def test_sensitivity_grid_shape_and_labels():
    t = E.sensitivity(Y, P, A, prices=[5_000, 10_000], costs=[100, 500])
    assert len(t) == 4
    row = t[(t.course_price == 10_000) & (t.cost_per_call == 500)].iloc[0]
    assert row.pct_leads_called == pytest.approx(0.6)
    assert row.gain_vs_everyone == pytest.approx(8_500 - 7_500)


def test_label_mentions_assumptions():
    assert "ASSUMPTIONS" in A.label() and "10,000" in A.label()
