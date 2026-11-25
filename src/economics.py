"""Turn calibrated probabilities into a call/no-call decision in rupees.

Every price, margin, cost and capacity figure comes from config.yaml and is an
ASSUMPTION. Simulations on historical data assume a lead's observed outcome is
what calling it would produce (no treatment data, so no causal / uplift claim):
"profit" = enrolments among the leads called × price × margin − calls × cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import load_config


@dataclass(frozen=True)
class Assumptions:
    course_price: float
    gross_margin: float
    cost_per_call: float
    counsellors: int = 5
    calls_per_counsellor_per_day: int = 40
    days_to_work_batch: int = 5
    currency: str = "₹"

    @classmethod
    def from_config(cls, cfg: dict | None = None, **overrides) -> "Assumptions":
        cfg = cfg or load_config()
        e, c = cfg["economics"], cfg["capacity"]
        vals = dict(course_price=e["course_price"], gross_margin=e["gross_margin"],
                    cost_per_call=e["cost_per_call"], counsellors=c["counsellors"],
                    calls_per_counsellor_per_day=c["calls_per_counsellor_per_day"],
                    days_to_work_batch=c["days_to_work_batch"], currency=e["currency"])
        vals.update(overrides)
        return cls(**vals)

    @property
    def value_per_enrolment(self) -> float:
        return self.course_price * self.gross_margin

    @property
    def break_even_probability(self) -> float:
        """A call pays off when P(convert) exceeds this."""
        return self.cost_per_call / self.value_per_enrolment

    @property
    def calls_per_day(self) -> int:
        return self.counsellors * self.calls_per_counsellor_per_day

    @property
    def batch_capacity(self) -> int:
        return self.calls_per_day * self.days_to_work_batch

    def label(self) -> str:
        c = self.currency
        return (f"ASSUMPTIONS: price {c}{self.course_price:,.0f}, margin {self.gross_margin:.0%}, "
                f"cost/call {c}{self.cost_per_call:,.0f}, capacity {self.calls_per_day} calls/day "
                f"({self.counsellors} counsellors × {self.calls_per_counsellor_per_day})")


def ev_per_call(p, a: Assumptions) -> np.ndarray:
    """Expected value of one call = P(convert) × price × margin − cost per call."""
    return np.asarray(p, dtype=float) * a.value_per_enrolment - a.cost_per_call


def rank(p) -> np.ndarray:
    """Indices from highest to lowest probability (stable)."""
    return np.argsort(-np.asarray(p, dtype=float), kind="mergesort")


def optimal_calls(p, a: Assumptions, capacity: int | None = None) -> int:
    """Number of leads to call (in score order): every lead with EV > 0,
    capped at `capacity` if given."""
    n = int((ev_per_call(p, a) > 0).sum())
    return min(n, capacity) if capacity is not None else n


def simulate(y, p, n_calls: int, a: Assumptions) -> dict:
    """Outcome of calling the top `n_calls` leads by score, using observed outcomes."""
    y = np.asarray(y, dtype=float)
    order = rank(p)
    called = y[order][:n_calls]
    enrol = called.sum()
    total = y.sum()
    profit = enrol * a.value_per_enrolment - n_calls * a.cost_per_call
    return {"calls": int(n_calls), "pct_leads_called": n_calls / len(y),
            "enrolments": float(enrol),
            "pct_enrolments_captured": enrol / total if total else 0.0,
            "profit": float(profit),
            "expected_profit": float(ev_per_call(np.asarray(p)[order][:n_calls], a).sum())}


def simulate_all(y, a: Assumptions) -> dict:
    y = np.asarray(y, dtype=float)
    n = len(y)
    return {"calls": n, "pct_leads_called": 1.0, "enrolments": float(y.sum()),
            "pct_enrolments_captured": 1.0,
            "profit": float(y.sum() * a.value_per_enrolment - n * a.cost_per_call)}


def simulate_random(y, n_calls: int, a: Assumptions) -> dict:
    """Expected outcome of calling `n_calls` leads picked at random."""
    y = np.asarray(y, dtype=float)
    enrol = n_calls * y.mean()
    return {"calls": int(n_calls), "pct_leads_called": n_calls / len(y),
            "enrolments": float(enrol),
            "pct_enrolments_captured": n_calls / len(y),
            "profit": float(enrol * a.value_per_enrolment - n_calls * a.cost_per_call)}


def profit_curve(y, p, a: Assumptions) -> pd.DataFrame:
    """Cumulative realised and expected profit after each call in score order."""
    y = np.asarray(y, dtype=float)
    order = rank(p)
    ys, ps = y[order], np.asarray(p, dtype=float)[order]
    n = np.arange(1, len(y) + 1)
    return pd.DataFrame({
        "calls": n, "pct_leads_called": n / len(y),
        "enrolments": np.cumsum(ys),
        "pct_enrolments_captured": np.cumsum(ys) / ys.sum(),
        "profit": np.cumsum(ys) * a.value_per_enrolment - n * a.cost_per_call,
        "expected_profit": np.cumsum(ev_per_call(ps, a)),
        "random_profit": n * ys.mean() * a.value_per_enrolment - n * a.cost_per_call,
    })


def sensitivity(y, p, base: Assumptions, prices, costs,
                capacity: int | None = None) -> pd.DataFrame:
    """Grid of price × cost-per-call → EV-optimal calling vs everyone vs random."""
    rows = []
    for price in prices:
        for cost in costs:
            a = Assumptions(**{**base.__dict__, "course_price": price, "cost_per_call": cost})
            n = optimal_calls(p, a, capacity)
            m, everyone = simulate(y, p, n, a), simulate_all(y, a)
            rnd = simulate_random(y, n, a)
            rows.append({
                "course_price": price, "cost_per_call": cost,
                "break_even_p": a.break_even_probability,
                "pct_leads_called": m["pct_leads_called"],
                "pct_enrolments_captured": m["pct_enrolments_captured"],
                "profit_model": m["profit"],
                "profit_call_everyone": everyone["profit"],
                "profit_random_same_calls": rnd["profit"],
                "gain_vs_everyone": m["profit"] - everyone["profit"],
                "gain_vs_random": m["profit"] - rnd["profit"],
            })
    return pd.DataFrame(rows)
