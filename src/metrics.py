"""Ranking metrics a sales manager cares about.

All functions take true labels `y` (0/1) and a score where higher = more
likely to convert. Leads are ranked by score (ties keep input order) and the
top k% are "called".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _ranked(y, score) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    if y.shape != score.shape:
        raise ValueError("y and score must have the same length")
    order = np.argsort(-score, kind="mergesort")
    return y[order]


def n_top(n: int, k: float) -> int:
    """Number of leads in the top-k fraction (rounded up, at least 1)."""
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    return max(1, math.ceil(round(k * n, 9)))


def precision_at_k(y, score, k: float) -> float:
    """Share of the top-k% leads that converted."""
    r = _ranked(y, score)
    return float(r[: n_top(len(r), k)].mean())


def recall_at_k(y, score, k: float) -> float:
    """Share of all conversions captured in the top-k% leads."""
    r = _ranked(y, score)
    total = r.sum()
    return float(r[: n_top(len(r), k)].sum() / total) if total else 0.0


def lift_at_k(y, score, k: float) -> float:
    """Precision@k divided by the overall conversion rate."""
    base = float(np.mean(y))
    return precision_at_k(y, score, k) / base if base else 0.0


def calls_per_enrolment(y, score, k: float) -> float:
    """Calls needed per enrolment when calling the top-k% (model ranking)."""
    p = precision_at_k(y, score, k)
    return 1 / p if p else float("inf")


def random_calls_per_enrolment(y) -> float:
    """Expected calls per enrolment when calling leads in random order."""
    base = float(np.mean(y))
    return 1 / base if base else float("inf")


def at_k_table(y, score, ks=(0.1, 0.2, 0.3, 0.4)) -> pd.DataFrame:
    """precision@k, recall@k, lift and calls/enrolment for several k."""
    rows = []
    for k in ks:
        rows.append({"top_k": f"{k:.0%}", "leads_called": n_top(len(y), k),
                     "precision": precision_at_k(y, score, k),
                     "recall": recall_at_k(y, score, k),
                     "lift": lift_at_k(y, score, k),
                     "calls_per_enrolment": calls_per_enrolment(y, score, k),
                     "random_calls_per_enrolment": random_calls_per_enrolment(y)})
    return pd.DataFrame(rows).set_index("top_k")


def lift_by_decile(y, score) -> pd.DataFrame:
    """Conversion rate and lift for each score decile (1 = highest scores)."""
    r = _ranked(y, score)
    n = len(r)
    base = r.mean()
    idx = np.arange(n)
    decile = np.floor(idx * 10 / n).astype(int) + 1
    df = pd.DataFrame({"decile": decile, "y": r})
    t = df.groupby("decile")["y"].agg(leads="size", conversions="sum", rate="mean")
    t["lift"] = t["rate"] / base if base else 0.0
    t["cum_recall"] = t["conversions"].cumsum() / r.sum() if r.sum() else 0.0
    return t


def gains_curve(y, score) -> pd.DataFrame:
    """Cumulative share of conversions captured vs share of leads called.

    Includes the origin (0, 0). The random baseline is the diagonal.
    """
    r = _ranked(y, score)
    n, total = len(r), r.sum()
    frac_called = np.arange(0, n + 1) / n
    captured = np.concatenate([[0.0], np.cumsum(r) / total if total else np.zeros(n)])
    return pd.DataFrame({"frac_called": frac_called, "frac_captured": captured,
                         "random": frac_called})
