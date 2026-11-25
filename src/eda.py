"""Funnel EDA helpers: segment tables, binned rates, missingness, and charts.

Chart titles are built from the computed tables (never typed by hand), so every
number in a title is traceable to the data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from src.clean import OTHER, merge_rare  # noqa: E402

MISSING_LABEL = "Missing"
INK = "#1f2933"
MUTED = "#9aa5b1"
ACCENT = "#d64545"
BASE = "#3e7cb1"


# ---------------------------------------------------------------- tables
def baseline_rate(df: pd.DataFrame, target: str = "Converted") -> float:
    return float(df[target].mean())


def segment_table(df: pd.DataFrame, col: str, target: str = "Converted",
                  min_share: float = 0.01) -> pd.DataFrame:
    """Volume, share, conversion rate and lift per category of `col`.

    Rare categories (< min_share of rows) are merged into "Other"; missing
    values are shown as their own "Missing" segment.
    """
    seg = merge_rare(df[col], min_share=min_share).fillna(MISSING_LABEL)
    base = df[target].mean()
    t = (df.assign(_seg=seg).groupby("_seg")[target]
         .agg(leads="size", conversions="sum", rate="mean"))
    t["share"] = t["leads"] / len(df)
    t["lift"] = t["rate"] / base
    t.index.name = col
    return t.sort_values("leads", ascending=False)


def binned_table(df: pd.DataFrame, col: str, bins: list[float],
                 labels: list[str], target: str = "Converted") -> pd.DataFrame:
    """Conversion by fixed bins of a numeric column (missing kept separate)."""
    b = pd.cut(df[col], bins=bins, labels=labels, right=False)
    b = b.cat.add_categories([MISSING_LABEL]).fillna(MISSING_LABEL)
    base = df[target].mean()
    t = (df.assign(_bin=b).groupby("_bin", observed=False)[target]
         .agg(leads="size", conversions="sum", rate="mean"))
    t = t[t["leads"] > 0]
    t["share"] = t["leads"] / len(df)
    t["lift"] = t["rate"] / base
    t.index.name = col
    return t


def missingness_table(df: pd.DataFrame, target: str = "Converted") -> pd.DataFrame:
    """% missing per column and whether being missing relates to conversion.

    For each column with any missing values: conversion rate when missing vs
    present, the gap in percentage points, and a chi-square p-value.
    """
    rows = []
    for c in df.columns:
        if c == target:
            continue
        miss = df[c].isna()
        pct = miss.mean()
        r = {"column": c, "pct_missing": pct,
             "rate_if_missing": np.nan, "rate_if_present": np.nan,
             "gap_pp": np.nan, "p_value": np.nan}
        if 0 < miss.sum() < len(df):
            rm = df.loc[miss, target].mean()
            rp = df.loc[~miss, target].mean()
            ct = pd.crosstab(miss, df[target])
            p = chi2_contingency(ct)[1] if ct.shape == (2, 2) else np.nan
            r.update(rate_if_missing=rm, rate_if_present=rp,
                     gap_pp=(rm - rp) * 100, p_value=p)
        rows.append(r)
    return (pd.DataFrame(rows).set_index("column")
            .sort_values("pct_missing", ascending=False))


def quadrant_table(t: pd.DataFrame) -> pd.DataFrame:
    """Label named segments by volume/rate quadrant using median splits."""
    q = t[~t.index.isin([OTHER, MISSING_LABEL])].copy()
    hv = q["leads"] >= q["leads"].median()
    hr = q["rate"] >= q["rate"].median()
    q["quadrant"] = np.select(
        [hv & hr, ~hv & hr, hv & ~hr], ["scale", "grow", "fix or filter"],
        default="deprioritise")
    return q


def mix_within(df: pd.DataFrame, mask: pd.Series, by: str,
               target: str = "Converted") -> pd.DataFrame:
    """Composition and conversion of a subset (`mask`) broken down by `by`."""
    sub = df[mask]
    t = sub.groupby(by)[target].agg(leads="size", rate="mean")
    t["share_of_subset"] = t["leads"] / len(sub)
    return t.sort_values("leads", ascending=False)


# ---------------------------------------------------------------- headlines
def _named(t: pd.DataFrame, min_share: float) -> pd.DataFrame:
    """Segments that are real categories with enough volume to act on."""
    return t[(~t.index.isin([OTHER, MISSING_LABEL])) & (t["share"] >= min_share)]


def headline_best_vs_biggest(t: pd.DataFrame, noun: str,
                             min_share: float = 0.02) -> str:
    """e.g. "Referral leads convert at 2.1× baseline but are 4% of volume"."""
    named = _named(t, min_share)
    best = named["lift"].idxmax()
    big = t.index[0]
    b = named.loc[best]
    if best == big:
        return (f"{best} {noun} are the largest group ({b.share:.0%} of volume) "
                f"and convert at {b.lift:.1f}× baseline")
    g = t.loc[big]
    return (f"{best} {noun} convert at {b.lift:.1f}× baseline but are "
            f"{b.share:.0%} of volume; the largest group, {big} "
            f"({g.share:.0%}), converts at {g.lift:.1f}×")


def headline_missing(t: pd.DataFrame, noun: str) -> str | None:
    """Headline when the 'Missing' segment is large and converts differently."""
    if MISSING_LABEL not in t.index:
        return None
    m = t.loc[MISSING_LABEL]
    return (f"{m.share:.0%} of leads left {noun} blank; they convert at "
            f"{m.lift:.1f}× baseline")


# ---------------------------------------------------------------- charts
def _save(fig, path: Path | str | None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def _wrap(title: str, width: int = 70) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(title, width))


def plot_segments(t: pd.DataFrame, baseline: float, title: str,
                  path: Path | str | None = None, top: int = 12):
    """Horizontal bars of conversion rate, annotated with volume share."""
    t = t.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    colors = [ACCENT if r > baseline else BASE for r in t["rate"]]
    ax.barh(t.index.astype(str), t["rate"] * 100, color=colors)
    ax.axvline(baseline * 100, color=INK, ls="--", lw=1)
    ax.set_ylim(-1.0, len(t) - 0.5)
    ax.text(baseline * 100, -0.8, f" baseline {baseline:.1%}",
            color=INK, fontsize=8, va="center")
    for i, (rate, share, n) in enumerate(zip(t["rate"], t["share"], t["leads"])):
        ax.text(rate * 100 + 0.8, i, f"{rate:.0%}  ·  {share:.0%} of leads (n={n:,})",
                va="center", fontsize=8, color=INK)
    ax.set_xlabel("Conversion rate (%)")
    ax.set_xlim(0, max(100, t["rate"].max() * 100 + 30))
    ax.set_title(_wrap(title), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_quadrant(t: pd.DataFrame, baseline: float, title: str,
                  path: Path | str | None = None):
    """Volume × conversion-rate scatter with median split lines."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x, y = t["leads"], t["rate"] * 100
    mx, my = x.median(), y.median()
    ax.scatter(x, y, s=60 + 400 * t["share"], color=BASE, alpha=0.75,
               edgecolor="white")
    for name, xi, yi in zip(t.index, x, y):
        ax.annotate(str(name), (xi, yi), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.axvline(mx, color=MUTED, ls="--", lw=1)
    ax.axhline(my, color=MUTED, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Lead volume (log scale)")
    ax.set_ylabel("Conversion rate (%)")
    # Quadrant labels hug the median lines at the plot's top/bottom edge,
    # where there are no points.
    ax.set_ylim(0, 112)
    kw = dict(fontsize=8, color=MUTED)
    ax.text(mx * 1.06, 110, "high volume · high rate → protect & scale",
            ha="left", va="top", **kw)
    ax.text(mx / 1.06, 110, "low volume · high rate → grow",
            ha="right", va="top", **kw)
    ax.text(mx * 1.06, 2, "high volume · low rate → fix or filter",
            ha="left", va="bottom", **kw)
    ax.text(mx / 1.06, 2, "low volume · low rate → deprioritise",
            ha="right", va="bottom", **kw)
    ax.set_title(_wrap(title), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_binned(t: pd.DataFrame, baseline: float, title: str, xlabel: str,
                path: Path | str | None = None, note: str | None = None):
    """Column chart of conversion rate per bin with volume labels."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(t))
    colors = [ACCENT if r > baseline else BASE for r in t["rate"]]
    ax.bar(xs, t["rate"] * 100, color=colors)
    ax.axhline(baseline * 100, color=INK, ls="--", lw=1)
    ax.text(len(t) - 0.5, baseline * 100, f"baseline {baseline:.1%}",
            ha="right", va="bottom", fontsize=8, color=INK)
    for xi, r, s in zip(xs, t["rate"], t["share"]):
        ax.text(xi, r * 100 + 1, f"{r:.0%}\n{s:.0%} of leads", ha="center",
                fontsize=8, color=INK)
    ax.set_xticks(xs, t.index.astype(str))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Conversion rate (%)")
    ax.set_ylim(0, min(100, t["rate"].max() * 100 + 15))
    ax.set_title(_wrap(title), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        fig.text(0.01, -0.02, _wrap(note, 110), fontsize=8, color=INK,
                 ha="left", va="top", style="italic")
    fig.tight_layout()
    return _save(fig, path)


def plot_leakage_gap(res: pd.DataFrame, title: str,
                     path: Path | str | None = None):
    """Side-by-side bars of AUC and precision@20% for honest vs leaky models.

    `res` index = model names, columns = metric names (values in 0-1).
    """
    metrics_ = list(res.columns)
    fig, axes = plt.subplots(1, len(metrics_), figsize=(8, 3.8))
    for ax, m in zip(np.atleast_1d(axes), metrics_):
        vals = res[m]
        colors = [MUTED if "leak" in str(i).lower() else BASE for i in res.index]
        ax.bar(range(len(vals)), vals * 100, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v * 100 + 1.5, f"{v:.3f}" if "AUC" in m else f"{v:.1%}",
                    ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(len(vals)), [str(i) for i in res.index], fontsize=8)
        ax.set_ylim(0, 110)
        ax.set_title(m, fontsize=10, color=INK)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(_wrap(title, 80), x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def plot_gains(g: pd.DataFrame, title: str, mark_k: float | None = None,
               path: Path | str | None = None):
    """Cumulative gains: share of conversions captured vs share of leads called."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(g["frac_called"] * 100, g["frac_captured"] * 100, color=ACCENT, lw=2,
            label="Call in model order")
    ax.plot(g["frac_called"] * 100, g["random"] * 100, color=MUTED, ls="--",
            label="Call in random order")
    if mark_k is not None:
        i = int(np.ceil(mark_k * (len(g) - 1)))
        x, y = g["frac_called"].iloc[i] * 100, g["frac_captured"].iloc[i] * 100
        ax.scatter([x], [y], color=ACCENT, zorder=3)
        ax.annotate(f"call top {x:.0f}% → {y:.0f}% of enrolments",
                    (x, y), xytext=(12, -18), textcoords="offset points",
                    fontsize=9, color=INK)
        ax.scatter([x], [x], color=MUTED, zorder=3)
        ax.annotate(f"random: {x:.0f}%", (x, x), xytext=(12, -6),
                    textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("% of leads called (highest score first)")
    ax.set_ylabel("% of all enrolments captured")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(_wrap(title, 65), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_reliability(tables: dict[str, pd.DataFrame], title: str,
                     path: Path | str | None = None):
    """Mean predicted vs observed conversion per decile, one line per model."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 100], [0, 100], color=MUTED, ls="--", lw=1, label="perfectly calibrated")
    palette = [ACCENT, BASE, INK]
    for (name, t), c in zip(tables.items(), palette):
        ax.plot(t["mean_predicted"] * 100, t["observed"] * 100, marker="o",
                color=c, label=name)
    ax.set_xlabel("Predicted conversion probability (%)")
    ax.set_ylabel("Actual conversion rate (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title(_wrap(title, 60), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_drivers(t: pd.DataFrame, title: str, path: Path | str | None = None):
    """Global drivers: mean |SHAP| bars with plain-English labels.

    `t` columns: label, importance, direction ('raises' / 'lowers').
    """
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.4))
    colors = [ACCENT if d == "raises" else BASE for d in t["direction"]]
    ax.barh(t["label"], t["importance"], color=colors)
    for i, (v, d) in enumerate(zip(t["importance"], t["direction"])):
        ax.text(v, i, f"  {'↑ raises' if d == 'raises' else '↓ lowers'} odds",
                va="center", fontsize=8, color=INK)
    ax.set_xlabel("Average impact on a lead's score (mean |SHAP|, log-odds)")
    ax.set_xlim(0, t["importance"].max() * 1.35)
    ax.set_title(_wrap(title, 70), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_profit_curve(curve: pd.DataFrame, title: str, markers: dict | None = None,
                      currency: str = "₹", note: str | None = None,
                      path: Path | str | None = None):
    """Cumulative profit vs number of calls, model order vs random order.

    `markers`: {label: n_calls} vertical markers (e.g. EV-optimal, capacity).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    x = curve["calls"]
    ax.plot(x, curve["profit"] / 1e5, color=ACCENT, lw=2, label="Call in score order")
    ax.plot(x, curve["random_profit"] / 1e5, color=MUTED, ls="--", label="Call in random order")
    for (lab, n), ls in zip((markers or {}).items(), [":", "-."]):
        ax.axvline(n, color=INK, ls=ls, lw=1)
        yv = curve["profit"].iloc[n - 1] / 1e5
        ax.annotate(f"{lab}\n{n:,} calls · {currency}{yv:.1f} lakh",
                    (n, yv), xytext=(-8, -38), textcoords="offset points", ha="right",
                    fontsize=8, color=INK)
    ax.set_xlabel("Number of leads called")
    ax.set_ylabel(f"Cumulative profit ({currency} lakh)")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(_wrap(title, 75), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    if note:
        fig.text(0.01, -0.02, _wrap(note, 115), fontsize=8, color=INK, ha="left",
                 va="top", style="italic")
    fig.tight_layout()
    return _save(fig, path)


def plot_personas(t: pd.DataFrame, baseline: float, title: str,
                  path: Path | str | None = None):
    """Persona bars: conversion rate, annotated with share of leads / conversions."""
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(t) + 1.5))
    colors = [ACCENT if r > baseline else BASE for r in t["conv_rate"]]
    ax.barh(t.index, t["conv_rate"] * 100, color=colors)
    ax.axvline(baseline * 100, color=INK, ls="--", lw=1)
    for i, (r, sl, sc) in enumerate(zip(t["conv_rate"], t["share_of_leads"],
                                        t["share_of_conversions"])):
        ax.text(r * 100 + 1, i, f"{r:.0%} convert · {sl:.0%} of leads · {sc:.0%} of enrolments",
                va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 150)
    ax.set_xlabel(f"Conversion rate (%) — dashed line = baseline {baseline:.1%}")
    ax.set_title(_wrap(title, 75), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_missingness(m: pd.DataFrame, title: str,
                     path: Path | str | None = None, top: int = 15,
                     leaky: list[str] | None = None):
    """% missing per column, coloured by whether missing leads convert differently.

    Columns in `leaky` (post-contact) are labelled and drawn hatched, since
    they will not be used by the model.
    """
    leaky = set(leaky or [])
    m = m[m["pct_missing"] > 0].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(m) + 1.4))
    colors = [ACCENT if (p < 0.01 and abs(g) >= 5) else MUTED
              for p, g in zip(m["p_value"], m["gap_pp"])]
    labels = [f"{c}  [post-contact]" if c in leaky else c for c in m.index]
    bars = ax.barh(labels, m["pct_missing"] * 100, color=colors)
    for bar, c in zip(bars, m.index):
        if c in leaky:
            bar.set_alpha(0.35)
            bar.set_hatch("//")
    for i, (pct, g) in enumerate(zip(m["pct_missing"], m["gap_pp"])):
        ax.text(pct * 100 + 0.8, i, f"{pct:.0%} missing · {g:+.0f} pp conv. when missing",
                va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of leads with the field missing  "
                  "(red = missing leads convert ≥5 pp differently, p<0.01)")
    ax.set_title(_wrap(title), loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, path)
