"""Counsellor Copilot — who to call, when to stop, what to say.

Run:  streamlit run app/streamlit_app.py
Works with no ANTHROPIC_API_KEY (call briefs fall back to a template).
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import briefs as B  # noqa: E402
from src import economics as E  # noqa: E402
from src.clean import TARGET, load_raw  # noqa: E402
from src.config import load_config  # noqa: E402

st.set_page_config(page_title="Counsellor Copilot", page_icon="☎", layout="wide")

# Streamlit Cloud: pick up the key from secrets if one is configured.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file
    pass

cfg = load_config()

# ---------------------------------------------------------------- design tokens
PAPER, INK, MUTED = "#F3F6F4", "#14213D", "#5B6577"
RULE = "#D5DED9"
HOT, WARM, COLD = "#E09F3E", "#2A7F7A", "#94A3B8"
NOT_CALLED = "#E4EBE8"
BAND = {"Hot": HOT, "Warm": WARM, "Cold": COLD}
DISPLAY = "'Bricolage Grotesque', 'Source Sans 3', sans-serif"
BODY = "'Source Sans 3', 'Source Sans Pro', sans-serif"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Source+Sans+3:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame {{ font-family: {BODY}; }}
.block-container {{ padding-top: 3.4rem; max-width: 1240px; }}
h1, h2, h3, h4 {{ font-family: {DISPLAY} !important; color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: #E4EBE8; border-right: 1px solid {RULE}; }}
[data-testid="stSidebar"] h2 {{ font-size: 1.15rem; margin-bottom: .2rem; }}
[data-testid="stSidebar"] h3 {{ font-size: .98rem; margin: 1.1rem 0 .1rem; color: {MUTED}; }}

/* wordmark + hero */
.wordmark {{ font-family: {DISPLAY}; font-weight: 800; font-size: 1.05rem; color: {INK}; display: flex;
  align-items: center; gap: .5rem; margin-bottom: 1.6rem; }}
.wordmark .dot {{ width: .7rem; height: .7rem; border-radius: 50%; background: {HOT}; display: inline-block; }}
.wordmark .src {{ font-family: {BODY}; font-weight: 400; font-size: .9rem; color: {MUTED}; margin-left: auto; }}
[data-testid="stMarkdownContainer"] p.hero {{ font-family: {DISPLAY}; font-weight: 600; font-size: clamp(1.9rem, 3.2vw, 2.9rem);
  line-height: 1.12; color: {INK}; max-width: 22em; margin: 0 0 .5rem; letter-spacing: -0.02em; }}
.hero b {{ font-weight: 800; color: {INK}; background: linear-gradient(transparent 62%, rgba(224,159,62,.45) 62%); }}
[data-testid="stMarkdownContainer"] p.hero-sub {{ font-size: 1.18rem; color: {MUTED}; max-width: 46em; margin: 0 0 1.6rem; line-height: 1.45; }}
.hero-sub b {{ color: {INK}; }}

/* figures row: one strip, divided by rules — not cards */
.figs {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 2px solid {INK};
  border-bottom: 1px solid {RULE}; margin-bottom: 1.4rem; }}
.fig {{ padding: .8rem 1rem .9rem 0; }}
.fig + .fig {{ padding-left: 1rem; border-left: 1px solid {RULE}; }}
.fig .v {{ font-family: {DISPLAY}; font-size: 1.85rem; font-weight: 800; color: {INK};
  font-variant-numeric: tabular-nums; line-height: 1.1; }}
.fig .k {{ font-size: .95rem; color: {INK}; margin-top: .15rem; }}
.fig .c {{ font-size: .85rem; color: {MUTED}; }}
@media (max-width: 900px) {{ .figs {{ grid-template-columns: repeat(2, 1fr); }}
  .fig:nth-child(3) {{ border-left: none; padding-left: 0; }} }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {DISPLAY}; font-size: 1.08rem; font-weight: 600; padding: .5rem 0; }}
.stTabs [aria-selected="true"] {{ color: {INK} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {HOT}; height: 3px; }}

/* band bar */
.bandbar {{ display: flex; height: 2.1rem; border-radius: 4px; overflow: hidden; margin: .4rem 0 .35rem; }}
.bandbar div {{ display: flex; align-items: center; padding-left: .55rem; font-size: .88rem; font-weight: 600;
  color: {INK}; white-space: nowrap; overflow: hidden; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.2rem; font-size: .9rem; color: {MUTED}; margin-bottom: 1rem; }}
.sw {{ display: inline-block; width: .75rem; height: .75rem; border-radius: 2px; margin-right: .35rem;
  vertical-align: -1px; }}

/* call slip */
.slip {{ background: #FFFFFF; border: 1px solid {RULE}; border-left: 6px solid {HOT}; border-radius: 6px;
  padding: 1.1rem 1.4rem 1.2rem; max-width: 52rem; }}
.slip.warm {{ border-left-color: {WARM}; }} .slip.cold {{ border-left-color: {COLD}; }}
.slip h4 {{ margin: 0 0 .1rem; font-size: 1.35rem; }}
.slip .meta {{ color: {MUTED}; margin-bottom: .9rem; }}
.slip dl {{ display: grid; grid-template-columns: 6.5rem 1fr; gap: .55rem 1rem; margin: 0; }}
.slip dt {{ font-family: {DISPLAY}; font-weight: 600; color: {MUTED}; }}
.slip dd {{ margin: 0; font-size: 1.06rem; line-height: 1.45; color: {INK}; }}
.slip dd.opener {{ font-family: {DISPLAY}; font-size: 1.2rem; font-weight: 600; }}
.tag {{ display: inline-block; font-size: .85rem; padding: .1rem .55rem; border-radius: 999px;
  background: #E4EBE8; color: {INK}; margin-left: .4rem; vertical-align: 2px; }}
.tag.ai {{ background: #D7EDEB; }}

/* persona comparison table */
.ptable {{ width: 100%; border-collapse: collapse; border: none !important; }}
.ptable th, .ptable td {{ border: none !important; }}
.ptable th {{ text-align: left; font-weight: 600; color: {MUTED}; font-size: .9rem; padding: .4rem .6rem .5rem 0;
  border-bottom: 2px solid {INK} !important; }}
.ptable td {{ padding: .85rem .6rem .85rem 0; border-bottom: 1px solid {RULE} !important; vertical-align: middle; }}
.ptable .name {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1.12rem; color: {INK}; }}
.ptable .num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
.rate {{ position: relative; height: .9rem; background: #E4EBE8; border-radius: 2px; min-width: 9rem; }}
.rate span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px; }}
.rate i {{ position: absolute; top: -.3rem; bottom: -.3rem; width: 2px; background: {INK}; }}
[data-testid="stMarkdownContainer"] p.note {{ color: {MUTED}; font-size: .95rem; max-width: 60em; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    scorer = joblib.load(ROOT / "models" / "scorer.joblib")
    personas = joblib.load(ROOT / "models" / "personas.joblib")
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    return scorer, personas, card


@st.cache_data
def load_demo():
    """Held-out test leads (never used for training) with scores and outcomes."""
    scored = pd.read_csv(ROOT / "models" / "scored_leads.csv")
    try:
        raw = load_raw()
    except FileNotFoundError:
        return None, None
    from src.model import split
    # Same row order as the notebooks, so ties in score break identically.
    _, test_raw = split(raw, cfg)
    test = (scored[scored.split == "test"].drop(columns=["split"])
            .set_index("Prospect ID").loc[test_raw["Prospect ID"]].reset_index())
    return test, test_raw.reset_index(drop=True)


@st.cache_data
def score_upload(df: pd.DataFrame):
    scorer, personas, _ = load_models()
    s = scorer.score(df)
    out = df[[c for c in ["Prospect ID", "Lead Number", "Do Not Call"] if c in df]].copy()
    if "Prospect ID" not in out:
        out["Prospect ID"] = [f"upload-{i}" for i in range(len(df))]
    if "Lead Number" not in out:
        out["Lead Number"] = np.arange(1, len(df) + 1)
    out = out.join(s[["probability", "score", "band", "drivers_text"]])
    out["drivers"] = s["drivers"].apply(json.dumps)
    out["persona"] = personas.assign(df).to_numpy()
    if TARGET in df:
        out[TARGET] = df[TARGET].to_numpy()
    return out.reset_index(drop=True)


def plot_style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, sans-serif", size=14, color=INK),
        margin=dict(l=10, r=10, t=56, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=14), entrywidth=150, entrywidthmode="pixels"),
        hoverlabel=dict(bgcolor="white", font_size=13))
    fig.update_xaxes(gridcolor=RULE, zeroline=False, linecolor=INK, ticks="outside")
    fig.update_yaxes(gridcolor=RULE, zeroline=False)
    return fig


scorer, personas_model, card = load_models()

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## Your team and prices")
st.sidebar.caption("These are assumptions, not measurements. Every number on the page "
                   "recalculates when you change them.")
e, c = cfg["economics"], cfg["capacity"]
st.sidebar.markdown("### Calling capacity")
counsellors = st.sidebar.slider("Counsellors", 1, 30, int(c["counsellors"]))
cpd = st.sidebar.slider("Calls per counsellor per day", 10, 100, int(c["calls_per_counsellor_per_day"]), step=5)
days = st.sidebar.slider("Days to work a batch of leads", 1, 20, int(c["days_to_work_batch"]))
st.sidebar.markdown("### Money")
price = st.sidebar.slider("Course price (₹)", 2_000, 50_000, int(e["course_price"]), step=1_000)
margin = st.sidebar.slider("Gross margin", 0.10, 0.90, float(e["gross_margin"]), step=0.05)
cost = st.sidebar.slider("Cost per call (₹)", 10, 500, int(e["cost_per_call"]), step=10)
A = E.Assumptions(course_price=price, gross_margin=margin, cost_per_call=cost,
                  counsellors=counsellors, calls_per_counsellor_per_day=cpd,
                  days_to_work_batch=days, currency=e["currency"])
st.sidebar.markdown(
    f"Each enrolment is worth **₹{A.value_per_enrolment:,.0f}** after delivery costs. "
    f"A call pays for itself when a lead has more than a **{A.break_even_probability:.1%}** chance "
    f"of enrolling. Your team can make **{A.calls_per_day:,}** calls a day, "
    f"**{A.batch_capacity:,}** per batch.")

# ---------------------------------------------------------------- data source
demo, raw_demo = load_demo()
with st.sidebar.expander("Score your own leads"):
    upload = st.file_uploader("CSV with the same columns as Leads.csv", type="csv",
                              help="Lead Origin, Lead Source, TotalVisits, Total Time Spent on Website, "
                                   "occupation and so on. A Converted column is optional.")
if upload is not None:
    raw_df = pd.read_csv(upload, na_values=["Select"])
    try:
        leads = score_upload(raw_df)
    except ValueError as err:
        st.error(f"This file can't be scored: {err}. Add the missing columns and upload it again.")
        st.stop()
    raw_leads = raw_df.reset_index(drop=True)
    source_note = f"{len(leads):,} uploaded leads"
elif demo is not None:
    leads, raw_leads = demo, raw_demo
    source_note = f"{len(leads):,} past leads the model never trained on"
else:
    st.warning("No demo leads: data/Leads.csv is missing. Open “Score your own leads” in the sidebar "
               "and upload a CSV.")
    st.stop()

has_outcomes = TARGET in leads.columns and leads[TARGET].notna().all()
callable_mask = leads.get("Do Not Call", pd.Series("No", index=leads.index)).astype(str) != "Yes"
dnc_count = int((~callable_mask).sum())

order = leads[callable_mask].sort_values("probability", ascending=False, kind="mergesort")
p_sorted = order["probability"].to_numpy()
n_ev = E.optimal_calls(p_sorted, A)
n_cut = E.optimal_calls(p_sorted, A, capacity=A.batch_capacity)

y_sorted = order[TARGET].to_numpy(dtype=float) if has_outcomes else p_sorted
n = len(y_sorted)
total = y_sorted.sum()
cum = np.concatenate([[0], np.cumsum(y_sorted)])
rate = total / n
enrol_cut = cum[n_cut]
capture = enrol_cut / total
calls_per_enrol = n_cut / enrol_cut if enrol_cut else float("inf")
random_enrol = n_cut * rate
profit = enrol_cut * A.value_per_enrolment - n_cut * A.cost_per_call
random_profit = random_enrol * A.value_per_enrolment - n_cut * A.cost_per_call
limit = "your team's capacity" if n_cut < n_ev else "break-even (further calls lose money)"

# ---------------------------------------------------------------- header
st.markdown(f'<div class="wordmark"><span class="dot"></span>Counsellor Copilot'
            f'<span class="src">{html.escape(source_note)}</span></div>', unsafe_allow_html=True)
verb = "reached" if has_outcomes else "would reach"
st.markdown(
    f'<p class="hero">This batch, call the top <b>{n_cut:,}</b> of {n:,} leads and stop there.</p>'
    f'<p class="hero-sub">In score order that {verb} <b>{capture:.0%} of enrolments</b>, about '
    f'<b>{enrol_cut - random_enrol:,.0f} more</b> than working the same list in random order. '
    f'The cutoff is set by {limit}.</p>', unsafe_allow_html=True)
st.markdown(f"""
<div class="figs">
  <div class="fig"><div class="v">{enrol_cut:,.0f}</div><div class="k">enrolments reached</div>
    <div class="c">random order: {random_enrol:,.0f}</div></div>
  <div class="fig"><div class="v">{calls_per_enrol:.2f}</div><div class="k">calls per enrolment</div>
    <div class="c">random order: {1 / rate:.2f}</div></div>
  <div class="fig"><div class="v">₹{profit / 1e5:,.1f} lakh</div><div class="k">projected profit</div>
    <div class="c">₹{(profit - random_profit) / 1e5:,.1f} lakh more than random</div></div>
  <div class="fig"><div class="v">{A.break_even_probability:.1%}</div><div class="k">break-even chance to enrol</div>
    <div class="c">₹{A.cost_per_call:,.0f} call vs ₹{A.value_per_enrolment:,.0f} enrolment</div></div>
</div>""", unsafe_allow_html=True)

basis = ("Replayed on these leads' recorded outcomes." if has_outcomes
         else "Expected values from calibrated scores (this file has no outcomes).")

TAB_NAMES = ["Who to call", "When to stop", "What to say", "Who they are"]
# Deep link: ?tab=When%20to%20stop opens that tab first (handy for sharing and screenshots).
first = st.query_params.get("tab")
ordered = ([first] if first in TAB_NAMES else []) + [t for t in TAB_NAMES if t != first]
tabs = dict(zip(ordered, st.tabs(ordered)))

# ---------------------------------------------------------------- who to call
with tabs["Who to call"]:
    in_cut = order.head(n_cut)["band"].value_counts()
    seg = [(b, int(in_cut.get(b, 0)), BAND[b]) for b in ("Hot", "Warm", "Cold")]
    seg.append(("Not called", n - n_cut, NOT_CALLED))
    bar = "".join(f'<div style="width:{k / n * 100:.3f}%;background:{col}" title="{name}: {k:,}">'
                  f'{name} {k:,}</div>' for name, k, col in seg if k)
    st.markdown(f"#### The {n_cut:,} calls in this batch", unsafe_allow_html=True)
    st.markdown(f'<div class="bandbar">{bar}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="legend"><span><span class="sw" style="background:{HOT}"></span>Hot: score '
        f'{scorer.cutoffs["hot"]:.0f}+, the top 20% of leads</span>'
        f'<span><span class="sw" style="background:{WARM}"></span>Warm: {scorer.cutoffs["warm"]:.0f} to '
        f'{scorer.cutoffs["hot"]:.0f}, the next 30%</span>'
        f'<span><span class="sw" style="background:{COLD}"></span>Cold: below {scorer.cutoffs["warm"]:.0f}</span>'
        + (f'<span>{dnc_count} Do Not Call lead(s) left out</span>' if dnc_count else "")
        + '</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 1])
    with left:
        bands = st.segmented_control("Show", ["Hot", "Warm", "Cold"], default=["Hot", "Warm", "Cold"],
                                     selection_mode="multi")
    view = order.copy()
    view.insert(0, "Rank", np.arange(1, len(view) + 1))
    view["Call"] = view["Rank"] <= n_cut
    view = view[view["band"].isin(bands or [])]
    export = order.assign(rank=np.arange(1, len(order) + 1), call=lambda d: d["rank"] <= n_cut)[
        ["rank", "call", "Prospect ID", "Lead Number", "score", "band", "persona", "drivers_text"]]
    with right:
        st.download_button("Download call list", export.to_csv(index=False).encode("utf-8"),
                           "call_list.csv", "text/csv", use_container_width=True)

    table = view[["Rank", "Call", "Lead Number", "score", "band", "persona", "drivers_text"]].rename(
        columns={"score": "Score", "band": "Band", "persona": "Lead group",
                 "drivers_text": "Why this score (↑ raises, ↓ lowers)", "Lead Number": "Lead"})
    styled = table.style.map(lambda b: f"background-color:{BAND[b]}33; font-weight:600", subset=["Band"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=470,
                 column_config={
                     "Rank": st.column_config.NumberColumn(width="small"),
                     "Call": st.column_config.CheckboxColumn("In batch", width="small"),
                     "Lead": st.column_config.NumberColumn(format="%d", width="small"),
                     "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f",
                                                              width="small"),
                     "Why this score (↑ raises, ↓ lowers)": st.column_config.TextColumn(width="large")})
    st.markdown('<p class="note">Score is the calibrated chance of enrolling, out of 100. '
                'Leads are ranked highest first; ties keep their original order.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- when to stop
with tabs["When to stop"]:
    st.markdown("#### Enrolments reached as you work down the list")
    x = np.arange(n + 1) / n * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random order", hoverinfo="skip",
                             line=dict(color=COLD, dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=x, y=cum / total * 100, name="Score order", line=dict(color=WARM, width=3.5),
                             hovertemplate="Call %{x:.0f}% of leads → %{y:.0f}% of enrolments<extra></extra>"))
    cx = n_cut / n * 100
    fig.add_shape(type="rect", x0=0, x1=cx, y0=0, y1=100, fillcolor=HOT, opacity=.10, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[cx], y=[capture * 100], mode="markers+text", showlegend=False,
                             marker=dict(size=12, color=HOT, line=dict(color=INK, width=1.5)),
                             text=[f"  stop at {n_cut:,} calls: {capture:.0%}"], textposition="middle right",
                             textfont=dict(family="Bricolage Grotesque", size=15, color=INK), hoverinfo="skip"))
    if n_ev != n_cut:
        fig.add_vline(x=n_ev / n * 100, line=dict(color=MUTED, dash="dot", width=1.5),
                      annotation_text=f"break-even would allow {n_ev:,}", annotation_position="bottom left",
                      annotation_font_color=MUTED)
    fig.update_xaxes(title="% of leads called, highest score first", range=[0, 100], ticksuffix="%")
    fig.update_yaxes(title="% of enrolments reached", range=[0, 102], ticksuffix="%")
    st.plotly_chart(plot_style(fig, 440), use_container_width=True)

    st.markdown("#### Profit after each call, at your prices")
    k = np.arange(n + 1)
    prof = cum * A.value_per_enrolment - k * A.cost_per_call
    rprof = k * rate * A.value_per_enrolment - k * A.cost_per_call
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=k, y=rprof / 1e5, name="Random order", line=dict(color=COLD, dash="dash", width=2),
                              hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=k, y=prof / 1e5, name="Score order", line=dict(color=WARM, width=3.5),
                              hovertemplate="%{x:,} calls → ₹%{y:.1f} lakh<extra></extra>"))
    fig2.add_vline(x=n_cut, line=dict(color=HOT, width=2))
    fig2.update_xaxes(title="Leads called")
    fig2.update_yaxes(title="₹ lakh")
    st.plotly_chart(plot_style(fig2, 340), use_container_width=True)

    extra_calls = A.calls_per_counsellor_per_day * A.days_to_work_batch
    nxt = cum[min(n_cut + extra_calls, n)] - enrol_cut
    st.markdown(
        f'<p class="note"><b>Should you add a counsellor?</b> One more counsellor makes {extra_calls:,} extra calls '
        f'per batch. The next {min(extra_calls, n - n_cut):,} leads below today\'s cutoff would bring about '
        f'<b>{nxt:,.0f} more enrolments</b> (₹{(nxt * A.value_per_enrolment - extra_calls * A.cost_per_call) / 1e5:,.1f} '
        f'lakh before salary). {basis} This is a projection from past leads, not a measured uplift. '
        f'{html.escape(A.label())}.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- what to say
with tabs["What to say"]:
    client = B.get_client()
    choices = order.head(max(n_cut, 50))
    labels = [f"{i + 1}. Lead {r['Lead Number']}, score {r['score']:.0f} ({r['band']}), {r['persona']}"
              for i, (_, r) in enumerate(choices.iterrows())]
    pick = st.selectbox("Choose a lead from the call list", range(len(labels)), format_func=lambda i: labels[i])
    row = choices.iloc[pick]
    raw_row = raw_leads.loc[raw_leads["Prospect ID"] == row["Prospect ID"]].iloc[0] \
        if "Prospect ID" in raw_leads else raw_leads.iloc[row.name]
    inp = B.lead_facts(raw_row, json.loads(row["drivers"]), row["score"], row["band"], row["persona"])
    if st.session_state.get("brief_for") != row["Prospect ID"]:
        with st.spinner("Writing the brief"):
            st.session_state["brief"] = B.generate_brief(str(row["Prospect ID"]), inp, client, cfg)
            st.session_state["brief_for"] = row["Prospect ID"]
    res = st.session_state["brief"]
    parts = {ln.split(":", 1)[0].strip(): ln.split(":", 1)[1].strip()
             for ln in res["text"].splitlines() if ":" in ln}
    is_ai = res["source"].startswith("llm")
    tag = (f'<span class="tag ai">Written by {html.escape(str(res["model"]))}, checked against the lead\'s data</span>'
           if is_ai else '<span class="tag">Template: no AI key set</span>')
    st.markdown(f"""
<div class="slip {row['band'].lower()}">
  <h4>Lead {row['Lead Number']}{tag}</h4>
  <div class="meta">Score {row['score']:.0f} out of 100, {row['band']}, {html.escape(row['persona'])}</div>
  <dl>
    <dt>Why</dt><dd>{html.escape(parts.get('Why', ''))}</dd>
    <dt>Objection</dt><dd>{html.escape(parts.get('Objection', ''))}</dd>
    <dt>Opener</dt><dd class="opener">{html.escape(parts.get('Opener', ''))}</dd>
  </dl>
</div>""", unsafe_allow_html=True)
    if is_ai and st.button("Write it again"):
        st.session_state.pop("brief_for", None)
        st.rerun()
    if res["notes"]:
        with st.expander("Why the AI draft was rejected"):
            st.write(res["notes"])
    with st.expander("Everything the brief was allowed to use"):
        st.json(inp)
    st.markdown('<p class="note">A brief helps a counsellor prepare. It is not a fact about the lead: '
                'check the lead record before the call.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- who they are
with tabs["Who they are"]:
    pt = pd.read_csv(ROOT / "reports" / "personas.csv").set_index("persona")
    base = pt["conversions"].sum() / pt["leads"].sum()
    counts = leads.loc[callable_mask, "persona"].value_counts()
    rows = ""
    for name, r in pt.iterrows():
        col = HOT if r["lift"] >= 1.5 else WARM if r["lift"] >= 0.8 else COLD
        rows += (f'<tr><td class="name">{html.escape(name)}</td>'
                 f'<td><div class="rate"><span style="width:{r["conv_rate"] * 100:.1f}%;background:{col}"></span>'
                 f'<i style="left:{base * 100:.1f}%"></i></div></td>'
                 f'<td class="num"><b>{r["conv_rate"]:.0%}</b> enrol</td>'
                 f'<td class="num">{r["share_of_leads"]:.0%} of leads<br>{r["share_of_conversions"]:.0%} of enrolments</td>'
                 f'<td class="num">{int(counts.get(name, 0)):,} here</td>'
                 f'<td>{html.escape(r["playbook"])}</td></tr>')
    st.markdown("#### Five kinds of lead, and how to treat each")
    st.markdown(f'<table class="ptable"><tr><th>Lead group</th><th colspan="2">Enrol rate (line marks the {base:.0%} average)</th>'
                f'<th>Size</th><th>In this list</th><th>Playbook</th></tr>{rows}</table>',
                unsafe_allow_html=True)
    st.markdown('<p class="note" style="margin-top:1rem">Rates measured on past leads the model never trained on. '
                'Groups describe who the leads are; within every group, the score still decides who is called '
                'first. Skipping a whole group cost enrolments when we tested it.</p>', unsafe_allow_html=True)
