"""Counsellor Copilot — who to call, when to stop, what to say.

Run:  streamlit run app/streamlit_app.py
Works with no ANTHROPIC_API_KEY (call briefs fall back to a template).
"""
from __future__ import annotations

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

st.set_page_config(page_title="Counsellor Copilot", page_icon="📞", layout="wide")

# Streamlit Cloud: pick up the key from secrets if one is configured.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file
    pass

cfg = load_config()
RED, BLUE, GREY ="#d64545", "#3e7cb1", "#9aa5b1"
BAND_COLORS = {"Hot": "#fde2e2", "Warm": "#fff4d6", "Cold": "#eef2f6"}


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


scorer, personas_model, card = load_models()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Assumptions")
st.sidebar.caption("All money and capacity figures below are **assumptions**. Change them; "
                   "every number in the app updates and is labelled with them.")
e, c = cfg["economics"], cfg["capacity"]
counsellors = st.sidebar.slider("Counsellors", 1, 30, int(c["counsellors"]))
cpd = st.sidebar.slider("Calls per counsellor per day", 10, 100, int(c["calls_per_counsellor_per_day"]), step=5)
days = st.sidebar.slider("Days to work a batch of leads", 1, 20, int(c["days_to_work_batch"]))
price = st.sidebar.slider("Course price (₹)", 2_000, 50_000, int(e["course_price"]), step=1_000)
margin = st.sidebar.slider("Gross margin", 0.10, 0.90, float(e["gross_margin"]), step=0.05)
cost = st.sidebar.slider("Cost per call (₹)", 10, 500, int(e["cost_per_call"]), step=10)
A = E.Assumptions(course_price=price, gross_margin=margin, cost_per_call=cost,
                  counsellors=counsellors, calls_per_counsellor_per_day=cpd,
                  days_to_work_batch=days, currency=e["currency"])
st.sidebar.markdown(f"**Value per enrolment:** ₹{A.value_per_enrolment:,.0f}  \n"
                    f"**Break-even P(convert):** {A.break_even_probability:.1%}  \n"
                    f"**Capacity:** {A.calls_per_day:,} calls/day · {A.batch_capacity:,} per batch")

# ---------------------------------------------------------------- header + data source
st.title("📞 Counsellor Copilot")
st.markdown("**Counsellors can't call every lead. Who to call, when to stop, what to say.**")

demo, raw_demo = load_demo()
upload = st.file_uploader("Score new leads: upload a CSV with the same columns as Leads.csv",
                          type="csv", help="Columns like Lead Origin, Lead Source, TotalVisits, "
                          "Total Time Spent on Website, occupation … The Converted column is optional.")
if upload is not None:
    raw_df = pd.read_csv(upload, na_values=["Select"])
    try:
        leads = score_upload(raw_df)
    except ValueError as err:
        st.error(f"Couldn't score this file: {err}")
        st.stop()
    raw_leads = raw_df.reset_index(drop=True)
    source_note = f"Uploaded file: {len(leads):,} leads"
elif demo is not None:
    leads, raw_leads = demo, raw_demo
    source_note = (f"Demo: {len(leads):,} held-out test leads (never seen in training), "
                   f"with their recorded outcomes")
else:
    st.warning("data/Leads.csv not found, so there are no demo leads. Upload a CSV to score leads.")
    st.stop()

has_outcomes = TARGET in leads.columns and leads[TARGET].notna().all()
callable_mask = leads.get("Do Not Call", pd.Series("No", index=leads.index)).astype(str) != "Yes"
dnc_count = int((~callable_mask).sum())
st.caption(f"{source_note} · {dnc_count} Do-Not-Call lead(s) excluded from call lists · "
           f"Model: {card['model']} ({card['calibration']} calibrated) · {A.label()}")

order = leads[callable_mask].sort_values("probability", ascending=False, kind="mergesort")
p_sorted = order["probability"].to_numpy()
n_ev = E.optimal_calls(p_sorted, A)
n_cut = E.optimal_calls(p_sorted, A, capacity=A.batch_capacity)

TAB_NAMES = ["Who to call", "When to stop", "What to say", "Personas"]
# Deep link: ?tab=When%20to%20stop opens that tab first (handy for sharing and screenshots).
first = st.query_params.get("tab")
ordered = ([first] if first in TAB_NAMES else []) + [t for t in TAB_NAMES if t != first]
tabs = dict(zip(ordered, st.tabs(ordered)))
tab1, tab2, tab3, tab4 = (tabs[t] for t in TAB_NAMES)

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Ranked call list")
    m1, m2, m3 = st.columns(3)
    m1.metric("Leads to call this batch", f"{n_cut:,}",
              help="Highest scores first; stops at capacity or when a call stops paying (EV ≤ 0).")
    m2.metric("Callable leads", f"{len(order):,}")
    reason = "capacity" if n_cut < n_ev else "break-even (EV ≤ 0)"
    m3.metric("Cutoff set by", reason)

    bands = st.multiselect("Show bands", ["Hot", "Warm", "Cold"], default=["Hot", "Warm", "Cold"])
    view = order.copy()
    view.insert(0, "rank", np.arange(1, len(view) + 1))
    view["call?"] = np.where(view["rank"] <= n_cut, "✅ call", "—")
    view = view[view["band"].isin(bands)]
    cols = ["rank", "call?", "Lead Number", "score", "band", "persona", "drivers_text"]
    st.dataframe(
        view[cols].rename(columns={"drivers_text": "top drivers (↑ raises, ↓ lowers the score)",
                                   "score": "score (0–100)"}),
        hide_index=True, use_container_width=True, height=460,
        column_config={"score (0–100)": st.column_config.ProgressColumn(
            "score (0–100)", min_value=0, max_value=100, format="%.0f")})
    export = order.assign(rank=np.arange(1, len(order) + 1), call=lambda d: d["rank"] <= n_cut)
    export = export[["rank", "call", "Prospect ID", "Lead Number", "score", "band", "persona",
                     "drivers_text"]]
    st.download_button("⬇️ Download call list (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "call_list.csv", "text/csv")
    st.caption("Score = calibrated probability of enrolling × 100. Bands from training-score deciles: "
               f"Hot ≥ {scorer.cutoffs['hot']:.0f} (top 20%), Warm ≥ {scorer.cutoffs['warm']:.0f} (next 30%), "
               "Cold below.")

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("When to stop calling")
    if has_outcomes:
        y_sorted = order[TARGET].to_numpy(dtype=float)
        basis = "replayed on the recorded outcomes of these leads"
    else:
        y_sorted = p_sorted  # expected enrolments from calibrated scores
        basis = "expected from calibrated scores (no outcomes in this file)"
    n = len(y_sorted)
    total = y_sorted.sum()
    cum = np.concatenate([[0], np.cumsum(y_sorted)])
    rate = total / n

    def stats(k):
        enrol = cum[k]
        return enrol, enrol / total, (k / enrol if enrol else float("inf")), \
            enrol * A.value_per_enrolment - k * A.cost_per_call

    me, mcap, mcpe, mprof = stats(n_cut)
    re_ = n_cut * rate
    rprof = re_ * A.value_per_enrolment - n_cut * A.cost_per_call
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calls (cutoff)", f"{n_cut:,}", f"{n_cut / n:.0%} of leads", delta_color="off")
    c2.metric("Enrolments captured", f"{mcap:.0%}", f"{mcap - n_cut / n:+.0%} vs random", )
    c3.metric("Calls per enrolment", f"{mcpe:.2f}", f"random: {1 / rate:.2f}", delta_color="off")
    c4.metric("Projected profit (assumed prices)", f"₹{mprof / 1e5:,.1f} lakh",
              f"₹{(mprof - rprof) / 1e5:+,.1f} lakh vs random")
    st.caption(f"Numbers are {basis}. Retrospective simulation: no causal or uplift claim. {A.label()}")

    x = np.arange(n + 1) / n * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=cum / total * 100, name="Call in score order",
                             line=dict(color=RED, width=3)))
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Call in random order",
                             line=dict(color=GREY, dash="dash")))
    fig.add_vline(x=n_cut / n * 100, line_dash="dot",
                  annotation_text=f"cutoff: {n_cut:,} calls → {mcap:.0%} of enrolments")
    if n_ev != n_cut:
        fig.add_vline(x=n_ev / n * 100, line_dash="dashdot", line_color=GREY,
                      annotation_text=f"EV>0 would be {n_ev:,} calls", annotation_position="bottom right")
    fig.update_layout(title="Gains curve: share of enrolments captured vs share of leads called",
                      xaxis_title="% of leads called (highest score first)",
                      yaxis_title="% of enrolments captured", height=460,
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=60))
    st.plotly_chart(fig, use_container_width=True)

    prof = cum * A.value_per_enrolment - np.arange(n + 1) * A.cost_per_call
    rprof_curve = np.arange(n + 1) * rate * A.value_per_enrolment - np.arange(n + 1) * A.cost_per_call
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=np.arange(n + 1), y=prof / 1e5, name="Score order",
                              line=dict(color=RED, width=3)))
    fig2.add_trace(go.Scatter(x=np.arange(n + 1), y=rprof_curve / 1e5, name="Random order",
                              line=dict(color=GREY, dash="dash")))
    fig2.add_vline(x=n_cut, line_dash="dot")
    fig2.update_layout(title=f"Cumulative profit (₹ lakh) at assumed prices", height=380,
                       xaxis_title="Number of leads called", yaxis_title="₹ lakh",
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)
    st.info(f"At these assumptions a call pays for itself above a **{A.break_even_probability:.1%}** chance "
            f"of enrolling, so the limit is usually counsellor time, not call cost. "
            f"Adding one counsellor adds {A.calls_per_counsellor_per_day * A.days_to_work_batch:,} calls per batch.")

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Call brief")
    client = B.get_client()
    if client is None:
        st.caption("No ANTHROPIC_API_KEY set: briefs come from a **deterministic template** built only "
                   "from the lead's data.")
    else:
        st.caption(f"Briefs from **{cfg['briefs']['model_name']}**, checked in code against the lead's data; "
                   "rejected output falls back to the template.")
    choices = order.head(max(n_cut, 50))
    labels = [f"#{i + 1} · Lead {r['Lead Number']} · score {r['score']:.0f} ({r['band']}) · {r['persona']}"
              for i, (_, r) in enumerate(choices.iterrows())]
    pick = st.selectbox("Pick a lead from the call list", range(len(labels)),
                        format_func=lambda i: labels[i])
    row = choices.iloc[pick]
    raw_row = raw_leads.loc[raw_leads["Prospect ID"] == row["Prospect ID"]].iloc[0] \
        if "Prospect ID" in raw_leads else raw_leads.iloc[row.name]
    inp = B.lead_facts(raw_row, json.loads(row["drivers"]), row["score"], row["band"], row["persona"])
    if st.button("Write brief", type="primary") or "brief_for" not in st.session_state \
            or st.session_state["brief_for"] != row["Prospect ID"]:
        with st.spinner("Writing brief…"):
            st.session_state["brief"] = B.generate_brief(str(row["Prospect ID"]), inp, client, cfg)
            st.session_state["brief_for"] = row["Prospect ID"]
    res = st.session_state["brief"]
    tag = "🤖 AI-written" if res["source"].startswith("llm") else "📝 Template"
    st.markdown(f"**{tag}** · source: `{res['source']}`" + (f" · model `{res['model']}`" if res["model"] else ""))
    st.code(res["text"], language=None, wrap_lines=True)
    if res["notes"]:
        with st.expander("Validator notes"):
            st.write(res["notes"])
    with st.expander("Exactly what the brief was allowed to use"):
        st.json(inp)
    st.caption("Briefs are aids for the counsellor, not facts about the lead.")

# ---------------------------------------------------------------- tab 4
with tab4:
    st.subheader("Lead personas")
    pt = pd.read_csv(ROOT / "reports" / "personas.csv").set_index("persona")
    counts = leads.loc[callable_mask, "persona"].value_counts()
    st.caption("Conversion rates measured on the held-out test leads. Personas describe who the leads are; "
               "the score still decides call order within each persona.")
    cols = st.columns(len(pt))
    for col, (name, r) in zip(cols, pt.iterrows()):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {name}")
                st.metric("Conversion", f"{r['conv_rate']:.0%}")
                colour = "green" if r["lift"] >= 1 else "red"
                st.markdown(f":{colour}[**{r['lift']:.1f}× baseline**]  \n"
                            f"**{r['share_of_leads']:.0%}** of leads · **{r['share_of_conversions']:.0%}** "
                            f"of enrolments  \n~{r['avg_minutes_on_site']:.0f} min on site on average  \n"
                            f"In this list: **{int(counts.get(name, 0)):,}** leads")
                st.info(r["playbook"])
