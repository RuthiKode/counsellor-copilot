"""Build reports/business_summary.md and a one-page PDF from saved results.

Every number is read from files the notebooks wrote (models/model_card.json,
reports/*.csv), so nothing in the summary is typed by hand.

    python -m src.report            # writes .md + .html, then .pdf via Edge/Chrome
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.config import ROOT, load_config

REP = ROOT / "reports"


def load_numbers() -> dict:
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    cut = pd.read_csv(REP / "cutoff_scenarios.csv", index_col=0)
    personas = pd.read_csv(REP / "personas.csv")
    recs = pd.read_csv(REP / "recommendations.csv")
    leak = pd.read_csv(REP / "leakage_experiment.csv", index_col=0)
    cfg = load_config()
    cap_row = cut[cut.index.str.startswith("Capacity")].iloc[0]
    rnd_row = cut[cut.index.str.startswith("Random")].iloc[0]
    h = card["headline_numbers"]
    return dict(card=card, h=h, cap=cap_row, rnd=rnd_row, personas=personas, recs=recs,
                leak=leak, cfg=cfg, n_test=card["test_leads"], base=card["test_baseline"])


def _assumption_line(cfg: dict) -> str:
    e, c = cfg["economics"], cfg["capacity"]
    return (f"Assumptions (not measured): course price ₹{e['course_price']:,}, gross margin "
            f"{e['gross_margin']:.0%}, ₹{e['cost_per_call']} per call, {c['counsellors']} counsellors × "
            f"{c['calls_per_counsellor_per_day']} calls/day, {c['days_to_work_batch']} days to work a batch.")


def build_markdown(n: dict) -> str:
    h, cap, rnd = n["h"], n["cap"], n["rnd"]
    extra = cap["enrolments"] - rnd["enrolments"]
    gain = (cap["profit"] - rnd["profit"]) / 1e5
    lines = [
        "# Counsellor Copilot: who to call, when to stop, what to say",
        "",
        "## Situation",
        f"Counsellors can't call every lead. In the historical data {n['base']:.1%} of leads enrolled. "
        f"At the assumed capacity, a team can call {cap['calls']:,.0f} of every {n['n_test']:,} new leads "
        f"in a week ({cap['pct_leads_called']:.0%}), so the order of calls decides how many enrolments it reaches.",
        "",
        "## Headline result",
        f"**Calling leads in score order, the top {h['k']:.0%} of leads would capture "
        f"{h['recall']:.0%} of enrolments at {h['lift']:.1f}× the baseline rate.** "
        f"With {cap['calls']:,.0f} calls, score order reaches {cap['pct_enrolments_captured']:.0%} of enrolments "
        f"vs {rnd['pct_enrolments_captured']:.0%} in random order: about {extra:,.0f} more enrolments "
        f"(₹{gain:.1f} lakh at the assumed prices). Tested on {n['n_test']:,} past leads the model never saw.",
        "",
        "![How many enrolments you reach as you work down the list](figures/03_gains.png)",
        "",
        "## Who the leads are",
        "| Lead group | Share of leads | Enrol rate | Share of enrolments | Playbook |",
        "|---|---|---|---|---|",
    ]
    for r in n["personas"].itertuples():
        lines.append(f"| {r.persona} | {r.share_of_leads:.0%} | {r.conv_rate:.0%} | "
                     f"{r.share_of_conversions:.0%} | {r.playbook} |")
    lines += ["", "## Three recommendations",
              "| # | Action | Owner | Evidence | Expected effect |", "|---|---|---|---|---|"]
    for i, r in enumerate(n["recs"].itertuples(), 1):
        lines.append(f"| {i} | {r.action} | {r.owner} | {r.evidence} | {r.effect} |")
    lines += ["", f"*{_assumption_line(n['cfg'])} Retrospective simulation on one company's past leads: "
                  "projections, not measured uplift. Call briefs are aids, not facts.*", ""]
    return "\n".join(lines)


def build_html(n: dict) -> str:
    h, cap, rnd = n["h"], n["cap"], n["rnd"]
    extra = cap["enrolments"] - rnd["enrolments"]
    gain = (cap["profit"] - rnd["profit"]) / 1e5
    esc = html.escape
    prow = "".join(
        f"<tr><td>{esc(r.persona)}</td><td>{r.share_of_leads:.0%}</td><td>{r.conv_rate:.0%}</td>"
        f"<td>{r.share_of_conversions:.0%}</td><td>{esc(r.playbook)}</td></tr>"
        for r in n["personas"].itertuples())
    rrow = "".join(
        f"<tr><td class='n'>{i}</td><td><b>{esc(r.action)}</b></td><td>{esc(r.owner)}</td>"
        f"<td>{esc(r.evidence)}</td><td>{esc(r.effect)}</td></tr>"
        for i, r in enumerate(n["recs"].itertuples(), 1))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Counsellor Copilot: business summary</title>
<style>
@page {{ size: A4; margin: 11mm 12mm; }}
body {{ font-family: "Segoe UI", Arial, sans-serif; font-size: 8.6pt; color: #1f2933; line-height: 1.32; margin: 0; }}
h1 {{ font-size: 15pt; margin: 0 0 2px; }}
.sub {{ color: #52606d; margin: 0 0 8px; font-size: 9pt; }}
h2 {{ font-size: 10pt; margin: 9px 0 4px; color: #d64545; text-transform: uppercase; letter-spacing: .04em; }}
.headline {{ background: #fdf0f0; border-left: 4px solid #d64545; padding: 7px 10px; font-size: 10pt; }}
.headline b {{ font-size: 11pt; }}
.row {{ display: flex; gap: 12px; align-items: flex-start; }}
.row img {{ width: 44%; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #e4e7eb; padding: 3px 5px; text-align: left; vertical-align: top; }}
th {{ background: #f5f7fa; font-weight: 600; }}
td.n {{ font-weight: 700; color: #d64545; }}
.foot {{ color: #616e7c; font-size: 7.4pt; margin-top: 8px; border-top: 1px solid #e4e7eb; padding-top: 4px; }}
</style></head><body>
<h1>Counsellor Copilot</h1>
<p class="sub">Who to call, when to stop, what to say: a lead-scoring plan for the admissions team</p>
<h2>Situation</h2>
<p>Counsellors can't call every lead. In the historical data <b>{n['base']:.1%}</b> of leads enrolled.
At the assumed capacity a team can call <b>{cap['calls']:,.0f}</b> of every <b>{n['n_test']:,}</b> new leads in a week
({cap['pct_leads_called']:.0%}), so the order of calls decides how many enrolments it reaches.</p>
<div class="headline"><b>Calling leads in score order, the top {h['k']:.0%} of leads would capture {h['recall']:.0%}
of enrolments at {h['lift']:.1f}&times; the baseline rate.</b><br>
With {cap['calls']:,.0f} calls, score order reaches <b>{cap['pct_enrolments_captured']:.0%}</b> of enrolments vs
<b>{rnd['pct_enrolments_captured']:.0%}</b> in random order: about <b>{extra:,.0f} more enrolments</b>
(&#8377;{gain:.1f} lakh at the assumed prices). Tested on {n['n_test']:,} past leads the model never saw.</div>
<h2>How fast enrolments come in &amp; who the leads are</h2>
<div class="row"><img src="figures/03_gains.png" alt="Gains chart">
<table><tr><th>Lead group</th><th>Share of leads</th><th>Enrol rate</th><th>Share of enrolments</th><th>Playbook</th></tr>{prow}</table></div>
<h2>Three recommendations</h2>
<table><tr><th>#</th><th>Action</th><th>Owner</th><th>Evidence</th><th>Expected effect</th></tr>{rrow}</table>
<p class="foot">{esc(_assumption_line(n['cfg']))} Retrospective simulation on one company's past leads (Kaggle
&ldquo;Lead Scoring, X Education&rdquo;): projections, not measured uplift. Call briefs are aids for counsellors, not facts.</p>
</body></html>"""


def find_browser() -> str | None:
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("google-chrome") or shutil.which("chromium")


def main():
    n = load_numbers()
    (REP / "business_summary.md").write_text(build_markdown(n), encoding="utf-8")
    html_path = REP / "business_summary.html"
    html_path.write_text(build_html(n), encoding="utf-8")
    browser = find_browser()
    if not browser:
        print("No Edge/Chrome found: open business_summary.html and print to PDF.")
        return
    pdf = REP / "business_summary.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html_path.as_uri()], check=False,
                   capture_output=True, timeout=120)
    print("wrote", REP / "business_summary.md", "and", pdf if pdf.exists() else "(PDF failed)")


if __name__ == "__main__":
    main()
