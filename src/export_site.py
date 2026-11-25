"""Export everything the static website needs into site/data/site.json.

The website recomputes the call plan in the browser (sliders), so this file
carries: the held-out test leads in score order (score, band, lead group,
drivers, template brief, recorded outcome), lead-group table, leakage result,
recommendations and defaults from config.yaml. No Prospect IDs.

    python -m src.export_site
"""
from __future__ import annotations

import json

import pandas as pd

from src import briefs as B
from src.clean import TARGET, load_raw
from src.config import ROOT, load_config
from src.model import split


def main():
    cfg = load_config()
    raw = load_raw()
    _, test_raw = split(raw, cfg)
    scored = pd.read_csv(ROOT / "models" / "scored_leads.csv")
    test = (scored[scored.split == "test"].set_index("Prospect ID")
            .loc[test_raw["Prospect ID"]].reset_index())
    raw_by_id = test_raw.set_index("Prospect ID")
    # Same ordering as the notebooks and the Streamlit app (stable sort on test-split order).
    test = test.sort_values("probability", ascending=False, kind="mergesort").reset_index(drop=True)

    leads = []
    for _, r in test.iterrows():
        drivers = json.loads(r["drivers"])
        inp = B.lead_facts(raw_by_id.loc[r["Prospect ID"]], drivers, r["score"], r["band"], r["persona"])
        brief = B.template_brief(inp)
        parts = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in brief.splitlines()}
        leads.append({
            "n": int(r["Lead Number"]), "p": round(float(r["probability"]), 6),
            "s": round(float(r["score"]), 1), "b": r["band"], "g": r["persona"],
            "d": [{"l": d["label"], "v": d["value"], "up": d["effect"] == "raises"} for d in drivers],
            "y": int(r[TARGET]), "dnc": r["Do Not Call"] == "Yes",
            "why": parts["Why"], "obj": parts["Objection"], "open": parts["Opener"],
            "facts": inp["facts"],
        })

    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    personas = pd.read_csv(ROOT / "reports" / "personas.csv")
    leak = pd.read_csv(ROOT / "reports" / "leakage_experiment.csv", index_col=0)
    recs = pd.read_csv(ROOT / "reports" / "recommendations.csv")
    scorer_cut = card["band_cutoffs"]
    out = {
        "defaults": {**cfg["economics"], **cfg["capacity"]},
        "cutoffs": scorer_cut,
        "model": {"name": card["model"], "calibration": card["calibration"],
                  "test_leads": card["test_leads"], "baseline": card["test_baseline"],
                  "headline": card["headline_numbers"]},
        "leakage": {k: {"auc": float(v["Test AUC"]), "p20": float(v["Precision @ top 20%"])}
                    for k, v in leak.iterrows()},
        "personas": personas[["persona", "leads", "conversions", "conv_rate", "share_of_leads",
                              "share_of_conversions", "lift", "avg_minutes_on_site",
                              "playbook"]].round(4).to_dict(orient="records"),
        "recommendations": recs.to_dict(orient="records"),
        "leads": leads,
    }
    path = ROOT / "site" / "data" / "site.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(leads)} leads)")


if __name__ == "__main__":
    main()
