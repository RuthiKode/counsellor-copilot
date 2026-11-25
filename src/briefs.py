"""Counsellor call briefs: ≤3 lines — why this lead, likely objection, opening line.

Guardrails
----------
* The LLM sees ONLY the lead's non-PII, creation-time features and its top-3
  SHAP drivers (no IDs, no post-contact columns).
* The prompt says: only cite reasons present in the drivers; invent nothing.
* `validate_brief` re-checks the output in code: format, no features outside
  the input, "Why" cites drivers only, no numbers that aren't in the input, no
  post-contact language. One retry with the rejection reasons; then a
  deterministic template.
* No ANTHROPIC_API_KEY → template only, so the app always runs.
* LLM briefs are cached on disk keyed by lead ID + model (+ input hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, load_config
from src.model import RAW_LABELS, describe_value

# Columns the brief may use (all known at lead creation; no IDs).
BRIEF_FIELDS = [
    "Lead Origin", "Lead Source", "What is your current occupation",
    "What matters most to you in choosing a course", "Specialization", "City",
    "Total Time Spent on Website", "TotalVisits", "Page Views Per Visit",
    "Do Not Email", "A free copy of Mastering The Interview",
]

# Words that reveal a feature is being talked about.
FEATURE_TERMS = {
    "Total Time Spent on Website": ["time on website", "time on the website", "time on site",
                                    "minutes", " min", "browsing", "browsed"],
    "TotalVisits": ["visit"],
    "Page Views Per Visit": ["page view", "pages per", "pages"],
    "Lead Origin": ["lead add form", "landing page", "enquiry form", "inquiry form", "api"],
    "Lead Source": ["google", "direct traffic", "olark", "chat", "organic search", "reference",
                    "referral", "referred", "welingak", "facebook"],
    "What is your current occupation": ["working professional", "unemployed", "student",
                                        "occupation", "job", "profession", "employ", "housewife",
                                        "businessman"],
    "What matters most to you in choosing a course": ["career prospects", "motivation",
                                                      "what matters most"],
    "Specialization": ["specialization", "specialisation", "management", "e-commerce",
                       "business administration", "tourism", "media and advertising",
                       "international business", "banking", "rural", "services excellence"],
    "City": ["mumbai", "thane", "maharashtra", "metro", "tier ii", "city"],
    "Do Not Email": ["opted out", "opt-out", "opt out", "unsubscribe", "no email"],
    "A free copy of Mastering The Interview": ["e-book", "ebook", "mastering the interview",
                                               "interview guide"],
}

# Post-contact / leaky information the brief must never claim to know.
FORBIDDEN_TERMS = ["tag", "lead quality", "lead profile", "asymmetrique", "spoke", "previous call",
                   "last call", "called before", "revert", "ringing", "switched off",
                   "already a student", "last activity", "sms", "conversation with",
                   "spoken", "follow-up call", "you told"]

LINE_PREFIXES = ("Why:", "Objection:", "Opener:")

SYSTEM_PROMPT = """You write call briefs for education counsellors at X Education.
Given one lead's facts and its top-3 score drivers, write EXACTLY three lines:
Why: <why this lead is likely to convert, citing only the drivers>
Objection: <the most likely objection, based only on the facts given>
Opener: <a friendly first sentence the counsellor can say>

Rules:
- Only cite reasons present in the provided drivers. Do not invent facts about the lead.
- Never mention previous calls, conversations, notes, ratings or anything not in the input.
- Do not use any number that is not in the input.
- Plain English, no jargon, no markdown, each line under 30 words."""


# ---------------------------------------------------------------- inputs
def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def lead_facts(row: pd.Series, drivers: list[dict], score: float, band: str,
               persona: str | None = None) -> dict:
    """Non-PII, creation-time facts for one lead + its top-3 drivers."""
    facts = {}
    for col in BRIEF_FIELDS:
        v = row.get(col)
        if _blank(v) or v == "Select":
            continue
        if col in ("Do Not Email", "A free copy of Mastering The Interview"):
            if str(v).lower() != "yes":
                continue
            v = "yes"
        facts[RAW_LABELS.get(col, col)] = describe_value(col, v)
    # Country is excluded: a blank country mostly means "never visited the site"
    # (a proxy for lead origin, see notebook 03), so it is not a reason to give a counsellor.
    out = {"lead_score": f"{score:.0f}/100", "band": band, "facts": facts,
           "top_drivers": [{"driver": d["label"], "value": d["value"],
                            "effect": f"{d['effect']} the score"}
                           for d in drivers if d["feature"] in BRIEF_FIELDS]}
    if persona:
        out["persona"] = persona
    return out


def _present_features(inp: dict) -> set[str]:
    """Features in the input: answered facts plus drivers (a blank field can
    itself be a driver, e.g. 'Main motivation: left blank')."""
    labels = {v: k for k, v in RAW_LABELS.items()}
    return {labels.get(k, k) for k in inp["facts"]} | _driver_features(inp)


def _driver_features(inp: dict) -> set[str]:
    labels = {v: k for k, v in RAW_LABELS.items()}
    return {labels.get(d["driver"], d["driver"]) for d in inp["top_drivers"]}


def _mentioned(text: str) -> set[str]:
    t = f" {text.lower()} "
    found = set()
    for feat, terms in FEATURE_TERMS.items():
        for term in terms:
            if term.strip() in ("api",):
                if re.search(r"\bapi\b", t):
                    found.add(feat)
            elif term in t:
                found.add(feat)
    return found


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


# ---------------------------------------------------------------- validator
def validate_brief(text: str, inp: dict) -> tuple[bool, list[str]]:
    """Return (ok, reasons). Checks format, grounding, numbers and leakage."""
    reasons = []
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if len(lines) != 3 or not all(ln.startswith(p) for ln, p in zip(lines, LINE_PREFIXES)):
        reasons.append("must be exactly 3 lines starting 'Why:', 'Objection:', 'Opener:'")
    low = (text or "").lower()
    bad = [w for w in FORBIDDEN_TERMS if re.search(rf"\b{re.escape(w)}", low)]
    if bad:
        reasons.append(f"mentions post-contact information: {bad}")
    present = _present_features(inp)
    extra = _mentioned(text) - present
    if extra:
        reasons.append(f"mentions features not in the input: {sorted(extra)}")
    why = next((ln for ln in lines if ln.startswith("Why:")), "")
    why_extra = _mentioned(why) - _driver_features(inp)
    if why_extra:
        reasons.append(f"'Why' cites reasons that are not drivers: {sorted(why_extra)}")
    allowed_nums = _numbers(json.dumps(inp, ensure_ascii=False))
    stray = _numbers(text) - allowed_nums
    if stray:
        reasons.append(f"uses numbers not in the input: {sorted(stray)}")
    return (not reasons), reasons


# ---------------------------------------------------------------- template
def template_brief(inp: dict) -> str:
    """Deterministic brief built only from the input (always passes validation)."""
    f = inp["facts"]
    ups = [d for d in inp["top_drivers"] if d["effect"].startswith("raises")]
    shown = ups or inp["top_drivers"]
    kind = "Strongest signals" if ups else "Main signals (all pull the score down)"
    why = "; ".join(f"{d['driver']}: {d['value']}" for d in shown)
    why = f"Why: Score {inp['lead_score']} ({inp['band']}). {kind}: {why}."

    occ = f.get(RAW_LABELS["What is your current occupation"])
    if occ == "Working Professional":
        obj = "Objection: Time: fitting the course around a full-time job."
    elif occ == "Student":
        obj = "Objection: Cost: fees may need family sign-off."
    elif occ == "Unemployed":
        obj = "Objection: Cost and payoff: will this course lead to work?"
    else:
        obj = "Objection: Unclear goal: ask what they want from the course before pitching."

    origin = f.get(RAW_LABELS["Lead Origin"], "")
    source = f.get(RAW_LABELS["Lead Source"], "")
    spec = f.get(RAW_LABELS["Specialization"])
    if origin == "Lead Add Form":
        via = " after being referred" if source in ("Reference", "Welingak Website") else ""
        opener = (f"Opener: Hi, you filled in our enquiry form{via}; is now a good time to talk "
                  f"about what you're looking for?")
    elif source == "Olark Chat":
        opener = "Opener: Hi, you reached out on our website chat; can I help with the question you had?"
    elif spec:
        opener = f"Opener: Hi, thanks for exploring our {spec} programme; what are you hoping it will do for you?"
    else:
        opener = "Opener: Hi, thanks for your interest in X Education; what made you look into our courses?"
    return "\n".join([why, obj, opener])


# ---------------------------------------------------------------- LLM
def get_client():
    """Anthropic client if ANTHROPIC_API_KEY is set, else None (template mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    return anthropic.Anthropic()


def _call_llm(client, model: str, inp: dict, feedback: list[str] | None, max_tokens: int) -> str:
    import anthropic
    msg = "Lead (JSON):\n" + json.dumps(inp, ensure_ascii=False, indent=1)
    if feedback:
        msg += ("\n\nYour previous brief was rejected because: " + "; ".join(feedback)
                + ". Write it again following every rule.")
    try:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": msg}],
        )
    except anthropic.APIError as e:  # network, auth, rate limit, server errors
        raise RuntimeError(f"API error: {type(e).__name__}") from e
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined")
    return "\n".join(b.text for b in resp.content if b.type == "text").strip()


def _cache_path(cache_dir: Path, lead_id: str, model: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{model}__{lead_id}")
    return cache_dir / f"{safe}.json"


def _hash(inp: dict) -> str:
    return hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest()[:16]


def generate_brief(lead_id: str, inp: dict, client=None, cfg: dict | None = None,
                   use_cache: bool = True) -> dict:
    """Return {"text", "source", "model", "notes"}.

    source ∈ {"llm", "llm (cached)", "template (no API key)",
              "template (LLM output rejected)", "template (LLM error)"}.
    """
    cfg = cfg or load_config()
    b = cfg["briefs"]
    model = b["model_name"]
    if client is None:
        return {"text": template_brief(inp), "source": "template (no API key)",
                "model": None, "notes": []}

    cache_dir = ROOT / b["cache_dir"]
    path = _cache_path(cache_dir, lead_id, model)
    h = _hash(inp)
    if use_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("input_hash") == h:
            return {**cached["result"], "source": cached["result"]["source"] + " (cached)"}

    notes, feedback = [], None
    for attempt in range(2):  # first try + 1 retry
        try:
            text = _call_llm(client, model, inp, feedback, b["max_tokens"])
        except RuntimeError as e:
            return {"text": template_brief(inp), "source": "template (LLM error)",
                    "model": model, "notes": [str(e)]}
        ok, reasons = validate_brief(text, inp)
        if ok:
            result = {"text": text, "source": "llm", "model": model, "notes": notes}
            cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"input_hash": h, "result": result}, indent=1),
                            encoding="utf-8")
            return result
        notes.append(f"attempt {attempt + 1} rejected: {'; '.join(reasons)}")
        feedback = reasons
    return {"text": template_brief(inp), "source": "template (LLM output rejected)",
            "model": model, "notes": notes}


# ---------------------------------------------------------------- samples
def write_sample_briefs(n: int = 20, out: Path | None = None, seed: int = 42) -> Path:
    """Pre-generate briefs for n Hot test leads (excluding Do Not Call)."""
    from src.clean import load_raw
    cfg = load_config()
    scored = pd.read_csv(ROOT / "models" / "scored_leads.csv")
    raw = load_raw().set_index("Prospect ID")
    hot = scored[(scored.split == "test") & (scored.band == "Hot") & (scored["Do Not Call"] != "Yes")]
    pick = hot.sample(min(n, len(hot)), random_state=seed).sort_values("score", ascending=False)
    client = get_client()
    lines = ["# Sample call briefs (Hot test leads)", "",
             f"{len(pick)} Hot leads sampled from the held-out test set (random_state={seed}). "
             "Each brief is labelled with how it was produced. LLM briefs are aids for the "
             "counsellor, not facts about the lead.", ""]
    for _, r in pick.iterrows():
        drivers = json.loads(r["drivers"])
        inp = lead_facts(raw.loc[r["Prospect ID"]], drivers, r["score"], r["band"], r.get("persona"))
        res = generate_brief(r["Prospect ID"], inp, client, cfg)
        label = res["source"] + (f" · {res['model']}" if res["model"] and "llm" in res["source"] else "")
        lines += [f"### Lead {r['Lead Number']} — score {r['score']:.0f} ({r['band']}), "
                  f"{r.get('persona', '')}", f"*Source: {label}*", "", "```", res["text"], "```", ""]
    out = out or ROOT / "reports" / "sample_briefs.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate sample call briefs.")
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args()
    print("wrote", write_sample_briefs(a.samples))
