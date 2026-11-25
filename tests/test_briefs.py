import pandas as pd
import pytest

from src import briefs as B

ROW = pd.Series({
    "Prospect ID": "abc", "Lead Number": 1, "Lead Origin": "Landing Page Submission",
    "Lead Source": "Google", "What is your current occupation": "Working Professional",
    "What matters most to you in choosing a course": "Better Career Prospects",
    "Specialization": "Finance Management", "City": float("nan"),
    "Total Time Spent on Website": 1140, "TotalVisits": 4.0, "Page Views Per Visit": 2.0,
    "Do Not Email": "No", "A free copy of Mastering The Interview": "No", "Tags": "Will revert",
})
DRIVERS = [
    {"feature": "Total Time Spent on Website", "label": "Time on website", "value": "19 min",
     "effect": "raises", "shap": 1.2},
    {"feature": "What is your current occupation", "label": "Occupation",
     "value": "Working Professional", "effect": "raises", "shap": 0.9},
    {"feature": "Country", "label": "Country", "value": "India", "effect": "lowers", "shap": -0.3},
]


@pytest.fixture
def inp():
    return B.lead_facts(ROW, DRIVERS, 91.0, "Hot", "Website browsers")


def test_input_has_no_ids_leaky_or_blank_fields(inp):
    s = str(inp)
    assert "abc" not in s and "Will revert" not in s and "Tags" not in s
    assert "City" not in inp["facts"]                      # blank omitted
    assert [d["driver"] for d in inp["top_drivers"]] == ["Time on website", "Occupation"]  # Country dropped


def test_template_passes_validator(inp):
    text = B.template_brief(inp)
    ok, reasons = B.validate_brief(text, inp)
    assert ok, reasons
    assert len(text.splitlines()) == 3
    assert "full-time job" in text and "Finance Management" in text


def test_good_llm_style_brief_passes(inp):
    text = ("Why: 19 min on the website and a working professional profile.\n"
            "Objection: Fitting study around a job.\n"
            "Opener: Hi, thanks for exploring our Finance Management programme!")
    assert B.validate_brief(text, inp)[0]


@pytest.mark.parametrize("text,needle", [
    ("Why: Lives in Mumbai.\nObjection: Cost.\nOpener: Hi!", "not in the input"),
    ("Why: Came from Google search.\nObjection: Cost.\nOpener: Hi!", "not drivers"),
    ("Why: 45 min on the website.\nObjection: Cost.\nOpener: Hi!", "numbers"),
    ("Why: 19 min on the website.\nObjection: Said they will revert.\nOpener: Hi!", "post-contact"),
    ("Why: 19 min on the website.\nOpener: Hi!", "3 lines"),
])
def test_validator_rejects(inp, text, needle):
    ok, reasons = B.validate_brief(text, inp)
    assert not ok and any(needle in r for r in reasons)


def test_no_key_falls_back_to_template(inp, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert B.get_client() is None
    res = B.generate_brief("abc", inp, client=None)
    assert res["source"] == "template (no API key)" and res["text"] == B.template_brief(inp)


class FakeClient:
    """Mimics client.messages.create; returns queued texts."""
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        t = self.texts.pop(0)
        block = type("B", (), {"type": "text", "text": t})()
        return type("R", (), {"content": [block], "stop_reason": "end_turn"})()


def test_retry_then_template(inp, tmp_path):
    cfg = {"briefs": {"model_name": "m", "cache_dir": str(tmp_path), "max_tokens": 100}}
    bad = "Why: Lives in Mumbai.\nObjection: Cost.\nOpener: Hi!"
    c = FakeClient([bad, bad])
    res = B.generate_brief("abc", inp, client=c, cfg=cfg, use_cache=False)
    assert c.calls == 2 and res["source"] == "template (LLM output rejected)"


def test_valid_llm_brief_is_cached(inp, tmp_path):
    cfg = {"briefs": {"model_name": "m", "cache_dir": str(tmp_path), "max_tokens": 100}}
    good = "Why: 19 min on the website.\nObjection: Time.\nOpener: Hi there!"
    c = FakeClient([good])
    assert B.generate_brief("abc", inp, client=c, cfg=cfg)["source"] == "llm"
    again = B.generate_brief("abc", inp, client=FakeClient([]), cfg=cfg)
    assert again["source"] == "llm (cached)" and again["text"] == good
