"""Load, clean and encode the X Education leads data.

Design: every cleaning decision (which columns to drop, rare categories,
caps, medians) is *learned on the training split only* by `LeadCleaner.fit`
and then applied unchanged to test data and to new uploaded leads. This keeps
the test set honest and makes scoring new leads reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.config import data_path

MISSING_TOKENS = ["Select"]
OTHER = "Other"
MISSING = "Missing"
TARGET = "Converted"

# Post-contact / sales-assigned columns (not known at lead creation) and IDs.
LEAKY_COLS = [
    "Tags", "Lead Quality", "Lead Profile",
    "Asymmetrique Activity Index", "Asymmetrique Profile Index",
    "Asymmetrique Activity Score", "Asymmetrique Profile Score",
    "Last Activity", "Last Notable Activity",
]
ID_COLS = ["Prospect ID", "Lead Number"]

CASE_FIXES = {"Lead Source": {"google": "Google"}}

# When is each column known? Written from the column meaning (data dictionary +
# values seen), not from the data. "unclear" = flagged to the project owner.
TIMING = {
    "Prospect ID": ("id", "Identifier, no predictive meaning"),
    "Lead Number": ("id", "Identifier, no predictive meaning"),
    "Lead Origin": ("creation", "How the lead entered the system (form, API, import)"),
    "Lead Source": ("creation", "Traffic source that produced the lead"),
    "Do Not Email": ("unclear", "Opt-out flag; could be set after the lead starts receiving emails"),
    "Do Not Call": ("unclear", "Opt-out flag; could be set after a call. Kept for training, excluded from call lists"),
    "TotalVisits": ("unclear", "Web activity measured at extraction; may include visits after first contact"),
    "Total Time Spent on Website": ("unclear", "Web activity measured at extraction; may include visits after first contact"),
    "Page Views Per Visit": ("unclear", "Web activity measured at extraction; may include visits after first contact"),
    "Last Activity": ("post-contact", "Latest logged activity, includes counsellor actions (e.g. 'SMS Sent', 'Had a Phone Conversation')"),
    "Country": ("creation", "Most likely IP / form geography at sign-up"),
    "Specialization": ("creation", "Sign-up form field"),
    "How did you hear about X Education": ("creation", "Sign-up form field"),
    "What is your current occupation": ("unclear", "Form field, but blank together with 'What matters most' for 99% of blanks: may be a later form step or counsellor-filled"),
    "What matters most to you in choosing a course": ("unclear", "Form field, but blank together with occupation for 99% of blanks: may be a later form step or counsellor-filled"),
    "Search": ("creation", "Ad channel the lead saw"),
    "Magazine": ("creation", "Ad channel the lead saw"),
    "Newspaper Article": ("creation", "Ad channel the lead saw"),
    "X Education Forums": ("creation", "Ad channel the lead saw"),
    "Newspaper": ("creation", "Ad channel the lead saw"),
    "Digital Advertisement": ("creation", "Ad channel the lead saw"),
    "Through Recommendations": ("creation", "Ad channel the lead saw"),
    "Receive More Updates About Our Courses": ("creation", "Sign-up form checkbox"),
    "Tags": ("post-contact", "Counsellor notes on call outcome (e.g. 'Will revert after reading the email')"),
    "Lead Quality": ("post-contact", "Counsellor's judgement of the lead after contact"),
    "Update me on Supply Chain Content": ("creation", "Sign-up form checkbox"),
    "Get updates on DM Content": ("creation", "Sign-up form checkbox"),
    "Lead Profile": ("post-contact", "Profile assigned by the sales team"),
    "City": ("creation", "Sign-up form field"),
    "Asymmetrique Activity Index": ("post-contact", "Company-computed score from activity/profile; assigned later"),
    "Asymmetrique Profile Index": ("post-contact", "Company-computed score from activity/profile; assigned later"),
    "Asymmetrique Activity Score": ("post-contact", "Company-computed score from activity/profile; assigned later"),
    "Asymmetrique Profile Score": ("post-contact", "Company-computed score from activity/profile; assigned later"),
    "I agree to pay the amount through cheque": ("creation", "Sign-up form checkbox"),
    "A free copy of Mastering The Interview": ("creation", "Sign-up form checkbox"),
    "Last Notable Activity": ("post-contact", "Latest notable activity, includes counsellor actions"),
}


# ---------------------------------------------------------------- pure helpers
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read Leads.csv, treating the form placeholder "Select" as missing."""
    path = Path(path) if path else data_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Place Leads.csv in data/.")
    return pd.read_csv(path, na_values=MISSING_TOKENS)


def fix_case_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge categories that differ only in spelling/case (e.g. google/Google)."""
    out = df.copy()
    for col, mapping in CASE_FIXES.items():
        if col in out:
            out[col] = out[col].replace(mapping)
    return out


def merge_rare(s: pd.Series, min_share: float = 0.01, other: str = OTHER,
               keep_na: bool = True) -> pd.Series:
    """Replace categories with < `min_share` of rows by `other`.

    Share is computed over all rows (including missing). Missing values are
    left as NaN when `keep_na` is True.
    """
    shares = s.value_counts(dropna=True) / len(s)
    rare = shares[shares < min_share].index
    out = s.where(~s.isin(rare), other)
    if not keep_na:
        out = out.fillna(MISSING)
    return out


def yes_no_to_int(s: pd.Series) -> pd.Series:
    """Map Yes/No (any case) to 1/0; anything else becomes NaN."""
    return s.astype("string").str.strip().str.lower().map({"yes": 1, "no": 0}).astype("float")


def is_yes_no(s: pd.Series) -> bool:
    vals = set(s.dropna().astype(str).str.strip().str.lower().unique())
    return bool(vals) and vals <= {"yes", "no"}


def missingness_is_predictive(miss: pd.Series, y: pd.Series,
                              alpha: float = 0.01, min_gap_pp: float = 5.0) -> bool:
    """True if rows with the field missing convert differently (chi-square p < alpha
    and absolute gap >= min_gap_pp percentage points)."""
    if miss.sum() == 0 or miss.all():
        return False
    gap = (y[miss].mean() - y[~miss].mean()) * 100
    p = chi2_contingency(pd.crosstab(miss, y))[1]
    return p < alpha and abs(gap) >= min_gap_pp


# ---------------------------------------------------------------- fitted cleaner
@dataclass
class LeadCleaner:
    """Learn cleaning rules on training data, apply them anywhere.

    Steps: drop IDs + `drop_cols` → Yes/No → 1/0 → drop near-constant columns →
    drop >`missing_threshold` missing columns unless missingness is predictive →
    numeric: cap at `cap_quantile`, median-impute, add is_missing flag →
    categorical: "Missing" level, merge rare (<`rare_share`) into "Other" →
    one-hot encode with a column set frozen at fit time.
    """
    drop_cols: list[str] = field(default_factory=lambda: ID_COLS + LEAKY_COLS)
    missing_threshold: float = 0.40
    rare_share: float = 0.01
    cap_quantile: float = 0.99
    near_constant_share: float = 0.995

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        X = fix_case_duplicates(df)
        return X.drop(columns=[c for c in self.drop_cols + [TARGET] if c in X])

    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "LeadCleaner":
        y = df[TARGET] if y is None else y
        X = self._prep(df)
        self.log_: list[dict] = [
            {"column": c, "action": "dropped",
             "reason": "identifier" if c in ID_COLS else "post-contact (leakage)"}
            for c in self.drop_cols if c in df]

        self.yes_no_ = [c for c in X if is_yes_no(X[c])]
        for c in self.yes_no_:
            X[c] = yes_no_to_int(X[c])

        self.dropped_ = []
        for c in X.columns:
            top = X[c].value_counts(normalize=True, dropna=False).iloc[0]
            if top >= self.near_constant_share:
                self.dropped_.append(c)
                self.log_.append({"column": c, "action": "dropped",
                                  "reason": f"near-constant ({top:.1%} one value)"})
                continue
            miss = X[c].isna()
            if miss.mean() > self.missing_threshold:
                if missingness_is_predictive(miss, y):
                    self.log_.append({"column": c, "action": "kept + missing flag",
                                      "reason": f"{miss.mean():.0%} missing, but missingness predicts conversion"})
                else:
                    self.dropped_.append(c)
                    self.log_.append({"column": c, "action": "dropped",
                                      "reason": f"{miss.mean():.0%} missing, missingness not predictive"})
        X = X.drop(columns=self.dropped_)
        self.yes_no_ = [c for c in self.yes_no_ if c not in self.dropped_]

        self.numeric_ = [c for c in X if pd.api.types.is_numeric_dtype(X[c])
                         and c not in self.yes_no_]
        self.categorical_ = [c for c in X if c not in self.numeric_ and c not in self.yes_no_]

        self.caps_ = {c: float(X[c].quantile(self.cap_quantile)) for c in self.numeric_}
        self.medians_ = {c: float(X[c].median()) for c in self.numeric_ + self.yes_no_}
        self.num_flags_ = [c for c in self.numeric_ if X[c].isna().any()]
        for c in self.numeric_:
            self.log_.append({"column": c, "action": "kept (numeric)",
                              "reason": f"capped at 99th pct = {self.caps_[c]:g}"
                              + ("; median-imputed + is_missing flag" if c in self.num_flags_ else "")})
        for c in self.yes_no_:
            self.log_.append({"column": c, "action": "kept (Yes/No → 1/0)", "reason": ""})

        self.keep_levels_ = {}
        for c in self.categorical_:
            s = X[c].fillna(MISSING)
            shares = s.value_counts(normalize=True)
            self.keep_levels_[c] = sorted(shares[shares >= self.rare_share].index)
            if not any(r["column"] == c for r in self.log_):
                self.log_.append({"column": c, "action": "kept (one-hot)",
                                  "reason": f"{len(self.keep_levels_[c])} levels kept, rest → Other; blanks → 'Missing' level"})

        self.feature_names_ = list(self._encode(X).columns)
        return self

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for c in self.numeric_:
            v = X[c].astype(float)
            if c in self.num_flags_:
                parts.append(v.isna().astype(int).rename(f"{c}__is_missing"))
            parts.append(v.clip(upper=self.caps_[c]).fillna(self.medians_[c]).rename(c))
        for c in self.yes_no_:
            parts.append(X[c].astype(float).fillna(self.medians_[c]).rename(c))
        for c in self.categorical_:
            s = X[c].fillna(MISSING)
            s = s.where(s.isin(self.keep_levels_[c]), OTHER)
            levels = self.keep_levels_[c] + ([OTHER] if OTHER not in self.keep_levels_[c] else [])
            for lvl in levels:
                parts.append((s == lvl).astype(int).rename(f"{c}={lvl}"))
        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prep(df)
        missing = [c for c in self.numeric_ + self.yes_no_ + self.categorical_ if c not in X]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        for c in self.yes_no_:
            X[c] = yes_no_to_int(X[c])
        out = self._encode(X)
        return out.reindex(columns=self.feature_names_, fill_value=0)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def decisions(self) -> pd.DataFrame:
        return pd.DataFrame(self.log_).drop_duplicates("column", keep="first").set_index("column")


def leakage_audit(cleaner: LeadCleaner, columns: list[str]) -> pd.DataFrame:
    """Audit table: column | when known | decision | reason."""
    dec = cleaner.decisions()
    rows = []
    for c in [c for c in columns if c != TARGET]:
        when, why = TIMING.get(c, ("unclear", "not documented"))
        d = dec.loc[c] if c in dec.index else None
        decision = d["action"] if d is not None else "kept"
        detail = d["reason"] if d is not None else ""
        reason = why if when in ("id", "post-contact") else (
            f"{why}. {detail}".strip(". ") if detail else why)
        rows.append({"column": c, "when_known": when, "decision": decision,
                     "reason": reason})
    return pd.DataFrame(rows)
