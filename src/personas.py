"""Lead personas: KMeans on scaled behavioural + profile features.

k is chosen in [3, 6] by silhouette score. Names and playbooks are produced by
explicit rules over each cluster's profile and conversion lift, so they are
reproducible (no hand-labelling after looking at the output).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.clean import TARGET, fix_case_duplicates

FEATURES = ["log_time", "log_visits", "log_pages", "lead_add_form", "api_origin",
            "olark_chat", "working_professional", "profile_blank", "email_opt_out"]


def persona_features(df: pd.DataFrame, caps: dict | None = None) -> pd.DataFrame:
    """Behaviour (log time/visits/pages, capped) + profile flags known at creation."""
    d = fix_case_duplicates(df)
    caps = caps or {}
    occ = d["What is your current occupation"].fillna("Missing")

    def num(col):
        v = d[col].fillna(0).astype(float)
        return np.log1p(v.clip(upper=caps.get(col, v.max())))

    return pd.DataFrame({
        "log_time": num("Total Time Spent on Website"),
        "log_visits": num("TotalVisits"),
        "log_pages": num("Page Views Per Visit"),
        "lead_add_form": (d["Lead Origin"] == "Lead Add Form").astype(int),
        "api_origin": (d["Lead Origin"] == "API").astype(int),
        "olark_chat": (d["Lead Source"] == "Olark Chat").astype(int),
        "working_professional": (occ == "Working Professional").astype(int),
        "profile_blank": (occ == "Missing").astype(int),
        "email_opt_out": (d["Do Not Email"] == "Yes").astype(int),
    }, index=df.index)[FEATURES]


def name_persona(p: pd.Series, overall: pd.Series) -> str:
    """Rule-based plain-English name from a cluster's mean profile."""
    if p["lead_add_form"] > 0.5:
        return "Form-fill referrals"
    if p["working_professional"] > 0.5:
        return "Engaged working professionals"
    if p["email_opt_out"] > 0.5:
        return "Email opt-outs"
    if p["olark_chat"] > 0.5 and p["log_time"] < overall["log_time"]:
        return "Chat-only drop-ins"
    if p["profile_blank"] > 0.5:
        return "Anonymous browsers"
    if p["log_time"] >= overall["log_time"]:
        return "Website browsers"
    return "Other leads"


def playbook(lift: float) -> str:
    """One-line playbook from conversion lift vs baseline."""
    # The lead score already ranks leads within every persona, so playbooks
    # never override it (tested in notebook 04: skipping low-lift personas
    # costs enrolments under a capacity limit).
    if lift >= 1.5:
        return "Call first: these sit at the top of every day's queue"
    if lift >= 0.8:
        return "Call in score order (Hot, then Warm)"
    if lift >= 0.4:
        return "Call Hot/Warm in score order; put Cold into chat/WhatsApp follow-up"
    return "Deprioritise: call only if Hot/Warm; otherwise low-cost nurture (no email)"


@dataclass
class PersonaModel:
    scaler: StandardScaler
    kmeans: KMeans
    caps: dict
    names: dict
    playbooks: dict
    silhouettes: dict

    def assign(self, df: pd.DataFrame) -> pd.Series:
        Z = self.scaler.transform(persona_features(df, self.caps))
        return pd.Series(self.kmeans.predict(Z), index=df.index).map(self.names)


def fit_personas(train: pd.DataFrame, k_range=range(3, 7), random_state: int = 42,
                 sample_size: int = 3000) -> PersonaModel:
    caps = {c: float(train[c].quantile(0.99)) for c in
            ["Total Time Spent on Website", "TotalVisits", "Page Views Per Visit"]}
    F = persona_features(train, caps)
    scaler = StandardScaler().fit(F)
    Z = scaler.transform(F)
    sils, models = {}, {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        sils[k] = float(silhouette_score(Z, km.labels_, sample_size=min(sample_size, len(Z)),
                                         random_state=random_state))
        models[k] = km
    best_k = max(sils, key=sils.get)
    km = models[best_k]
    prof = F.groupby(km.labels_).mean()
    rates = train[TARGET].groupby(km.labels_).mean()
    base = train[TARGET].mean()
    names, plays = {}, {}
    for c in prof.index:
        n = name_persona(prof.loc[c], F.mean())
        # Disambiguate if two clusters get the same rule-based name.
        if n in names.values():
            n = f"{n} ({'higher' if rates[c] > base else 'lower'}-converting)"
        names[c] = n
        plays[n] = playbook(rates[c] / base)
    return PersonaModel(scaler, km, caps, names, plays, sils)


def persona_table(pm: PersonaModel, df: pd.DataFrame) -> pd.DataFrame:
    """Per persona: volume share, conversion rate, lift, share of conversions,
    average behaviour, playbook — evaluated on `df` (e.g. the test leads)."""
    persona = pm.assign(df)
    base = df[TARGET].mean()
    F = persona_features(df, pm.caps)
    t = df.assign(persona=persona).groupby("persona")[TARGET].agg(
        leads="size", conversions="sum", conv_rate="mean")
    t["share_of_leads"] = t["leads"] / len(df)
    t["share_of_conversions"] = t["conversions"] / t["conversions"].sum()
    t["lift"] = t["conv_rate"] / base
    t["avg_minutes_on_site"] = (np.expm1(F["log_time"]) / 60).groupby(persona).mean()
    t["playbook"] = [pm.playbooks[n] for n in t.index]
    return t.sort_values("lift", ascending=False)
