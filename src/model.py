"""Train, compare, calibrate and explain lead-scoring models.

Everything that chooses something (class weights, LightGBM params, the A-vs-B
decision, calibration method, band cutoffs) uses cross-validation on the
TRAINING split only. The test split is touched once, for reporting.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.clean import MISSING, OTHER, TARGET, LeadCleaner
from src.config import load_config
from src import metrics

# LightGBM is fed arrays (see `lightgbm`); sklearn then warns about names. Harmless.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ---------------------------------------------------------------- split + models
def split(df: pd.DataFrame, cfg: dict | None = None):
    """Stratified train/test split using config test_size and random_state."""
    cfg = cfg or load_config()
    m = cfg["model"]
    return train_test_split(df, test_size=m["test_size"], stratify=df[TARGET],
                            random_state=m["random_state"])


def cv_splitter(cfg: dict | None = None) -> StratifiedKFold:
    cfg = cfg or load_config()
    return StratifiedKFold(n_splits=cfg["model"]["cv_folds"], shuffle=True,
                           random_state=cfg["model"]["random_state"])


def logistic(class_weight=None, random_state: int = 42, C: float = 1.0) -> Pipeline:
    """Scaled logistic regression."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000,
                                  random_state=random_state)),
    ])


def _to_array(X):
    return np.asarray(X, dtype=float)


def lightgbm(random_state: int = 42, **params) -> Pipeline:
    """LightGBM fed plain arrays (feature names like 'City=Thane & Outskirts'
    contain characters LightGBM rejects)."""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import FunctionTransformer
    base = dict(random_state=random_state, n_jobs=1, verbose=-1, deterministic=True,
                force_row_wise=True, bagging_freq=1)
    base.update(params)
    return Pipeline([("to_array", FunctionTransformer(_to_array)),
                     ("lgbm", LGBMClassifier(**base))])


def oof_proba(estimator, X, y, cv) -> np.ndarray:
    """Out-of-fold predicted probabilities on the training data."""
    return cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba")[:, 1]


# ---------------------------------------------------------------- model A / B
def select_logistic(X, y, cfg: dict | None = None) -> tuple[str, pd.DataFrame]:
    """Pick class_weight (None vs 'balanced') by out-of-fold PR-AUC."""
    cfg = cfg or load_config()
    rs, cv = cfg["model"]["random_state"], cv_splitter(cfg)
    rows = {}
    for cw in (None, "balanced"):
        p = oof_proba(logistic(cw, rs), X, y, cv)
        rows[str(cw)] = {"PR-AUC (OOF)": average_precision_score(y, p),
                         "precision@20% (OOF)": metrics.precision_at_k(y, p, 0.2)}
    t = pd.DataFrame(rows).T
    best = t["PR-AUC (OOF)"].idxmax()
    return best, t


def odds_ratios(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Odds ratios from a scaled logistic regression, in interpretable units.

    0/1 features: odds ratio for 0 → 1 (yes vs no).
    Numeric features: odds ratio per 1 standard deviation (SD shown).
    """
    scale = pipe.named_steps["scale"]
    coef = pipe.named_steps["lr"].coef_[0]
    sd = pd.Series(scale.scale_, index=X.columns)
    binary = X.isin([0, 1]).all()
    per_unit = np.where(binary, coef / sd.values, coef)
    t = pd.DataFrame({"feature": X.columns, "label": [plain_label(c) for c in X.columns],
                      "binary": binary.values, "sd": sd.values,
                      "odds_ratio": np.exp(per_unit)})
    t["unit"] = np.where(t["binary"], "yes vs no", "per 1 SD")
    t["abs_log_or"] = np.abs(np.log(t["odds_ratio"]))
    return t.sort_values("abs_log_or", ascending=False).drop(columns="abs_log_or")


def tune_lightgbm(X, y, cfg: dict | None = None):
    """Light Optuna search (≤ lgbm_trials) maximising out-of-fold PR-AUC."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = cfg or load_config()
    rs, cv = cfg["model"]["random_state"], cv_splitter(cfg)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 64),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        p = oof_proba(lightgbm(rs, **params), X, y, cv)
        return average_precision_score(y, p)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=cfg["model"]["lgbm_trials"])
    return study.best_params, study


def compare_a_b(X, y, est_a, est_b, cfg: dict | None = None) -> tuple[str, pd.DataFrame]:
    """Out-of-fold comparison on lift in the top 2 deciles (top 20%).

    B wins only if its top-20% precision beats A's by >= min_gain_pp_top20.
    """
    cfg = cfg or load_config()
    cv = cv_splitter(cfg)
    rows = {}
    for name, est in (("A: logistic", est_a), ("B: LightGBM", est_b)):
        p = oof_proba(est, X, y, cv)
        rows[name] = {"precision@20% (OOF)": metrics.precision_at_k(y, p, 0.2),
                      "lift@20% (OOF)": metrics.lift_at_k(y, p, 0.2),
                      "PR-AUC (OOF)": average_precision_score(y, p)}
    t = pd.DataFrame(rows).T
    gain_pp = (t.loc["B: LightGBM", "precision@20% (OOF)"]
               - t.loc["A: logistic", "precision@20% (OOF)"]) * 100
    t.attrs["gain_pp"] = gain_pp
    winner = "B" if gain_pp >= cfg["model"]["min_gain_pp_top20"] else "A"
    return winner, t


# ---------------------------------------------------------------- calibration
def choose_calibration(est, X, y, cfg: dict | None = None):
    """Compare sigmoid vs isotonic (each fitted with inner 5-fold CV) by
    out-of-fold Brier score. Returns (method, table, oof calibrated probs)."""
    cfg = cfg or load_config()
    cv = cv_splitter(cfg)
    rows, oofs = {}, {}
    for method in ("sigmoid", "isotonic"):
        cal = CalibratedClassifierCV(clone(est), method=method, cv=cfg["model"]["cv_folds"])
        p = oof_proba(cal, X, y, cv)
        oofs[method] = p
        rows[method] = {"Brier (OOF)": brier_score_loss(y, p),
                        "mean predicted": p.mean(), "actual rate": float(np.mean(y))}
    t = pd.DataFrame(rows).T
    best = t["Brier (OOF)"].idxmin()
    return best, t, oofs[best]


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed conversion per predicted-probability decile."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    return d.groupby("bin").agg(mean_predicted=("p", "mean"), observed=("y", "mean"),
                                leads=("y", "size"))


# ---------------------------------------------------------------- score + bands
def to_score(p) -> np.ndarray:
    """Lead score 0–100 = calibrated probability × 100."""
    return np.round(np.asarray(p, dtype=float) * 100, 1)


def band_cutoffs(scores, hot: float = 0.20, warm: float = 0.30) -> dict:
    """Score cutoffs from deciles: Hot = top `hot` share, Warm = next `warm`."""
    s = np.asarray(scores, dtype=float)
    return {"hot": float(np.quantile(s, 1 - hot)),
            "warm": float(np.quantile(s, 1 - hot - warm))}


def assign_band(scores, cutoffs: dict) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.where(s >= cutoffs["hot"], "Hot",
                    np.where(s >= cutoffs["warm"], "Warm", "Cold"))


# ---------------------------------------------------------------- plain English
RAW_LABELS = {
    "Total Time Spent on Website": "Time on website",
    "TotalVisits": "Website visits",
    "Page Views Per Visit": "Pages per visit",
    "Lead Origin": "Lead origin",
    "Lead Source": "Lead source",
    "What is your current occupation": "Occupation",
    "What matters most to you in choosing a course": "Main motivation",
    "Specialization": "Specialization",
    "City": "City",
    "Country": "Country",
    "Do Not Email": "Opted out of email",
    "A free copy of Mastering The Interview": "Asked for free interview e-book",
}


def raw_column(feature: str) -> str:
    if "__is_missing" in feature:
        return feature.split("__is_missing")[0]
    return feature.split("=", 1)[0]


def plain_label(feature: str) -> str:
    col = raw_column(feature)
    label = RAW_LABELS.get(col, col)
    if "__is_missing" in feature:
        return f"{label} not recorded"
    if "=" in feature:
        lvl = feature.split("=", 1)[1]
        if lvl == MISSING:
            return f"{label} left blank"
        if lvl == OTHER:
            return f"{label}: other / rare"
        return f"{label}: {lvl}"
    return label


def describe_value(col: str, value) -> str:
    """Human-readable raw value for a driver line."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "left blank"
    if col == "Total Time Spent on Website":
        return f"{float(value) / 60:.0f} min"
    if col in ("TotalVisits",):
        return f"{float(value):.0f}"
    if col == "Page Views Per Visit":
        return f"{float(value):.1f}"
    return str(value)


# ---------------------------------------------------------------- SHAP
def shap_values(base_model, X_background: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    """SHAP values (log-odds contribution per feature) for the uncalibrated model."""
    import shap
    steps = base_model.named_steps
    if "lr" in steps:
        scale, lr = steps["scale"], steps["lr"]
        bg = pd.DataFrame(scale.transform(X_background), columns=X.columns)
        xs = pd.DataFrame(scale.transform(X), columns=X.columns)
        expl = shap.LinearExplainer(lr, shap.maskers.Independent(bg, max_samples=len(bg)))
        return np.asarray(expl.shap_values(xs))
    expl = shap.TreeExplainer(steps["lgbm"])
    sv = expl.shap_values(_to_array(X))
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    return sv[..., 1] if sv.ndim == 3 else sv


def global_drivers(sv: np.ndarray, X: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Top features by mean |SHAP|, with plain-English label and direction
    ('raises' if higher values / 'yes' push the score up)."""
    imp = np.abs(sv).mean(axis=0)
    rows = []
    for j, f in enumerate(X.columns):
        x = X[f].to_numpy(dtype=float)
        c = np.corrcoef(x, sv[:, j])[0, 1] if x.std() > 0 and sv[:, j].std() > 0 else 0.0
        rows.append({"feature": f, "label": plain_label(f), "importance": imp[j],
                     "direction": "raises" if c > 0 else "lowers"})
    return (pd.DataFrame(rows).sort_values("importance", ascending=False)
            .head(top).reset_index(drop=True))


def group_by_raw(sv: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum SHAP values of all encoded features that come from the same raw column
    (one-hot levels, is_missing flags), so drivers read as e.g. 'Lead origin'."""
    df = pd.DataFrame(sv, columns=feature_names)
    return df.T.groupby([raw_column(f) for f in feature_names]).sum().T


def top_drivers(grouped: pd.DataFrame, raw: pd.DataFrame, n: int = 3) -> list[list[dict]]:
    """Per lead: the n raw columns with the largest |SHAP|, with value + direction."""
    out = []
    g = grouped.to_numpy()
    cols = list(grouped.columns)
    for i in range(len(grouped)):
        idx = np.argsort(-np.abs(g[i]))[:n]
        lead = []
        for j in idx:
            col = cols[j]
            val = raw.iloc[i][col] if col in raw.columns else None
            lead.append({"feature": col, "label": RAW_LABELS.get(col, col),
                         "value": describe_value(col, val),
                         "effect": "raises" if g[i, j] > 0 else "lowers",
                         "shap": round(float(g[i, j]), 4)})
        out.append(lead)
    return out


def format_drivers(drivers: list[dict]) -> str:
    arrow = {"raises": "↑", "lowers": "↓"}
    return "; ".join(f"{d['label']}: {d['value']} {arrow[d['effect']]}" for d in drivers)


# ---------------------------------------------------------------- scorer artifact
@dataclass
class LeadScorer:
    """Everything needed to score new leads: cleaner → calibrated model → bands."""
    cleaner: LeadCleaner
    calibrated: object
    base_model: object
    model_name: str
    calibration: str
    cutoffs: dict
    background: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrated.predict_proba(self.cleaner.transform(df))[:, 1]

    def score(self, df: pd.DataFrame, n_drivers: int = 3) -> pd.DataFrame:
        X = self.cleaner.transform(df)
        p = self.calibrated.predict_proba(X)[:, 1]
        s = to_score(p)
        sv = shap_values(self.base_model, self.background, X)
        grouped = group_by_raw(sv, list(X.columns))
        drivers = top_drivers(grouped, df.reset_index(drop=True), n_drivers)
        return pd.DataFrame({"probability": p, "score": s,
                             "band": assign_band(s, self.cutoffs),
                             "drivers": drivers,
                             "drivers_text": [format_drivers(d) for d in drivers]},
                            index=df.index)
