"""
Prediction module (used by the FastAPI dashboard).

Two ridge-penalised Cox models, one per trial, trained on the TRAIN split only (the same split and the same
settings that models.py / gbsg2.py chose). The test split is never used here.

    ACTG 175 : feature set of models.py (raw features, continuous columns scaled, arm as 3 dummies), alpha = 10
    GBSG2    : feature set of gbsg2.py (log1p on skewed columns, continuous columns scaled, therapy flag), alpha = 10

Output of a prediction: the model-based probability of staying event-free (ACTG 175) or recurrence-free (GBSG2)
at the trial horizon, for every treatment option, plus the full predicted curve.

IMPORTANT: this is a research demonstration. The "difference between treatments" shown for one patient is the
Cox-model estimate. The validation stage of this project found NO reliable evidence that patient-level differences
are real, so it must not be read as personal treatment advice.

Run directly to (re)train and save both models:   python src/predictor.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "predictors.joblib"

ACTG_ARMS = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+zal", 3: "ddI"}
ACTG_TAU = 900
GBSG_ARMS = {0: "No hormonal therapy", 1: "Hormonal therapy"}
GBSG_TAU = 1460

ACTG_CONT = ["age", "wtkg", "preanti", "cd40", "cd80"]
ACTG_BINARY = ["hemo", "homo", "drugs", "z30", "race", "gender", "symptom"]
GBSG_CONT = ["age", "tsize", "pnodes", "progrec", "estrec"]
GBSG_SKEWED = ["tsize", "pnodes", "progrec", "estrec"]
GBSG_FEATS = GBSG_CONT + ["post", "g2", "g3"]

SEED = 42


def _cox(cont_cols, alpha=10.0):
    pre = ColumnTransformer([("sc", StandardScaler(), cont_cols)], remainder="passthrough")
    return make_pipeline(pre, CoxPHSurvivalAnalysis(alpha=alpha))


def _surv_at(fn, t):
    """Survival probability at time t (clipped to the last observed time of the step function)."""
    lo, hi = fn.domain
    return float(fn(min(max(t, lo), hi)))


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def _actg_columns():
    return [c for c in pd.read_csv(PROCESSED / "X_train_trees.csv", index_col="row").columns if c != "oprior"] + \
           ["arm_1", "arm_2", "arm_3"]


def _load_actg_train():
    X = pd.read_csv(PROCESSED / "X_train_trees.csv", index_col="row").drop(columns=["oprior"])
    y = pd.read_csv(PROCESSED / "y_train.csv", index_col="row")
    t = pd.read_csv(PROCESSED / "treatment_train.csv", index_col="row")
    for a in (1, 2, 3):
        X[f"arm_{a}"] = (t["arms"] == a).astype(int)
    ys = Surv.from_arrays(event=y["cens"].astype(bool).values, time=y["days"].values.astype(float))
    return X, ys


def _load_gbsg_train():
    raw = pd.read_csv(RAW / "GBSG2_original.csv")
    df = pd.DataFrame({
        "age": raw["age"], "tsize": raw["tsize"], "pnodes": raw["pnodes"], "progrec": raw["progrec"],
        "estrec": raw["estrec"], "post": (raw["menostat"] == "Post").astype(int),
        "g2": (raw["tgrade"] == "II").astype(int), "g3": (raw["tgrade"] == "III").astype(int),
        "treat": (raw["horTh"] == "yes").astype(int), "days": raw["time"].astype(float), "cens": raw["cens"].astype(int)})
    ids = pd.read_csv(PROCESSED / "gbsg2_split_ids.csv")
    train_rows = ids.loc[ids["set"] == "train", "row"].values
    d = df.iloc[train_rows].reset_index(drop=True)
    return d


def _gbsg_design(frame):
    Z = frame[GBSG_FEATS].copy()
    Z[GBSG_SKEWED] = np.log1p(Z[GBSG_SKEWED])
    Z["treat"] = frame["treat"].values
    return Z


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_all():
    # ACTG 175
    X, ys = _load_actg_train()
    actg = _cox(ACTG_CONT).fit(X, ys)
    actg_ranges = {c: [float(X[c].min()), float(X[c].max())] for c in ACTG_CONT}
    actg_risk = actg.predict(X)

    # GBSG2
    d = _load_gbsg_train()
    Z = _gbsg_design(d)
    ysg = Surv.from_arrays(event=d["cens"].astype(bool).values, time=d["days"].values)
    gbsg = _cox(GBSG_CONT).fit(Z, ysg)
    gbsg_ranges = {c: [float(d[c].min()), float(d[c].max())] for c in GBSG_CONT}
    gbsg_risk = gbsg.predict(Z)

    bundle = {
        "actg": {"model": actg, "columns": list(X.columns), "ranges": actg_ranges,
                 "train_risk": np.sort(actg_risk), "n_train": int(len(X)), "events_train": int(ys["event"].sum())},
        "gbsg": {"model": gbsg, "columns": list(Z.columns), "ranges": gbsg_ranges,
                 "train_risk": np.sort(gbsg_risk), "n_train": int(len(Z)), "events_train": int(ysg["event"].sum())},
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    return bundle


class Predictor:
    def __init__(self):
        if MODEL_FILE.exists():
            self.b = joblib.load(MODEL_FILE)
        else:
            self.b = train_all()

    # ---------------- shared ----------------
    @staticmethod
    def _curve(model, Xrow, tau, n_points=36):
        cox = model[-1]
        Xt = model[:-1].transform(Xrow)
        fn = cox.predict_survival_function(Xt)[0]
        grid = np.linspace(0, tau, n_points)
        return grid, np.array([_surv_at(fn, t) if t > 0 else 1.0 for t in grid]), _surv_at(fn, tau)

    @staticmethod
    def _percentile(sorted_risk, r):
        return float(np.searchsorted(sorted_risk, r) / len(sorted_risk) * 100)

    @staticmethod
    def _range_warnings(ranges, values):
        out = []
        for c, (lo, hi) in ranges.items():
            v = values[c]
            if v < lo or v > hi:
                out.append(f"{c} = {v:g} is outside the range seen in training ({lo:g} to {hi:g}); extrapolation.")
        return out

    # ---------------- ACTG 175 ----------------
    def predict_actg(self, p: dict):
        b = self.b["actg"]
        base = {c: float(p[c]) for c in ACTG_CONT}
        base.update({c: int(p[c]) for c in ACTG_BINARY})
        kar = int(p["karnof"])
        base["karnof_70_80"] = int(kar <= 80)
        base["karnof_90"] = int(kar == 90)
        results, curves = {}, {}
        for arm, name in ACTG_ARMS.items():
            row = dict(base)
            for a in (1, 2, 3):
                row[f"arm_{a}"] = int(arm == a)
            Xrow = pd.DataFrame([row])[b["columns"]]
            grid, s, s_tau = self._curve(b["model"], Xrow, ACTG_TAU)
            risk = float(b["model"].predict(Xrow)[0])
            results[name] = {"event_free_prob": s_tau, "risk_score": risk,
                             "risk_percentile": self._percentile(b["train_risk"], risk)}
            curves[name] = s.tolist()
        ref = results["ZDV"]["event_free_prob"]
        for name, r in results.items():
            r["difference_vs_ZDV_pts"] = (r["event_free_prob"] - ref) * 100
        return {"trial": "ACTG 175", "horizon_days": ACTG_TAU, "outcome": "event-free (no CD4 drop, AIDS or death)",
                "times": np.linspace(0, ACTG_TAU, 36).tolist(), "arms": results, "curves": curves,
                "warnings": self._range_warnings(b["ranges"], base)}

    # ---------------- GBSG2 ----------------
    def predict_gbsg(self, p: dict):
        b = self.b["gbsg"]
        base = {c: float(p[c]) for c in GBSG_CONT}
        base["post"] = int(p["post"])
        grade = int(p["grade"])
        base["g2"], base["g3"] = int(grade == 2), int(grade == 3)
        results, curves = {}, {}
        for arm, name in GBSG_ARMS.items():
            frame = pd.DataFrame([{**base, "treat": arm}])
            Z = _gbsg_design(frame)[b["columns"]]
            grid, s, s_tau = self._curve(b["model"], Z, GBSG_TAU)
            risk = float(b["model"].predict(Z)[0])
            results[name] = {"event_free_prob": s_tau, "risk_score": risk,
                             "risk_percentile": self._percentile(b["train_risk"], risk)}
            curves[name] = s.tolist()
        ref = results["No hormonal therapy"]["event_free_prob"]
        for name, r in results.items():
            r["difference_vs_no_therapy_pts"] = (r["event_free_prob"] - ref) * 100
        return {"trial": "GBSG2", "horizon_days": GBSG_TAU, "outcome": "recurrence-free survival",
                "times": np.linspace(0, GBSG_TAU, 36).tolist(), "arms": results, "curves": curves,
                "warnings": self._range_warnings(b["ranges"], base)}

    def info(self):
        return {k: {"n_train": v["n_train"], "events_train": v["events_train"], "ranges": v["ranges"]}
                for k, v in self.b.items()}


if __name__ == "__main__":
    bundle = train_all()
    print(f"Saved {MODEL_FILE}")
    for k, v in bundle.items():
        print(f"  {k}: trained on {v['n_train']} patients, {v['events_train']} events")
    pr = Predictor()
    demo = dict(age=35, wtkg=75, hemo=0, homo=1, drugs=0, karnof=90, z30=1, preanti=140, race=0, gender=1,
                symptom=0, cd40=340, cd80=890)
    out = pr.predict_actg(demo)
    print(json.dumps({k: round(v["event_free_prob"], 3) for k, v in out["arms"].items()}))
    sys.exit(0)