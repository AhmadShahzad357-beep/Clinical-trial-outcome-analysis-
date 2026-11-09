"""
FastAPI dashboard for the clinical trial outcome analysis.

Run from the project folder:
    uvicorn app.main:app --reload
then open http://127.0.0.1:8000     (interactive API docs: http://127.0.0.1:8000/docs)

The dashboard reads the tables produced by the analysis scripts (results/tables, data/processed) and serves a
prediction endpoint backed by src/predictor.py. Nothing is recomputed from the test set.
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from lifelines import KaplanMeierFitter
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import predictor as P  # noqa: E402

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PROCESSED = ROOT / "data" / "processed"
STATIC = Path(__file__).resolve().parent / "static"

CONFIG = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))

app = FastAPI(
    title="Clinical Trial Outcome Analysis",
    description="Treatment effects, survival prediction and heterogeneity validation on ACTG 175 and GBSG2.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/figures", StaticFiles(directory=FIGURES), name="figures")

predictor = P.Predictor()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def records(df):
    return json.loads(df.to_json(orient="records"))


def split_pm(s):
    """'0.6843 +- 0.0144' -> (0.6843, 0.0144)"""
    a, b = str(s).split("+-")
    return float(a), float(b)


def split_range(s):
    a, b = str(s).split(" - ")
    return float(a), float(b)


@lru_cache(maxsize=1)
def km_curves():
    out = {}
    a = pd.read_csv(PROCESSED / "actg175_clean.csv")
    names = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+zal", 3: "ddI"}
    grid = list(range(0, 1201, 20))
    actg = {}
    for k, nm in names.items():
        s = a[a["arms"] == k]
        km = KaplanMeierFitter().fit(s["days"], s["cens"])
        actg[nm] = {"n": int(len(s)), "survival": [round(float(km.predict(t)), 4) for t in grid]}
    out["actg175"] = {"times": grid, "groups": actg, "horizon": 900, "ylabel": "Event-free probability"}

    g = pd.read_csv(ROOT / "data" / "raw" / "GBSG2_original.csv")
    grid2 = list(range(0, 2401, 40))
    gb = {}
    for lab, nm in (("no", "No hormonal therapy"), ("yes", "Hormonal therapy")):
        s = g[g["horTh"] == lab]
        km = KaplanMeierFitter().fit(s["time"], s["cens"])
        gb[nm] = {"n": int(len(s)), "survival": [round(float(km.predict(t)), 4) for t in grid2]}
    out["gbsg2"] = {"times": grid2, "groups": gb, "horizon": 1460, "ylabel": "Recurrence-free probability"}
    return out


# ----------------------------------------------------------------------
# Request models (validation = realistic clinical ranges)
# ----------------------------------------------------------------------
class ActgPatient(BaseModel):
    age: float = Field(35, ge=12, le=90, description="years")
    wtkg: float = Field(75, ge=30, le=170, description="weight, kg")
    karnof: int = Field(90, description="Karnofsky score: 70, 80, 90 or 100")
    cd40: float = Field(340, ge=0, le=1500, description="CD4 count at baseline")
    cd80: float = Field(890, ge=0, le=6000, description="CD8 count at baseline")
    preanti: float = Field(140, ge=0, le=3500, description="days of prior antiretroviral therapy")
    hemo: int = Field(0, ge=0, le=1)
    homo: int = Field(0, ge=0, le=1)
    drugs: int = Field(0, ge=0, le=1)
    z30: int = Field(0, ge=0, le=1, description="ZDV in the 30 days before study")
    race: int = Field(0, ge=0, le=1, description="0 white, 1 non-white")
    gender: int = Field(1, ge=0, le=1, description="0 female, 1 male")
    symptom: int = Field(0, ge=0, le=1)


class GbsgPatient(BaseModel):
    age: float = Field(55, ge=18, le=90)
    tsize: float = Field(25, ge=1, le=150, description="tumour size, mm")
    pnodes: float = Field(3, ge=0, le=60, description="positive lymph nodes")
    progrec: float = Field(30, ge=0, le=3000, description="progesterone receptor, fmol")
    estrec: float = Field(35, ge=0, le=1500, description="oestrogen receptor, fmol")
    post: int = Field(1, ge=0, le=1, description="1 postmenopausal")
    grade: int = Field(2, ge=1, le=3, description="tumour grade 1, 2 or 3")


# ----------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    url = os.environ.get("LIVE_DEMO_URL") or CONFIG.get("live_demo_url") or ""
    return {"live_demo_url": url, "repo_url": CONFIG.get("repo_url", ""), "seed": CONFIG.get("seed", 42)}


# ----------------------------------------------------------------------
# Analysis results
# ----------------------------------------------------------------------
@app.get("/api/overview")
def overview():
    mt = table("models_test")
    gm = table("gbsg2_models_test")
    eff = table("effects_summary")
    ge = table("gbsg2_effects").iloc[0]
    return {
        "trials": [
            {"name": "ACTG 175", "disease": "HIV", "patients": 2139, "events": 521, "event_rate": 24.4,
             "design": "Randomized, 4 arms", "horizon_days": 900,
             "best_hr_adj": float(eff["HR_adj"].min()), "best_rmst_days": float(eff["RMST_fark_900d_din"].max()),
             "cox_test_c": float(mt.loc[0, "C_harrell"]), "test_events": 104},
            {"name": "GBSG2", "disease": "Node-positive breast cancer", "patients": 686, "events": 299,
             "event_rate": 43.6, "design": "Hormonal therapy vs none (baseline imbalance)", "horizon_days": 1460,
             "best_hr_adj": float(ge["HR_adjusted"]), "best_rmst_days": float(ge["RMST_fark_crude"]),
             "cox_test_c": float(gm.loc[0, "C_harrell"]), "test_events": 60},
        ],
        "split": {"actg_train": 1711, "actg_test": 428, "gbsg_train": 548, "gbsg_test": 138},
    }


@app.get("/api/km")
def km():
    return km_curves()


@app.get("/api/effects")
def effects():
    eff = table("effects_summary")
    rows = []
    for r in eff.to_dict("records"):
        lo, hi = split_range(r["RMST_95%"])
        rlo, rhi = split_range(r["risk_fark_95%"])
        rows.append({"arm": r["arm"], "hr_unadj": r["HR_unadj"], "hr_adj": r["HR_adj"],
                     "event_free_900": r["event_free_900d_%"], "risk_diff": r["risk_fark_900d_pt"],
                     "risk_lo": rlo, "risk_hi": rhi, "nnt": r["NNT"],
                     "rmst": r["RMST_fark_900d_din"], "rmst_lo": lo, "rmst_hi": hi})
    cens = table("eda_censoring_by_arm")
    pair = table("effects_pairwise_logrank")
    sub = table("eda_forest_treat_subgroups")
    return {"arms": rows, "censoring": records(cens), "pairwise": records(pair), "subgroups": records(sub),
            "reference_event_free_900": 65.0}


@app.get("/api/models")
def models():
    def parse_cv(df):
        out = []
        for r in df.to_dict("records"):
            c, se = split_pm(r["C_harrell"])
            out.append({"model": r["model"], "c": c, "se": se})
        return out

    mt, gmt = table("models_test"), table("gbsg2_models_test")
    return {
        "actg": {"cv": parse_cv(table("models_cv")), "test": records(mt), "grid": records(table("models_grid")),
                 "chosen": json.load(open(PROCESSED / "models_choice.json"))["chosen"]},
        "gbsg": {"cv": parse_cv(table("gbsg2_models_cv")), "test": records(gmt),
                 "chosen": json.load(open(PROCESSED / "gbsg2_choice.json"))["chosen"]},
    }


@app.get("/api/heterogeneity")
def heterogeneity():
    val = table("validation_results")
    gval = table("gbsg2_validation")
    return {"actg": {"cate_summary": records(table("cate_summary")), "validation": records(val)},
            "gbsg": {"cate_summary": records(table("gbsg2_cate_summary")), "validation": records(gval)}}


@app.get("/api/figures")
def figures():
    return sorted(p.name for p in FIGURES.glob("*.png"))


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
@app.get("/api/predict/info")
def predict_info():
    return predictor.info()


@app.post("/api/predict/actg175")
def predict_actg(p: ActgPatient):
    if p.karnof not in (70, 80, 90, 100):
        raise HTTPException(422, "karnof must be 70, 80, 90 or 100")
    return predictor.predict_actg(p.model_dump())


@app.post("/api/predict/gbsg2")
def predict_gbsg(p: GbsgPatient):
    return predictor.predict_gbsg(p.model_dump())