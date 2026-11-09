"""
GBSG2 - ACTG175 wala hi protocol, doosre trial par (sab kuch is ek file mein)
Chalane ka tareeqa (project folder se):
    python src/gbsg2.py
Jaldi test (chhota version):  ... gbsg2.py --fast        Window na khule:  ... gbsg2.py --no-show
(Pehle models.py, cate.py, validation.py ka kaam ho chuka ho: yeh unke functions dobara istemal karta hai.)

ZAROORI FARQ (ACTG175 se): GBSG2 mein 'horTh' (hormonal therapy) purely random nahi hai. Dono groups shuru mein alag hain
(older, postmenopausal mareezon ko zyada milti hai). Isliye yahan:
  - asar 'adjusted' naapa jata hai: propensity score (kis ko therapy milne ka imkaan) wala AIPW tareeqa,
  - DR pseudo-outcome mein propensity score estimated (cross-fit) hota hai, constant nahi,
  - global null mein labels propensity ke hisaab se barabar groups ke ANDAR shuffle hote hain.
Horizon: TAU = 1460 din (4 saal; is waqt tak ~229 mareez abhi follow-up mein hain), sensitivity 1095 din.

Tarteeb:
  Step 1  Data tayyari aur checks            Step 2  EDA (balance, overlap, KM)
  Step 3  Train/Test split + features        Step 4  Prediction models (Cox, RSF, GBM)
  Step 5  Treatment ka asar (crude vs adjusted)   Step 6  CATE (5 tareeqe)
  Step 7  Validation (constant-effect, slope, null, test)     Step 8  Graph aur save
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M                                   # ACTG175 wale functions (generic hisse)
import cate as C
import validation as V
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
FIG_DIR, TAB_DIR = ROOT / "results" / "figures", ROOT / "results" / "tables"
FAST = "--fast" in sys.argv
SHOW = "--no-show" not in sys.argv

SEED = 42
TAU, TAU2 = 1460, 1095
N_FOLDS = 5
N_TREES = 80 if FAST else 200
N_BOOT = 100 if FAST else 500
N_PERM = 8 if FAST else 40
CONT = ["age", "tsize", "pnodes", "progrec", "estrec"]
SKEWED = ["tsize", "pnodes", "progrec", "estrec"]
FEATS = CONT + ["post", "g2", "g3"]
ARM = {0: "Hormonal therapy nahi", 1: "Hormonal therapy"}
COL = {"Cox": "#2F5D8A", "RSF": "#3A9D8F", "GBM": "#E4893B", "Cox_base": "#8E5BA8"}
MC = {"S-Cox": "#9AA5B1", "T-Cox": "#2F5D8A", "T-RSF": "#3A9D8F", "DR-Ridge": "#C8553D", "DR-RF": "#8E5BA8"}
METHODS = list(MC)

plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})

# models.py / cate.py ke generic functions ko GBSG2 ke liye set karna
M.TAU, M.TIMES = TAU, np.linspace(365, TAU, 12)
M.CONT_COLS, M.COX_DROP, M.N_TREES, M.CV_REPEATS = CONT, [], N_TREES, (1 if FAST else 2)
C.TAU, C.CONT_COLS, C.COX_DROP, C.N_TREES = TAU, CONT, [], N_TREES


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


def logx(X):
    Z = X.copy(); Z[SKEWED] = np.log1p(Z[SKEWED]); return Z


def surv(days, cens):
    return Surv.from_arrays(event=np.asarray(cens).astype(bool), time=np.asarray(days, float))


# ==================================================================
# STEP 1: DATA
# ==================================================================
t0 = time.time()
head("STEP 1: Data tayyari aur checks")
raw = pd.read_csv(RAW / "GBSG2_original.csv")
df = pd.DataFrame({
    "age": raw["age"], "tsize": raw["tsize"], "pnodes": raw["pnodes"], "progrec": raw["progrec"], "estrec": raw["estrec"],
    "post": (raw["menostat"] == "Post").astype(int),
    "g2": (raw["tgrade"] == "II").astype(int), "g3": (raw["tgrade"] == "III").astype(int),      # grade I reference
    "treat": (raw["horTh"] == "yes").astype(int), "days": raw["time"].astype(float), "cens": raw["cens"].astype(int)})
n_all = len(df)
print(f"{n_all} mareez, {int(df.cens.sum())} events ({df.cens.mean()*100:.1f}%), missing {int(df.isna().sum().sum())}")
print(f"Treatment (horTh = yes): {int(df.treat.sum())} mareez ({df.treat.mean()*100:.0f}%). Follow-up: median {df.days.median():.0f}, max {df.days.max():.0f} din")
print(f"Features: {FEATS} | outcome: days, cens (recurrence-free survival)")
print(f"Horizon TAU = {TAU} din: us waqt tak abhi follow-up mein: {int((df.days >= TAU).sum())} mareez\n")
a_all = df["treat"].values
Xall = df[FEATS]

# ==================================================================
# STEP 2: EDA
# ==================================================================
head("STEP 2: EDA (balance, overlap, Kaplan-Meier)")
sk = pd.Series({c: skew(df[c]) for c in CONT})
print("Skewness:", ", ".join(f"{k} {v:.2f}" for k, v in sk.items()), "(1 se zyada = lambi dum)")
smd = {}
for c in FEATS:
    t_, u_ = df.loc[df.treat == 1, c], df.loc[df.treat == 0, c]
    smd[c] = (t_.mean() - u_.mean()) / np.sqrt((t_.var() + u_.var()) / 2)
smd = pd.Series(smd)
print("Baseline balance (SMD, 0.1 se kam = barabar):", ", ".join(f"{k} {v:+.2f}" for k, v in smd.items()))
bad = smd[smd.abs() > 0.1].index.tolist()
print(f"   0.1 se upar ({len(bad)}): {bad}  ->  groups shuru mein ALAG hain: yeh randomized comparison ki tarah nahi, adjusted analysis chahiye.")
Xlog_all = logx(Xall)
ps_model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
ps_eda = np.zeros(n_all)
for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xlog_all, a_all):
    ps_eda[va] = ps_model.fit(Xlog_all.iloc[tr], a_all[tr]).predict_proba(Xlog_all.iloc[va])[:, 1]
print(f"Propensity (therapy milne ka imkaan, sirf covariates se): {ps_eda.min():.2f} se {ps_eda.max():.2f}. "
      f"0.1 se kam ya 0.9 se zyada: {int(((ps_eda < .1) | (ps_eda > .9)).sum())} mareez (overlap theek hai)")
lr = logrank_test(df.days[df.treat == 1], df.days[df.treat == 0], df.cens[df.treat == 1], df.cens[df.treat == 0])
km = {g: KaplanMeierFitter().fit(df.days[df.treat == g], df.cens[df.treat == g], label=f"{ARM[g]} (n={int((df.treat == g).sum())})") for g in (0, 1)}
print(f"Kaplan-Meier: {TAU} din par event se bache hue: " + ", ".join(f"{ARM[g]} {float(km[g].predict(TAU))*100:.1f}%" for g in (0, 1))
      + f" | log-rank p = {lr.p_value:.4f}\n")

# ==================================================================
# STEP 3: TRAIN/TEST SPLIT + FEATURES
# ==================================================================
head("STEP 3: Train/Test split aur features")
strat = (df.treat.astype(str) + "_" + df.cens.astype(str)).values
tr_idx, te_idx = train_test_split(np.arange(n_all), test_size=0.2, random_state=SEED, stratify=strat)
tr_idx, te_idx = np.sort(tr_idx), np.sort(te_idx)
pd.DataFrame({"row": np.r_[tr_idx, te_idx], "set": ["train"] * len(tr_idx) + ["test"] * len(te_idx)}).sort_values("row").to_csv(
    PROCESSED / "gbsg2_split_ids.csv", index=False)
dtr, dte = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
ytr, yte = surv(dtr.days, dtr.cens), surv(dte.days, dte.cens)
a_tr, a_te = dtr.treat.values, dte.treat.values
ntr, nte = len(dtr), len(dte)
print(f"Train: {ntr} mareez, {int(dtr.cens.sum())} events | Test: {nte} mareez, {int(dte.cens.sum())} events (sirf akhir mein khulega)")
print("Test mein sirf itne events hain, is liye test ke nateeje ka daira chaura hoga.")
strat_tr = (dtr.treat.astype(str) + "_" + dtr.cens.astype(str)).values
Xtr_raw, Xte_raw = dtr[FEATS], dte[FEATS]

# Cox ke liye log1p behtar hai ya nahi? Train ke CV se (saada variant jo best ke 1 SE ke andar ho)
cvA, _ = M.cv_model("Cox", {"alpha": 1.0}, Xtr_raw.assign(treat=a_tr), ytr, strat_tr)
cvB, _ = M.cv_model("Cox", {"alpha": 1.0}, logx(Xtr_raw).assign(treat=a_tr), ytr, strat_tr)
mA, seA = M.mean_se(cvA["C_harrell"]); mB, seB = M.mean_se(cvB["C_harrell"])
best_m, best_v = max((mA, "A"), (mB, "B"))
best_se = seA if best_v == "A" else seB
use_log = (best_v == "B") and not (mA >= mB - best_se)          # A tabhi chunte hain jab woh best ke 1 SE ke andar ho
print(f"\nCox ke liye skewed features par log1p: A (sirf scaling) C-index {mA:.4f}+-{seA:.4f} | B (log1p) {mB:.4f}+-{seB:.4f}")
print(f"Chuna gaya: {'B (log1p)' if use_log else 'A (sirf scaling)'}  (saada variant jo sab se achhe ke 1 SE ke andar ho)")
Xc_tr, Xc_te = (logx(Xtr_raw), logx(Xte_raw)) if use_log else (Xtr_raw, Xte_raw)      # Cox / ridge ke features
Xt_tr, Xt_te = Xtr_raw, Xte_raw                                                          # trees ke features (raw)
print(f"Trees ke liye raw features. Scaling sirf Cox/ridge mein, har fold ke andar. [{time.time()-t0:.0f} sec]\n")

# ==================================================================
# STEP 4: PREDICTION MODELS
# ==================================================================
head("STEP 4: Prediction models (treatment feature ke tor par shamil)")
grids = {"Cox": [dict(alpha=a) for a in ((10, 1) if FAST else (10, 1, 0.1))],
         "RSF": [dict(min_samples_leaf=l, max_features=f) for l, f in (((30, "sqrt"), (15, "sqrt")) if FAST else ((30, "sqrt"), (30, 0.6), (15, "sqrt"), (15, 0.6)))],
         "GBM": [dict(n_estimators=n, max_depth=d) for n, d in (((100, 1), (100, 2)) if FAST else ((100, 1), (200, 1), (100, 2), (200, 2)))]}
LONG = {"Cox": "Cox (ridge)", "RSF": "Random Survival Forest", "GBM": "Gradient Boosting"}
XP = {"Cox": Xc_tr.assign(treat=a_tr), "RSF": Xt_tr.assign(treat=a_tr), "GBM": Xt_tr.assign(treat=a_tr)}
XPte = {"Cox": Xc_te.assign(treat=a_te), "RSF": Xt_te.assign(treat=a_te), "GBM": Xt_te.assign(treat=a_te)}
chosen, grid_res, grid_oof, cidx = {}, {}, {}, {}
for kind, grid in grids.items():
    res, oofs = [], []
    for p in grid:
        r, o = M.cv_model(kind, p, XP[kind], ytr, strat_tr, want_oof=True)
        res.append(r); oofs.append(o)
    idx, bst = M.pick_simplest(res)
    chosen[kind], cidx[kind], grid_res[kind], grid_oof[kind] = grid[idx], idx, res, oofs
    print(f"{LONG[kind]}:")
    for i, p in enumerate(grid):
        m_, se_ = M.mean_se(res[i]["C_harrell"])
        print(f"   {M.fmt_params(p):42s} C-index {m_:.4f} +- {se_:.4f}" + ("   <-- chuna gaya" if i == idx else ("   (sab se achha)" if i == bst else "")))
cv_runs = {k: grid_res[k][cidx[k]] for k in grids}
oof_pred = {k: grid_oof[k][cidx[k]] for k in grids}
ref, _ = M.cv_model("Cox", chosen["Cox"], Xc_tr, ytr, strat_tr)                           # Cox, treatment ke baghair
cv_runs["Cox_base"] = ref
names = {**LONG, "Cox_base": "Cox, treatment ke baghair (reference)"}
tab = []
for k, r in cv_runs.items():
    row = {"model": names[k]}
    for mt in ["C_harrell", "C_uno", "AUC_900", "IBS_skill"]:
        m_, se_ = M.mean_se(r[mt]); row[mt.replace("900", str(TAU))] = f"{m_:.4f} +- {se_:.4f}"
    tab.append(row)
print("\nTrain ka cross-validation:"); print(pd.DataFrame(tab).to_string(index=False))
means = {k: cv_runs[k]["C_harrell"].mean() for k in grids}
print(f"Treatment shamil karne se Cox ka C-index {means['Cox'] - ref['C_harrell'].mean():+.4f} badla.")

test_rows, test_risk, fitted = [], {}, {}
for kind in grids:
    m = M.make_model(kind, chosen[kind]).fit(XP[kind], ytr)
    r, risk, _ = M.evaluate(m, XP[kind], ytr, XPte[kind], yte)
    fitted[kind], test_risk[kind] = m, risk
    test_rows.append({"kind": kind, "model": LONG[kind], **r})
rng = np.random.default_rng(SEED)
boots = {k: [] for k in grids}
for _ in range(N_BOOT):
    ix = rng.integers(0, nte, nte)
    if yte["event"][ix].sum() < 5:
        continue
    for k in grids:
        boots[k].append(concordance_index_censored(yte["event"][ix], yte["time"][ix], test_risk[k][ix])[0])
boots = {k: np.array(v) for k, v in boots.items()}
tdf = pd.DataFrame(test_rows)
tdf["C_95%"] = [f"{np.percentile(boots[k], 2.5):.3f} - {np.percentile(boots[k], 97.5):.3f}" for k in tdf["kind"]]
print("\nTEST (ek baar):")
print(tdf[["model", "C_harrell", "C_95%", "C_uno", "AUC_900", "IBS_skill"]].round(4).rename(columns={"AUC_900": f"AUC_{TAU}"}).to_string(index=False))
for k in ("RSF", "GBM"):
    d_ = boots[k] - boots["Cox"]
    print(f"   {LONG[k]} minus Cox (test C-index): {d_.mean():+.4f}  ({np.percentile(d_, 2.5):+.4f} se {np.percentile(d_, 97.5):+.4f})")
for k in grids:
    print(f"   {LONG[k]:24s} CV {cv_runs[k]['C_harrell'].mean():.4f} | Test {tdf.loc[tdf.kind == k, 'C_harrell'].iloc[0]:.4f}")
print(f"[{time.time()-t0:.0f} sec]\n")
tdf.drop(columns="kind").round(4).rename(columns={"AUC_900": f"AUC_{TAU}"}).to_csv(TAB_DIR / "gbsg2_models_test.csv", index=False)
pd.DataFrame(tab).to_csv(TAB_DIR / "gbsg2_models_cv.csv", index=False)
C.COX_ALPHA, C.RSF_LEAF, C.RSF_MAXF = chosen["Cox"]["alpha"], chosen["RSF"]["min_samples_leaf"], chosen["RSF"]["max_features"]


# ==================================================================
# NUISANCE (outcome model mu aur propensity ps) aur DR pseudo-outcome (propensity ke saath)
# psi = mu1 - mu0 + (A/e)(Y* - mu1) - ((1-A)/(1-e))(Y* - mu0)   [e = therapy milne ka estimated imkaan]
# ==================================================================
def nuisance(Xfit, yfit, afit, Xnew):
    out = {}
    for g in (0, 1):
        idx = np.where(afit == g)[0]
        out[g] = C.SurvModel("cox", len(idx), len(afit)).fit(Xfit.iloc[idx], yfit[idx]).rmst(Xnew, tau=TAU)
    ps = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xfit, afit).predict_proba(Xnew)[:, 1]
    return out[1], out[0], np.clip(ps, 0.05, 0.95)


def psi_from(a, d, mu1, mu0, ps):
    Y = np.zeros(len(a)); days, ev = d["days"].values, d["cens"].values
    for g in (0, 1):                                   # censoring ka KM har arm ka apna
        m_ = a == g
        Y[m_] = C.ipcw_outcome(days[m_], ev[m_], TAU, days[m_], ev[m_])
    return mu1 - mu0 + (a / ps) * (Y - mu1) - ((1 - a) / (1 - ps)) * (Y - mu0), Y


# ==================================================================
# STEP 5: TREATMENT KA ASAR (crude vs adjusted)
# ==================================================================
head("STEP 5: Treatment ka asar (crude vs adjusted)")
Xc_all = logx(Xall) if use_log else Xall
mu1_a, mu0_a, ps_a = np.zeros(n_all), np.zeros(n_all), np.zeros(n_all)
for tr, va in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED + 1).split(Xc_all, a_all):
    ys_ = surv(df.days.values[tr], df.cens.values[tr])
    mu1_a[va], mu0_a[va], ps_a[va] = nuisance(Xc_all.iloc[tr], ys_, a_all[tr], Xc_all.iloc[va])
psi_all, _ = psi_from(a_all, df[["days", "cens"]], mu1_a, mu0_a, ps_a)
ate, ate_se = psi_all.mean(), psi_all.std(ddof=1) / np.sqrt(n_all)
gcomp = float((mu1_a - mu0_a).mean())                      # sirf Cox outcome model se (propensity ke baghair) adjusted fark
crude = C.km_rmst(df.days[df.treat == 1].values, df.cens[df.treat == 1].values, tau=TAU) - \
        C.km_rmst(df.days[df.treat == 0].values, df.cens[df.treat == 0].values, tau=TAU)
rg = np.random.default_rng(SEED)
bs = []
for _ in range(300 if not FAST else 80):
    cb = []
    for g in (0, 1):
        s = df[df.treat == g]; s = s.iloc[rg.integers(0, len(s), len(s))]
        cb.append(C.km_rmst(s.days.values, s.cens.values, tau=TAU))
    bs.append(cb[1] - cb[0])
c_lo, c_hi = np.percentile(bs, [2.5, 97.5])
Xlc = logx(df[FEATS]).assign(days=df.days, cens=df.cens, treat=df.treat)
cph_c = CoxPHFitter().fit(df[["days", "cens", "treat"]], "days", "cens")
cph_a = CoxPHFitter().fit(Xlc, "days", "cens")
hc, ha = cph_c.summary.loc["treat"], cph_a.summary.loc["treat"]
ph_p = float(proportional_hazard_test(cph_a, Xlc, time_transform="rank").summary.loc["treat", "p"])
print(f"RMST fark ({TAU} din tak, therapy vs nahi):")
print(f"   crude (sirf KM)              : {crude:+.1f} din  (95% daira {c_lo:+.1f} se {c_hi:+.1f})")
print(f"   adjusted (AIPW, propensity)  : {ate:+.1f} din  (95% daira {ate-1.96*ate_se:+.1f} se {ate+1.96*ate_se:+.1f})")
print(f"   adjusted (sirf Cox model se)  : {gcomp:+.1f} din  (daira nahi; AIPW se zyada mustaqil magar model par nirbhar)")
print(f"Hazard ratio: crude {np.exp(hc['coef']):.2f} ({hc['exp(coef) lower 95%']:.2f} - {hc['exp(coef) upper 95%']:.2f}, p {hc['p']:.4f}) | "
      f"adjusted {np.exp(ha['coef']):.2f} ({ha['exp(coef) lower 95%']:.2f} - {ha['exp(coef) upper 95%']:.2f}, p {ha['p']:.4f})")
print(f"Proportional hazards check (therapy ka p): {ph_p:.3f}  ({'qaida theek' if ph_p >= 0.05 else 'HR waqt ke saath badalta hai'})")
print(f"Matlab: crude aur adjusted farq ek taraf hai ya nahi, dekhein. Randomized na hone ki wajah se adjusted par zyada bharosa.\n")
pd.DataFrame([{"horizon_din": TAU, "RMST_fark_crude": round(crude, 1), "crude_lo": round(c_lo, 1), "crude_hi": round(c_hi, 1),
               "RMST_fark_adjusted": round(ate, 1), "RMST_fark_Cox_model": round(gcomp, 1), "adj_lo": round(ate - 1.96 * ate_se, 1), "adj_hi": round(ate + 1.96 * ate_se, 1),
               "HR_crude": round(float(np.exp(hc["coef"])), 3), "HR_adjusted": round(float(np.exp(ha["coef"])), 3)}]).to_csv(
    TAB_DIR / "gbsg2_effects.csv", index=False)

# ==================================================================
# STEP 6: CATE (5 tareeqe, out-of-fold train par, test ke liye train se)
# ==================================================================
head("STEP 6: CATE (mareez-ba-mareez asar)")
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xc_tr, strat_tr))
mu = {k: {g: np.zeros(ntr) for g in (0, 1)} for k in ("cox", "rsf", "scox")}
ps_oof = np.zeros(ntr)
for tr, va in folds:
    for kind, Xf in (("cox", Xc_tr), ("rsf", Xt_tr)):
        for g in (0, 1):
            idx = tr[a_tr[tr] == g]
            mu[kind][g][va] = C.SurvModel(kind, len(idx), ntr).fit(Xf.iloc[idx], ytr[idx]).rmst(Xf.iloc[va], tau=TAU)
    ms = C.SurvModel("cox", len(tr), ntr, alpha=C.ALPHA_S).fit(Xc_tr.iloc[tr].assign(treat=a_tr[tr]), ytr[tr])
    for g in (0, 1):
        mu["scox"][g][va] = ms.rmst(Xc_tr.iloc[va].assign(treat=g), tau=TAU)
    ps_oof[va] = np.clip(make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(
        Xc_tr.iloc[tr], a_tr[tr]).predict_proba(Xc_tr.iloc[va])[:, 1], 0.05, 0.95)
psi_tr, _ = psi_from(a_tr, dtr[["days", "cens"]], mu["cox"][1], mu["cox"][0], ps_oof)

oof = {m: np.zeros(ntr) for m in METHODS}
for key, nm in (("scox", "S-Cox"), ("cox", "T-Cox"), ("rsf", "T-RSF")):
    oof[nm] = mu[key][1] - mu[key][0]
for tr, va in folds:
    sc = StandardScaler().fit(Xc_tr.iloc[tr])
    oof["DR-Ridge"][va] = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc.transform(Xc_tr.iloc[tr]), psi_tr[tr]).predict(sc.transform(Xc_tr.iloc[va]))
    oof["DR-RF"][va] = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=30, max_features=0.5, n_jobs=-1,
                                             random_state=SEED).fit(Xt_tr.iloc[tr], psi_tr[tr]).predict(Xt_tr.iloc[va])
# test ke liye (poore train par seekh kar)
mu1_t, mu0_t, ps_t = nuisance(Xc_tr, ytr, a_tr, Xc_te)
test_c = {}
for kind, Xf, Xn, nm in (("cox", Xc_tr, Xc_te, "T-Cox"), ("rsf", Xt_tr, Xt_te, "T-RSF")):
    r_ = {g: C.SurvModel(kind, int((a_tr == g).sum()), ntr).fit(Xf.iloc[a_tr == g], ytr[a_tr == g]).rmst(Xn, tau=TAU) for g in (0, 1)}
    test_c[nm] = r_[1] - r_[0]
ms = C.SurvModel("cox", ntr, ntr, alpha=C.ALPHA_S).fit(Xc_tr.assign(treat=a_tr), ytr)
test_c["S-Cox"] = ms.rmst(Xc_te.assign(treat=1), tau=TAU) - ms.rmst(Xc_te.assign(treat=0), tau=TAU)
sc = StandardScaler().fit(Xc_tr)
rid = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc.transform(Xc_tr), psi_tr)
test_c["DR-Ridge"] = rid.predict(sc.transform(Xc_te))
coef = pd.Series(rid.coef_, index=Xc_tr.columns)
_rg = np.random.default_rng(SEED + 7)
bsc = np.array([RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc.transform(Xc_tr.iloc[i_]), psi_tr[i_]).coef_
                for i_ in (_rg.integers(0, ntr, ntr) for _ in range(60 if FAST else 300))])
coef_lo, coef_hi = np.percentile(bsc, 2.5, axis=0), np.percentile(bsc, 97.5, axis=0)
test_c["DR-RF"] = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=30, max_features=0.5, n_jobs=-1,
                                        random_state=SEED).fit(Xt_tr, psi_tr).predict(Xt_te)
psi_te, _ = psi_from(a_te, dte[["days", "cens"]], mu1_t, mu0_t, ps_t)
print(f"Train par asar ka average (AIPW psi): {psi_tr.mean():+.1f} din | poore data par adjusted: {ate:+.1f} din")
print(f"   {'tareeqa':10s} {'average':>9s} {'SD':>7s} {'10%':>8s} {'90%':>8s} {'asar>0 (%)':>11s}")
cate_rows = []
for m_ in METHODS:
    v = oof[m_]
    print(f"   {m_:10s} {v.mean():9.1f} {v.std():7.1f} {np.percentile(v, 10):8.1f} {np.percentile(v, 90):8.1f} {(v > 0).mean()*100:10.1f}%")
    cate_rows.append({"method": m_, "mean": round(v.mean(), 1), "SD": round(v.std(), 1), "p10": round(np.percentile(v, 10), 1),
                      "p90": round(np.percentile(v, 90), 1), "share_positive_%": round((v > 0).mean() * 100, 1)})
print("\nDR-Ridge: kaun se features therapy ka asar badalte hain (din mein, 1 SD barhne par; 95% bootstrap daira):")
ct = pd.DataFrame({"coef": coef.round(1), "lo": np.round(coef_lo, 1), "hi": np.round(coef_hi, 1)})
ct["daira_0_ko_chhuta"] = ["haan" if (l <= 0 <= h) else "NAHI (alag)" for l, h in zip(ct["lo"], ct["hi"])]
ct = ct.reindex(coef.abs().sort_values(ascending=False).index)
print(ct.to_string()); ct.to_csv(TAB_DIR / "gbsg2_dr_ridge_coefs.csv")
cc = pd.DataFrame(oof).corr(method="spearman").round(2)
print("Tareeqon ka aapas ka rishta (Spearman):"); print(cc.to_string())
pd.DataFrame(cate_rows).to_csv(TAB_DIR / "gbsg2_cate_summary.csv", index=False)
print(f"[{time.time()-t0:.0f} sec]\n")

# ==================================================================
# STEP 7: VALIDATION (ACTG175 jaisi jaanch)
# ==================================================================
head("STEP 7: Validation")
B = 200 if FAST else 1000
res_tr = {m_: V.evaluate(psi_tr, oof[m_], rng, B) for m_ in METHODS}
res_te = {m_: V.evaluate(psi_te, test_c[m_], rng, B) for m_ in METHODS}
for R in (res_tr, res_te):
    ph_ = V.holm([R[m_]["slope_p"] if not np.isnan(R[m_]["slope_p"]) else 1.0 for m_ in METHODS])
    for m_, p_ in zip(METHODS, ph_):
        R[m_]["slope_p_holm"] = p_
vrows = []
for nm, R in (("TRAIN (out-of-fold)", res_tr), ("TEST (ek baar)", res_te)):
    print(nm + ":")
    print(f"   {'tareeqa':9s} {'gain% (95% daira)':>24s} {'slope (95% daira)':>24s} {'p (Holm)':>9s} {'top-bot din (95%)':>24s}")
    for m_ in METHODS:
        r = R[m_]
        print(f"   {m_:9s} {r['gain']:7.2f} ({r['gain_lo']:6.2f},{r['gain_hi']:6.2f})   {r['slope']:6.2f} "
              f"({r['slope']-1.96*r['slope_se']:6.2f},{r['slope']+1.96*r['slope_se']:6.2f})   {V.fmt_p(r['slope_p_holm']):>9s}   "
              f"{r['top_minus_bottom']:6.1f} ({r['tb_lo']:6.1f},{r['tb_hi']:6.1f})")
        vrows.append({"set": nm.split()[0].lower(), "method": m_, "gain_%": round(r["gain"], 2), "gain_lo": round(r["gain_lo"], 2),
                      "gain_hi": round(r["gain_hi"], 2), "slope": round(r["slope"], 3), "p_holm": round(r["slope_p_holm"], 4),
                      "top_minus_bottom_din": round(r["top_minus_bottom"], 1)})
    ev = [m_ for m_ in METHODS if R[m_]["slope_p_holm"] < 0.05 and R[m_]["slope"] > 0 and R[m_]["gain_lo"] > 0]
    print("   Saboot (slope Holm p < 0.05, slope > 0, gain daira 0 se oopar):", ev if ev else "KOI NAHI", "\n")

# global null: labels ko propensity ke barabar groups ke ANDAR shuffle karna (confounding bacha kar asar mita dena)
bins = pd.qcut(ps_oof, 5, labels=False, duplicates="drop")


def dr_ridge_perm(a_vec):
    m1, m0 = np.zeros(ntr), np.zeros(ntr)
    for tr, va in folds:
        for g, arr in ((1, m1), (0, m0)):
            idx = tr[a_vec[tr] == g]
            arr[va] = C.SurvModel("cox", len(idx), ntr).fit(Xc_tr.iloc[idx], ytr[idx]).rmst(Xc_tr.iloc[va], tau=TAU)
    psi_, _ = psi_from(a_vec, dtr[["days", "cens"]], m1, m0, ps_oof)
    tau_ = np.zeros(ntr)
    for tr, va in folds:
        s_ = StandardScaler().fit(Xc_tr.iloc[tr])
        tau_[va] = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(s_.transform(Xc_tr.iloc[tr]), psi_[tr]).predict(s_.transform(Xc_tr.iloc[va]))
    lc = np.mean((psi_ - psi_.mean()) ** 2)
    return (lc - np.mean((psi_ - tau_) ** 2)) / lc * 100, V.ols_slope(psi_, tau_)[0]


obs_g, obs_s = dr_ridge_perm(a_tr)
gs_, ss_ = [], []
for _ in range(N_PERM):
    a_p = a_tr.copy()
    for b_ in np.unique(bins):
        ii = np.where(bins == b_)[0]; a_p[ii] = rng.permutation(a_tr[ii])
    g_, s_ = dr_ridge_perm(a_p); gs_.append(g_); ss_.append(s_)
gs_, ss_ = np.array(gs_), np.array(ss_)
p_g, p_s = (1 + np.sum(gs_ >= obs_g)) / (1 + N_PERM), (1 + np.sum(ss_ >= obs_s)) / (1 + N_PERM)
print(f"Global null (DR-Ridge, {N_PERM} shuffle, propensity-groups ke andar):")
print(f"   gain : asli {obs_g:6.2f}% | null average {gs_.mean():6.2f}%, 95% hissa {np.percentile(gs_, 95):6.2f}% tak | p = {p_g:.3f}")
print(f"   slope: asli {obs_s:6.2f}  | null average {ss_.mean():6.2f} | p = {p_s:.3f}")
pd.DataFrame(vrows).to_csv(TAB_DIR / "gbsg2_validation.csv", index=False)
print(f"[{time.time()-t0:.0f} sec]\n")

# ==================================================================
# STEP 8: GRAPHS
# ==================================================================
head("STEP 8: Graphs")
# ---- Figure 1: data aur asar ----
fig = plt.figure(figsize=(19, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
ax = fig.add_subplot(gs[0, 0])
for g, col in ((0, "#2F5D8A"), (1, "#E4893B")):
    km[g].plot_survival_function(ax=ax, color=col, ci_alpha=0.12, lw=2)
ax.set_ylim(0.2, 1.02); ax.set_xlabel("Din"); ax.set_ylabel("Recurrence-free bache rehne ka imkaan"); ax.legend(loc="lower left")
ax.set_title(f"1. Kaplan-Meier (log-rank p = {lr.p_value:.4f})")
ax = fig.add_subplot(gs[0, 1])
s_ = smd.sort_values(key=abs)
ax.barh(s_.index, s_.values, color=["#C8553D" if abs(v) > 0.1 else "#3A9D8F" for v in s_.values], height=0.6)
ax.axvline(0.1, color="#333333", ls="--", lw=1.1); ax.axvline(-0.1, color="#333333", ls="--", lw=1.1); ax.axvline(0, color="#333333", lw=0.8)
for i, v in enumerate(s_.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.set_xlim(-0.35, 0.8); ax.set_xlabel("SMD (therapy wale minus nahi wale)")
ax.legend(handles=[Patch(color="#C8553D", label="|SMD| > 0.1 (alag)"), Patch(color="#3A9D8F", label="barabar"),
                   Line2D([0], [0], color="#333333", ls="--", label="+-0.1 ki hadd")], loc="lower right")
ax.set_title("2. Baseline balance: groups shuru mein ALAG hain")
ax = fig.add_subplot(gs[0, 2])
ax.hist(ps_eda[a_all == 0], bins=20, alpha=0.75, color="#2F5D8A", label="Therapy nahi", edgecolor="white")
ax.hist(ps_eda[a_all == 1], bins=20, alpha=0.75, color="#E4893B", label="Therapy", edgecolor="white")
ax.set_xlabel("Therapy milne ka imkaan (propensity)"); ax.set_ylabel("Mareezon ki tadaad"); ax.legend(loc="upper right")
ax.set_title("3. Overlap: dono groups ke mareez har jagah maujood")
ax = fig.add_subplot(gs[1, 0])
vals = [(crude, c_lo, c_hi, "#9AA5B1", "Crude (sirf KM)"), (ate, ate - 1.96 * ate_se, ate + 1.96 * ate_se, "#C8553D", "Adjusted (AIPW)")]
for i, (v, lo, hi, col, lab) in enumerate(vals):
    ax.bar(i, v, color=col, width=0.55)
    ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color="#222222", capsize=6, lw=1.6)
    ax.text(i, hi + 3, f"{v:+.0f} din", ha="center", fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels([v[4] for v in vals]); ax.axhline(0, color="#333333", lw=0.8)
ax.set_ylabel(f"Extra event-free din ({TAU} din tak)"); ax.set_ylim(min(0, min(v[1] for v in vals)) - 30, max(v[2] for v in vals) * 1.15)
ax.set_title("4. RMST fark: crude vs adjusted (daira = 95%)")
ax = fig.add_subplot(gs[1, 1])
for i, (nm, s, col) in enumerate((("Crude", hc, "#9AA5B1"), ("Adjusted", ha, "#C8553D"))):
    hr_, lo_, hi_ = np.exp(s["coef"]), s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]
    ax.plot([lo_, hi_], [i, i], color=col, lw=2.5); ax.scatter(hr_, i, color=col, s=80, zorder=3)
    ax.text(hi_ * 1.03, i, f"{hr_:.2f} ({lo_:.2f}-{hi_:.2f})", va="center", fontsize=10)
ax.axvline(1, color="#333333", ls="--", lw=1.2); ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([0.5, 0.7, 1, 1.4]); ax.set_xticklabels(["0.5", "0.7", "1", "1.4"]); ax.set_xlim(0.45, 1.8); ax.set_ylim(-0.6, 1.6)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Crude", "Adjusted"]); ax.set_xlabel("Hazard ratio (1 se neeche = therapy behtar)")
ax.set_title(f"5. Hazard ratio (PH check p = {ph_p:.3f})")
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
lines = [f"{n_all} mareez, {int(df.cens.sum())} events", f"Therapy: {int(df.treat.sum())} mareez", "",
         f"Therapy wale {TAU} din par:", f"   {float(km[1].predict(TAU))*100:.1f}% bache  (nahi wale {float(km[0].predict(TAU))*100:.1f}%)", "",
         "Adjusted asar:", f"   {ate:+.0f} din  (HR {np.exp(ha['coef']):.2f})", "",
         "Dhyan: groups shuru mein alag the", "(age SMD +0.57, menopause +0.61),", "is liye adjusted par bharosa."]
ax.text(0.02, 0.98, "6. Khulasa", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
ax.text(0.02, 0.9, "\n".join(lines), fontsize=11, va="top", linespacing=1.5, transform=ax.transAxes)
fig.suptitle("GBSG2: data aur hormonal therapy ka asar", fontsize=16, fontweight="bold")
fig.savefig(FIG_DIR / "gbsg2_1_data_effect.png")
print(f"   [graph save hua] {FIG_DIR / 'gbsg2_1_data_effect.png'}")

# ---- Figure 2: models, CATE, validation ----
fig = plt.figure(figsize=(19, 15.5), constrained_layout=True)
gs = fig.add_gridspec(3, 3)
kinds = list(grids) + ["Cox_base"]
lab = {"Cox": "Cox", "RSF": "RSF", "GBM": "GBM", "Cox_base": "Cox\n(treatment\nke baghair)"}
ax = fig.add_subplot(gs[0, 0])
for i, k in enumerate(kinds):
    m_, se_ = M.mean_se(cv_runs[k]["C_harrell"])
    ax.bar(i, m_, color=COL[k], width=0.6); ax.errorbar(i, m_, yerr=se_, color="#222222", capsize=5, lw=1.6)
    ax.text(i, m_ + se_ + 0.004, f"{m_:.3f}", ha="center", fontweight="bold")
ax.axhline(0.5, color="#333333", ls="--", lw=1.2, label="0.5 = sikka uchhalna")
ax.set_xticks(range(len(kinds))); ax.set_xticklabels([lab[k] for k in kinds]); ax.set_ylim(0.45, max(means.values()) + 0.07)
ax.set_ylabel("C-index (train ka CV)"); ax.legend(loc="upper right"); ax.set_title("1. Prediction ki darustgi (CV)")
ax = fig.add_subplot(gs[0, 1])
for i, k in enumerate(grids):
    tm = tdf.loc[tdf.kind == k, "C_harrell"].iloc[0]; lo_, hi_ = np.percentile(boots[k], [2.5, 97.5])
    ax.errorbar(i, tm, yerr=[[tm - lo_], [hi_ - tm]], fmt="o", color=COL[k], ms=9, capsize=6, lw=2)
    ax.scatter(i, cv_runs[k]["C_harrell"].mean(), marker="D", s=70, color="#222222", zorder=4)
ax.axhline(0.5, color="#333333", ls="--", lw=1.2); ax.set_xticks(range(3)); ax.set_xticklabels(list(grids)); ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0.45, max(np.percentile(boots[k], 97.5) for k in grids) + 0.04); ax.set_ylabel("C-index")
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", ms=9, label="Test (95% bootstrap daira)"),
                   Line2D([0], [0], marker="D", color="w", markerfacecolor="#222222", ms=8, label="Train ka CV")], loc="upper right")
ax.set_title("2. Test par (sirf ek baar)")
ax = fig.add_subplot(gs[0, 2])
for k in grids:
    s = oof_pred[k]; ok_ = ~np.isnan(s); q = pd.qcut(pd.Series(s[ok_]), 4, labels=False, duplicates="drop")
    xs, ys_, los, his = [], [], [], []
    for g_ in sorted(q.unique()):
        ii = np.where(ok_)[0][(q == g_).values]
        kf = KaplanMeierFitter().fit(dtr.days.iloc[ii], dtr.cens.iloc[ii]); ci = kf.confidence_interval_survival_function_
        ix_ = ci.index[ci.index <= TAU][-1]
        xs.append(s[ii].mean()); ys_.append(float(kf.predict(TAU))); los.append(float(ci.loc[ix_].iloc[0])); his.append(float(ci.loc[ix_].iloc[1]))
    ax.errorbar(xs, ys_, yerr=[np.array(ys_) - np.array(los), np.array(his) - np.array(ys_)], fmt="-o", color=COL[k], label=k, capsize=3, lw=1.8, ms=6)
ax.plot([0.2, 1], [0.2, 1], color="#333333", ls="--", lw=1.2, label="Bilkul sahi"); ax.set_xlim(0.2, 1); ax.set_ylim(0.2, 1)
ax.set_xlabel(f"Model ka imkaan ({TAU} din tak bache rehne ka)"); ax.set_ylabel("Asal (Kaplan-Meier)")
ax.legend(loc="upper left"); ax.set_title("3. Calibration (CV)")
ax = fig.add_subplot(gs[1, 0])
bp = ax.boxplot([oof[m_] for m_ in METHODS], patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color="#222222", lw=1.6))
for p_, m_ in zip(bp["boxes"], METHODS):
    p_.set_facecolor(MC[m_]); p_.set_alpha(0.9)
ax.axhline(ate, color="#333333", ls="--", lw=1.3, label="Poore data ka adjusted average asar"); ax.axhline(0, color="#999999", lw=0.8)
ax.set_xticklabels(METHODS, rotation=20); ax.set_ylabel(f"Asar (extra event-free din, {TAU} din tak)"); ax.legend(loc="upper left")
ax.set_title("4. CATE: mareezon ka asar (out-of-fold)")


def forest2(ax, key, title, xlabel, refs, xlim=None):
    y = 0
    for m_ in METHODS[::-1]:
        for R, mk, dy in ((res_tr, "o", 0.18), (res_te, "D", -0.18)):
            r = R[m_]
            lo, hi, v = key(r)
            ax.plot([lo, hi], [y + dy] * 2, color=MC[m_], lw=2.2, alpha=0.9 if R is res_tr else 0.55)
            ax.scatter(v, y + dy, color=MC[m_], s=45, marker=mk, zorder=3)
        y += 1
    for x, ls_ in refs:
        ax.axvline(x, color="#333333", ls=ls_, lw=1.2)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(METHODS[::-1]); ax.set_xlabel(xlabel); ax.set_title(title)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", ms=8, label="Train (out-of-fold)"),
                       Line2D([0], [0], marker="D", color="w", markerfacecolor="#666666", ms=8, label="Test")], loc="upper right")
    if xlim:
        ax.set_xlim(xlim)


ax = fig.add_subplot(gs[1, 1])
forest2(ax, lambda r: (r["gain_lo"], r["gain_hi"], r["gain"]), "5. Constant-effect se behtar? (gain %)", "gain % (0 se oopar = behtar)", [(0, "--")])
ax = fig.add_subplot(gs[1, 2])
forest2(ax, lambda r: (r["slope"] - 1.96 * r["slope_se"], r["slope"] + 1.96 * r["slope_se"], r["slope"]), "6. Calibration slope",
        "slope (0 = farq shor, 1 = bilkul sahi)", [(0, "--"), (1, ":")], xlim=(-10, 12))
for k_, (arr, obs, p_, nm_) in enumerate(((gs_, obs_g, p_g, "gain %"), (ss_, obs_s, p_s, "slope"))):
    ax = fig.add_subplot(gs[2, k_])
    ax.hist(arr, bins=12, color="#9AA5B1", edgecolor="white", label="Null (shuffle)")
    ax.axvline(obs, color="#C8553D", lw=2.5, label=f"Asli ({obs:.2f})")
    ax.set_xlabel(f"DR-Ridge ka {nm_}"); ax.set_ylabel("Shuffle ki tadaad"); ax.legend(loc="upper right")
    ax.set_title(f"{7+k_}. Global null: {nm_} (p = {p_:.3f})")
ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
ev_te = [m_ for m_ in METHODS if res_te[m_]["slope_p_holm"] < 0.05 and res_te[m_]["slope"] > 0 and res_te[m_]["gain_lo"] > 0]
lines = [f"Cox test C-index: {tdf.loc[tdf.kind == 'Cox', 'C_harrell'].iloc[0]:.3f}",
         f"RSF: {tdf.loc[tdf.kind == 'RSF', 'C_harrell'].iloc[0]:.3f}, GBM: {tdf.loc[tdf.kind == 'GBM', 'C_harrell'].iloc[0]:.3f}", "",
         f"Adjusted asar: {ate:+.0f} din ({TAU} din tak)", "",
         "Mareez-ba-mareez farq ka saboot:", f"   test par: {ev_te if ev_te else 'koi nahi'}",
         f"   global null p: gain {p_g:.2f}, slope {p_s:.2f}"]
ax.text(0.02, 0.98, "9. Khulasa", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
ax.text(0.02, 0.9, "\n".join(lines), fontsize=11.5, va="top", linespacing=1.5, transform=ax.transAxes)
fig.suptitle("GBSG2: prediction, CATE aur validation", fontsize=16, fontweight="bold")
fig.savefig(FIG_DIR / "gbsg2_2_models_cate_validation.png")
print(f"   [graph save hua] {FIG_DIR / 'gbsg2_2_models_cate_validation.png'}")
json.dump(dict(chosen=chosen, use_log=bool(use_log), tau=TAU, seed=SEED), open(PROCESSED / "gbsg2_choice.json", "w"), indent=2)
print(f"\nGBSG2 mukammal ({time.time()-t0:.0f} sec). Tables: results\\tables\\gbsg2_*.csv")
if SHOW:
    plt.show()