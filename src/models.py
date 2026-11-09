"""
MODELS - ACTG175: outcome ki prediction (Cox, Random Survival Forest, Gradient Boosting)
Chalane ka tareeqa (project folder se):
    python src/models.py
Jaldi test karne ke liye (chhota version):  ... models.py --fast
Window na kholni ho:                        ... models.py --no-show

Tarteeb:
  Step 1: Data load (features engineering ke saare files) + treatment ko feature banana
  Step 2: Har model ke chhote grid ko TRAIN ke andar repeated cross-validation se parakhna
  Step 3: Teeno tuned models ka muqabla (CV par) + reference
  Step 4: Test data par EK BAAR final jaanch (bootstrap daire ke saath)
  Step 5: Graph aur tables save
QAIDA: model aur uske settings (hyperparameters) sirf TRAIN ke CV se chune jate hain.
Test data sirf Step 4 mein, aur sirf ek baar khulta hai.
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
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter

try:
    from sksurv.util import Surv
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                cumulative_dynamic_auc, integrated_brier_score)
except ImportError:
    print("scikit-survival install nahi hai. PowerShell mein chalayein:  python -m pip install scikit-survival")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
FAST = "--fast" in sys.argv
SHOW = "--no-show" not in sys.argv

SEED = 42
CV_SPLITS = 5
CV_REPEATS = 1 if FAST else 2
N_JOBS = -1
N_BOOT = 100 if FAST else 500
TAU = 900                                           # "900 din par" naapne ka waqt
TIMES = np.linspace(150, 900, 16)                   # Brier score ke waqt
USE_TREATMENT = True                                # arms (3 dummy) ko prediction ke features mein shamil karna
CONT_COLS = ["age", "wtkg", "preanti", "cd40", "cd80"]
COX_DROP = ["oprior"]                               # features.py jaisa: Cox se oprior hata hua
N_TREES = 100 if FAST else 200

# Har grid SAADA se PECHIDA ki tarteeb mein: 1-SE qaide se sab se saada achha wala chuna jayega
COX_GRID = [dict(alpha=a) for a in ((10, 1) if FAST else (100, 10, 1, 0.1))]
RSF_GRID = [dict(min_samples_leaf=l, max_features=f) for l, f in
            (((60, "sqrt"), (25, "sqrt")) if FAST else ((60, "sqrt"), (25, "sqrt"), (25, 0.6), (10, "sqrt")))]
GBM_GRID = [dict(n_estimators=n, max_depth=d) for n, d in
            (((100, 1), (100, 2)) if FAST else ((100, 1), (200, 1), (100, 2), (200, 2), (300, 2)))]
GRIDS = {"Cox": COX_GRID, "RSF": RSF_GRID, "GBM": GBM_GRID}
LONG = {"Cox": "Cox (ridge)", "RSF": "Random Survival Forest", "GBM": "Gradient Boosting"}
COLORS = {"Cox": "#2F5D8A", "RSF": "#3A9D8F", "GBM": "#E4893B", "Cox_base": "#8E5BA8"}

plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


# ==================================================================
# STEP 1: DATA
# Features engineering ke saave kiye hue files. Treatment (arms) ko 3 dummy columns bana kar
# features mein jod dete hain, kyunke mareez ka outcome us treatment par bhi nirbhar karta hai.
# Cox ke liye scaling har fold ke andar hoti hai (leakage se bachne ke liye).
# ==================================================================
def load_split(name):
    X = pd.read_csv(PROCESSED / f"X_{name}_trees.csv", index_col="row")
    y = pd.read_csv(PROCESSED / f"y_{name}.csv", index_col="row")
    t = pd.read_csv(PROCESSED / f"treatment_{name}.csv", index_col="row")
    if USE_TREATMENT:
        for a in (1, 2, 3):
            X[f"arm_{a}"] = (t["arms"] == a).astype(int)
    ys = Surv.from_arrays(event=y["cens"].astype(bool).values, time=y["days"].values.astype(float))
    return X, ys, y, t


def features_for(kind, X):
    return X.drop(columns=[c for c in COX_DROP if c in X.columns]) if kind.startswith("Cox") else X


def make_model(kind, p):
    if kind.startswith("Cox"):
        pre = ColumnTransformer([("sc", StandardScaler(), CONT_COLS)], remainder="passthrough")
        return make_pipeline(pre, CoxPHSurvivalAnalysis(alpha=p["alpha"]))
    if kind == "RSF":
        return RandomSurvivalForest(n_estimators=N_TREES, min_samples_leaf=p["min_samples_leaf"],
                                    max_features=p["max_features"], n_jobs=N_JOBS, random_state=SEED)
    return GradientBoostingSurvivalAnalysis(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                            learning_rate=0.1, subsample=0.7, min_samples_leaf=25,
                                            random_state=SEED)


# ==================================================================
# NAAP (metrics) - asaan lafz mein:
#  C-index (Harrell): do mareezon mein se kis ka event pehle hoga, model kitni baar sahi bata paya.
#      0.5 = sikka uchhalna, 1.0 = bilkul sahi. Trial data mein 0.65 se 0.75 aam hai.
#  C-index (Uno, 900 din): wahi, lekin censoring ka asar sambhal kar aur 900 din tak.
#  AUC (900 din): 900 din tak jin par event hua aur jin par nahi, unhe model kitna alag kar paya.
#  Brier score: model ne jo imkaan bataya wo asli nateeje se kitna door tha (chhota = behtar).
#  IBS skill: Brier score ka "sirf average curve" wale model se muqabla. 0 = koi fayda nahi,
#      jitna oopar utna model us saade curve se behtar.
# ==================================================================
def km_at(y_train, times):
    x, s = kaplan_meier_estimator(y_train["event"], y_train["time"])
    return np.array([s[np.searchsorted(x, t, side="right") - 1] if t >= x[0] else 1.0 for t in times])


def evaluate(model, Xa, ya, Xb, yb):
    risk = model.predict(Xb)
    tmin, tmax = yb["time"].min(), yb["time"].max()
    times = TIMES[(TIMES > tmin) & (TIMES < tmax)]
    out = {"C_harrell": concordance_index_censored(yb["event"], yb["time"], risk)[0],
           "C_uno": concordance_index_ipcw(ya, yb, risk, tau=TAU)[0],
           "AUC_900": cumulative_dynamic_auc(ya, yb, risk, [TAU])[0][0]}
    fns = model.predict_survival_function(Xb)
    S = np.vstack([f(times) for f in fns])
    ibs = integrated_brier_score(ya, yb, S, times)
    ibs_km = integrated_brier_score(ya, yb, np.tile(km_at(ya, times), (len(yb), 1)), times)
    out["IBS"], out["IBS_skill"] = ibs, 1 - ibs / ibs_km
    s900 = np.array([f(TAU) for f in fns])
    return out, risk, s900


def cv_model(kind, params, X, y, strata, want_oof=False):
    """Sirf TRAIN ke andar repeated cross-validation."""
    rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=SEED)
    rows, oof = [], np.full(len(X), np.nan)
    for i, (a, b) in enumerate(rskf.split(X, strata)):
        m = make_model(kind, params).fit(X.iloc[a], y[a])
        r, _, s900 = evaluate(m, X.iloc[a], y[a], X.iloc[b], y[b])
        rows.append(r)
        if want_oof and i < CV_SPLITS:               # pehle repeat ke out-of-fold andaze (calibration ke liye)
            oof[b] = s900
    return pd.DataFrame(rows), oof


def mean_se(s):
    return s.mean(), s.std() / np.sqrt(CV_SPLITS)


def pick_simplest(results):
    """Sab se saada config jo sab se achhe C-index ke 1 standard error ke andar ho."""
    means = [r["C_harrell"].mean() for r in results]
    best = int(np.argmax(means)); se = results[best]["C_harrell"].std() / np.sqrt(CV_SPLITS)
    for i, m in enumerate(means):
        if m >= means[best] - se:
            return i, best


def fmt_params(p):
    return ", ".join(f"{k}={v}" for k, v in p.items())


# ------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    Xtr_all, ytr, ytr_df, ttr = load_split("train")
    Xte_all, yte, yte_df, tte = load_split("test")
    strata = (ttr["arms"].astype(str) + "_" + ytr_df["cens"].astype(str)).values

    head("STEP 1: Data")
    print(f"Train: {len(Xtr_all)} mareez, {int(ytr['event'].sum())} events | Test: {len(Xte_all)} mareez, {int(yte['event'].sum())} events (abhi khola nahi jayega)")
    print(f"Features (trees ke liye): {list(Xtr_all.columns)}")
    print(f"Cox ke liye: oprior hata hua, continuous columns har fold mein scale hote hain.")
    print(f"Treatment (arms) features mein shamil: {USE_TREATMENT}")
    print(f"Cross-validation: {CV_SPLITS}-fold x {CV_REPEATS} baar | {'FAST mode (chhota version)' if FAST else 'poora version'}\n")

    # ---------------- Step 2 ----------------
    head("STEP 2: Har model ke settings ka grid (TRAIN ke CV se)")
    print("Qaida: jo setting sab se achhe C-index ke 1 standard error ke andar ho, un mein se sab se SAADA chunte hain")
    print("(saada model kam overfit karta hai).\n")
    grid_rows, chosen, grid_res, grid_oof, chosen_idx = [], {}, {}, {}, {}
    for kind, grid in GRIDS.items():
        X = features_for(kind, Xtr_all)
        res, oofs = [], []
        for p in grid:
            r, oof = cv_model(kind, p, X, ytr, strata, want_oof=True)
            res.append(r); oofs.append(oof)
            m_c, se_c = mean_se(r["C_harrell"])
            grid_rows.append({"model": kind, "settings": fmt_params(p), "C_harrell": round(m_c, 4), "SE": round(se_c, 4),
                              "C_uno": round(r["C_uno"].mean(), 4), "IBS_skill": round(r["IBS_skill"].mean(), 4)})
        idx, best = pick_simplest(res)
        chosen[kind], chosen_idx[kind], grid_res[kind], grid_oof[kind] = grid[idx], idx, res, oofs
        print(f"{LONG[kind]}:")
        for i, p in enumerate(grid):
            m_c, se_c = mean_se(res[i]["C_harrell"])
            tag = "   <-- chuna gaya" if i == idx else ("   (sab se achha)" if i == best else "")
            print(f"   {fmt_params(p):42s} C-index {m_c:.4f} +- {se_c:.4f}{tag}")
        if best == len(grid) - 1 and idx == best:
            print("   Dhyan: sab se achha setting grid ke aakhri kinare par hai, shayad zyada pechida setting bhi aazmani chahiye.")
        print(f"   [{time.time()-t0:.0f} sec guzre]\n")
    pd.DataFrame(grid_rows).to_csv(TAB_DIR / "models_grid.csv", index=False)

    # ---------------- Step 3 ----------------
    head("STEP 3: Teeno tuned models ka muqabla (TRAIN ke CV par)")
    cv_tab, oof_pred, cv_runs = [], {}, {}
    for kind in GRIDS:                                  # Step 2 ke nateeje dobara istemal (dobara nahi chalaya)
        cv_runs[kind], oof_pred[kind] = grid_res[kind][chosen_idx[kind]], grid_oof[kind][chosen_idx[kind]]
    ref = None
    if USE_TREATMENT:      # reference: Cox, treatment ke baghair
        Xb = features_for("Cox", Xtr_all.drop(columns=[c for c in Xtr_all.columns if c.startswith("arm_")]))
        ref, _ = cv_model("Cox_base", chosen["Cox"], Xb, ytr, strata)
        cv_runs["Cox_base"] = ref
    names = {**LONG, "Cox_base": "Cox, treatment ke baghair (reference)"}
    for kind, r in cv_runs.items():
        row = {"model": names[kind]}
        for mt in ["C_harrell", "C_uno", "AUC_900", "IBS_skill"]:
            m_, se_ = mean_se(r[mt]); row[mt] = f"{m_:.4f} +- {se_:.4f}"
        cv_tab.append(row)
    cv_df = pd.DataFrame(cv_tab)
    print(cv_df.to_string(index=False))
    cv_df.to_csv(TAB_DIR / "models_cv.csv", index=False)
    means = {k: cv_runs[k]["C_harrell"].mean() for k in GRIDS}
    top = max(means, key=means.get); se_top = cv_runs[top]["C_harrell"].std() / np.sqrt(CV_SPLITS)
    simplest = next(k for k in ["Cox", "RSF", "GBM"] if means[k] >= means[top] - se_top)
    print(f"\nSab se achha: {LONG[top]} ({means[top]:.4f}). Sab se saada jo 1 SE ({se_top:.4f}) ke andar ho: {LONG[simplest]}.")
    if ref is not None:
        gain = cv_runs["Cox"]["C_harrell"].mean() - ref["C_harrell"].mean()
        print(f"Treatment (arms) shamil karne se Cox ka C-index {gain:+.4f} badla.")
    print(f"[{time.time()-t0:.0f} sec guzre]\n")

    # ---------------- Step 4 ----------------
    head("STEP 4: Test data par EK BAAR jaanch (ab test khula)")
    test_rows, test_risk, test_s900, fitted = [], {}, {}, {}
    for kind in GRIDS:
        Xa, Xb = features_for(kind, Xtr_all), features_for(kind, Xte_all)
        m = make_model(kind, chosen[kind]).fit(Xa, ytr)
        r, risk, s900 = evaluate(m, Xa, ytr, Xb, yte)
        fitted[kind], test_risk[kind], test_s900[kind] = m, risk, s900
        test_rows.append({"kind": kind, "model": LONG[kind], **r})
    rng = np.random.default_rng(SEED)
    n_te = len(yte); boots = {k: [] for k in GRIDS}
    for _ in range(N_BOOT):
        ix = rng.integers(0, n_te, n_te)
        if yte["event"][ix].sum() < 5:
            continue
        for k in GRIDS:
            boots[k].append(concordance_index_censored(yte["event"][ix], yte["time"][ix], test_risk[k][ix])[0])
    boots = {k: np.array(v) for k, v in boots.items()}
    tdf = pd.DataFrame(test_rows)
    tdf["C_harrell_95%"] = [f"{np.percentile(boots[k], 2.5):.3f} - {np.percentile(boots[k], 97.5):.3f}" for k in tdf["kind"]]
    show = tdf[["model", "C_harrell", "C_harrell_95%", "C_uno", "AUC_900", "IBS_skill"]].copy()
    for c in ["C_harrell", "C_uno", "AUC_900", "IBS_skill"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))
    tdf.drop(columns="kind").round(4).to_csv(TAB_DIR / "models_test.csv", index=False)
    print("\nModelon ka aapas ka fark (test C-index, bootstrap 95% daira; 0 daire ke andar = koi saaf fark nahi):")
    for k in ("RSF", "GBM"):
        d = boots[k] - boots["Cox"]
        print(f"   {LONG[k]} minus Cox: {np.mean(d):+.4f}  ({np.percentile(d, 2.5):+.4f} se {np.percentile(d, 97.5):+.4f})")
    print(f"\nCV C-index vs Test C-index:")
    for kind in GRIDS:
        cvm = cv_runs[kind]["C_harrell"].mean(); tm = tdf.loc[tdf.kind == kind, "C_harrell"].iloc[0]
        print(f"   {LONG[kind]:24s} CV {cvm:.4f} | Test {tm:.4f} | farq {tm - cvm:+.4f}")
    print(f"[{time.time()-t0:.0f} sec guzre]\n")

    # ---------------- Step 5 ----------------
    head("STEP 5: Graph aur save")
    fig = plt.figure(figsize=(20, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    kinds = list(GRIDS) + (["Cox_base"] if ref is not None else [])
    lab = {"Cox": "Cox", "RSF": "RSF", "GBM": "GBM", "Cox_base": "Cox\n(treatment\nke baghair)"}

    # 1. CV C-index
    ax = fig.add_subplot(gs[0, 0])
    for i, k in enumerate(kinds):
        m_, se_ = mean_se(cv_runs[k]["C_harrell"])
        ax.bar(i, m_, color=COLORS[k], width=0.6)
        ax.errorbar(i, m_, yerr=se_, color="#222222", capsize=5, lw=1.6)
        ax.text(i, m_ + se_ + 0.004, f"{m_:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="#333333", ls="--", lw=1.2, label="0.5 = sikka uchhalna")
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels([lab[k] for k in kinds])
    lo_ = min(0.5, min(cv_runs[k]["C_harrell"].mean() for k in kinds)) - 0.03
    ax.set_ylim(0.45, max(means.values()) + 0.06); ax.set_ylabel("C-index (train ka cross-validation)")
    ax.legend(loc="upper right"); ax.set_title("1. Prediction ki darustgi (CV, bara = behtar)")

    # 2. IBS skill
    ax = fig.add_subplot(gs[0, 1])
    for i, k in enumerate(kinds):
        m_, se_ = mean_se(cv_runs[k]["IBS_skill"])
        ax.bar(i, m_, color=COLORS[k], width=0.6)
        ax.errorbar(i, m_, yerr=se_, color="#222222", capsize=5, lw=1.6)
        ax.text(i, m_ + se_ + 0.002, f"{m_:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#333333", ls="--", lw=1.2, label="0 = sirf average curve jitna")
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels([lab[k] for k in kinds])
    ax.set_ylabel("IBS skill (CV)"); ax.legend(loc="upper right")
    ax.set_title("2. Imkaan ki sahi andazi (Brier skill, bara = behtar)")

    # 3. Test vs CV
    ax = fig.add_subplot(gs[0, 2])
    for i, k in enumerate(GRIDS):
        tm = tdf.loc[tdf.kind == k, "C_harrell"].iloc[0]
        lo_, hi_ = np.percentile(boots[k], [2.5, 97.5])
        ax.errorbar(i, tm, yerr=[[tm - lo_], [hi_ - tm]], fmt="o", color=COLORS[k], ms=9, capsize=6, lw=2)
        ax.scatter(i, cv_runs[k]["C_harrell"].mean(), marker="D", s=70, color="#222222", zorder=4)
    ax.axhline(0.5, color="#333333", ls="--", lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(GRIDS)); ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0.45, max(hi for hi in [np.percentile(boots[k], 97.5) for k in GRIDS]) + 0.04)
    ax.set_ylabel("C-index")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#666666", ms=9, label="Test (95% bootstrap daira)"),
                       Line2D([0], [0], marker="D", color="w", markerfacecolor="#222222", ms=8, label="Train ka CV"),
                       Line2D([0], [0], color="#333333", ls="--", label="0.5")], loc="upper right")
    ax.set_title("3. Test par kya hua (sirf ek baar)")

    # 4. Calibration (OOF, train)
    ax = fig.add_subplot(gs[1, 0])
    for k in GRIDS:
        s = oof_pred[k]; ok = ~np.isnan(s)
        q = pd.qcut(pd.Series(s[ok]), 5, labels=False, duplicates="drop")
        xs, ys_, los, his = [], [], [], []
        for g_ in sorted(q.unique()):
            ii = np.where(ok)[0][(q == g_).values]
            kf = KaplanMeierFitter().fit(ytr_df["days"].iloc[ii], ytr_df["cens"].iloc[ii])
            ci = kf.confidence_interval_survival_function_
            idx_ = ci.index[ci.index <= TAU][-1]
            xs.append(s[ii].mean()); ys_.append(float(kf.predict(TAU)))
            los.append(float(ci.loc[idx_].iloc[0])); his.append(float(ci.loc[idx_].iloc[1]))
        ax.errorbar(xs, ys_, yerr=[np.array(ys_) - np.array(los), np.array(his) - np.array(ys_)], fmt="-o",
                    color=COLORS[k], label=k, capsize=3, lw=1.8, ms=6)
    lims = [0.45, 1.0]; ax.plot(lims, lims, color="#333333", ls="--", lw=1.2, label="Bilkul sahi")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"Model ka batata imkaan ({TAU} din tak bache rehne ka)")
    ax.set_ylabel("Asal mein kitne bache (Kaplan-Meier)")
    ax.legend(loc="upper left"); ax.set_title("4. Calibration (CV): imkaan asli nateeje se milta hai?")

    # 5. Cox HR (lifelines, scaled)
    ax = fig.add_subplot(gs[1, 1])
    Xc = pd.read_csv(PROCESSED / "X_train_cox.csv", index_col="row")
    if USE_TREATMENT:
        for a in (1, 2, 3):
            Xc[f"arm_{a}"] = (ttr["arms"] == a).astype(int)
    cph = CoxPHFitter(penalizer=0.01).fit(Xc.assign(days=ytr_df["days"], cens=ytr_df["cens"]), "days", "cens")
    sm = cph.summary.sort_values("coef")
    for i, (nm, r) in enumerate(sm.iterrows()):
        hr, lo_, hi_ = np.exp(r["coef"]), r["exp(coef) lower 95%"], r["exp(coef) upper 95%"]
        col = "#C8553D" if lo_ > 1 else ("#3A9D8F" if hi_ < 1 else "#9AA5B1")
        ax.plot([lo_, hi_], [i, i], color=col, lw=2.2); ax.scatter(hr, i, color=col, s=45, zorder=3)
    ax.axvline(1, color="#333333", ls="--", lw=1.2)
    ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0.4, 0.6, 1, 1.5, 2]); ax.set_xticklabels(["0.4", "0.6", "1", "1.5", "2"])
    ax.set_yticks(range(len(sm))); ax.set_yticklabels(sm.index)
    ax.set_xlabel("Hazard ratio (continuous: 1 SD barhne par, baqi: 0 se 1)")
    ax.legend(handles=[Patch(color="#C8553D", label="Khatra zyada"), Patch(color="#3A9D8F", label="Khatra kam"),
                       Patch(color="#9AA5B1", label="Fark pakka nahi")], loc="lower right")
    ax.set_title("5. Cox: kaun sa feature khatra kitna badlata hai")

    # 6. GBM importance
    ax = fig.add_subplot(gs[1, 2])
    gb = fitted["GBM"]
    imp = pd.Series(gb.feature_importances_, index=features_for("GBM", Xtr_all).columns).sort_values()
    ax.barh(imp.index, imp.values, color=COLORS["GBM"], height=0.65)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlim(0, imp.max() * 1.15); ax.set_xlabel("Importance (GBM)")
    ax.set_title("6. Boosting ke nazdeek kaun sa feature zyada kaam ka")

    fig.suptitle("Prediction models ka muqabla  -  ACTG175", fontsize=16, fontweight="bold")
    fig.savefig(FIG_DIR / "models_1_comparison.png")
    print(f"   [graph save hua] {FIG_DIR / 'models_1_comparison.png'}")
    for f in ("models_grid.csv", "models_cv.csv", "models_test.csv"):
        print(f"   [table save hua] {TAB_DIR / f}")
    with open(PROCESSED / "models_choice.json", "w", encoding="utf-8") as fh:
        json.dump(dict(chosen=chosen, use_treatment=USE_TREATMENT, cv_splits=CV_SPLITS, cv_repeats=CV_REPEATS,
                       simplest_within_1se=simplest, best_cv=top), fh, indent=2)
    print(f"   [save hua] {PROCESSED / 'models_choice.json'}")
    print(f"\nModels mukammal ({time.time()-t0:.0f} sec). Agla qadam: heterogeneity (kis mareez ko kaunsa treatment zyada faida de).")
    if SHOW:
        plt.show()