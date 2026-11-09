"""
CATE - ACTG175: treatment ka asar HAR MAREEZ par alag? (heterogeneity)
Chalane ka tareeqa (project folder se):
    python src/cate.py
Jaldi test (chhota version):  ... cate.py --fast        Window na khule:  ... cate.py --no-show

CATE = Conditional Average Treatment Effect = "is jaise mareez (age, cd40, symptom...) ko ZDV akele ki
jagah doosra arm dene par kitne din zyada event ke baghair guzarenge". Yahan asar RMST ke fark (din mein,
TAU = 900 din tak) mein naapa jata hai. Har arm (ZDV+ddI, ZDV+zal, ddI) ZDV ke muqabley mein alag hai.

Tareeqe (sab Python mein, koi R nahi):
  S-Cox   : ek hi Cox model, arm sirf ek column. Hazard ratio sab ke liye ek jaisa; asar sirf mareez ke
            baseline khatre ki wajah se badalta hai (reference: "sirf khatre se farq").
  T-Cox   : har arm ka alag Cox model. Dono ka fark = asar.
  T-RSF   : har arm ka alag Random Survival Forest.
  DR-Ridge: doubly-robust pseudo-outcome + linear (ridge) model: kaun se features asar badalte hain.
  DR-RF   : wahi pseudo-outcome + random forest (nonlinear).
Sab ke liye out-of-fold (cross-fitting) andaze: mareez ka andaza usi model se jo us mareez ko dekhe baghair bana.

Test data par sirf PREDICTIONS save hoti hain (koi jaanch nahi). Jaanch validation.py mein, ek baar.
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter

try:
    from sksurv.util import Surv
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.ensemble import RandomSurvivalForest
except ImportError:
    print("scikit-survival install nahi hai:  python -m pip install scikit-survival --no-deps  (aur README dekhein)")
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
TAU = 900
ARMS = (1, 2, 3)
ARM_NAMES = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+zal", 3: "ddI"}
ARM_COLORS = {0: "#2F5D8A", 1: "#E4893B", 2: "#3A9D8F", 3: "#8E5BA8"}
N_FOLDS = 5
N_TREES = 80 if FAST else 200
CONT_COLS = ["age", "wtkg", "preanti", "cd40", "cd80"]
COX_DROP = ["oprior"]
METHODS = ["S-Cox", "T-Cox", "T-RSF", "DR-Ridge", "DR-RF"]
MCOL = {"S-Cox": "#9AA5B1", "T-Cox": "#2F5D8A", "T-RSF": "#3A9D8F", "DR-Ridge": "#C8553D", "DR-RF": "#8E5BA8"}
PS_FLOOR = 0.05            # censoring ka KM kabhi is se neeche na jaye (weights na phatein)
ALPHA_S = 0.1              # S-Cox mein arm ka asar ridge se dabna nahi chahiye, isliye halki penalty

# models.py ne jo settings chuni thi wohi (dobara tune nahi karte)
try:
    CHOSEN = json.load(open(PROCESSED / "models_choice.json", encoding="utf-8"))["chosen"]
except Exception:
    CHOSEN = {"Cox": {"alpha": 10}, "RSF": {"min_samples_leaf": 60, "max_features": "sqrt"}}
COX_ALPHA = CHOSEN["Cox"]["alpha"]
RSF_LEAF, RSF_MAXF = CHOSEN["RSF"]["min_samples_leaf"], CHOSEN["RSF"]["max_features"]

plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


# ==================================================================
# DATA
# ==================================================================
def load(name):
    X = pd.read_csv(PROCESSED / f"X_{name}_trees.csv", index_col="row")
    yd = pd.read_csv(PROCESSED / f"y_{name}.csv", index_col="row")
    arms = pd.read_csv(PROCESSED / f"treatment_{name}.csv", index_col="row")["arms"].values
    ys = Surv.from_arrays(event=yd["cens"].astype(bool).values, time=yd["days"].values.astype(float))
    return X, ys, yd, arms


# ==================================================================
# SURVIVAL MODEL + RMST
# RMST = pehle TAU din mein mareez average kitne din "event ke baghair" guzarta hai = predicted survival
# curve ke neeche ka raqba. Model har mareez ke liye curve deta hai, hum raqba nikalte hain.
# ==================================================================
def rmst_from_surv(S, t, tau):
    m = t < tau
    tt = np.concatenate([[0.0], t[m], [tau]])
    SS = np.hstack([np.ones((S.shape[0], 1)), S[:, m]])
    return (SS * np.diff(tt)).sum(axis=1)


class SurvModel:
    def __init__(self, kind, n_fit, n_total, alpha=None):
        self.kind, self.n_fit, self.n_total = kind, n_fit, n_total
        self.alpha = COX_ALPHA if alpha is None else alpha

    def _prep(self, X):
        X = X.drop(columns=[c for c in COX_DROP if c in X.columns]) if self.kind == "cox" else X
        return X

    def fit(self, X, y):
        X = self._prep(X)
        if self.kind == "cox":
            self.pre = ColumnTransformer([("sc", StandardScaler(), [c for c in CONT_COLS if c in X.columns])],
                                         remainder="passthrough").fit(X)
            self.est = CoxPHSurvivalAnalysis(alpha=self.alpha).fit(self.pre.transform(X), y)
        else:
            leaf = max(10, int(round(RSF_LEAF * self.n_fit / self.n_total)))     # chhote group ke liye chhota leaf
            self.pre = None
            self.est = RandomSurvivalForest(n_estimators=N_TREES, min_samples_leaf=leaf, max_features=RSF_MAXF,
                                            n_jobs=-1, random_state=SEED).fit(X.values, y)
        return self

    def rmst(self, X, tau=TAU):
        X = self._prep(X)
        Z = self.pre.transform(X) if self.pre is not None else X.values
        S = self.est.predict_survival_function(Z, return_array=True)
        return rmst_from_surv(S, self.est.unique_times_, tau)


def with_arm(X, arm):
    """S-learner ke liye: sab mareezon ko arm 'arm' par farz karna (arm_1..arm_3 dummy)."""
    Z = X.copy()
    for a in ARMS:
        Z[f"arm_{a}"] = int(arm == a)
    return Z


# ==================================================================
# DR (doubly robust) PSEUDO-OUTCOME
# Problem: asli asar (har mareez ka) kabhi dikhta nahi (mareez ko sirf ek arm milta hai) aur censoring bhi hai.
# Hal: IPCW outcome Y* = (TAU tak ka observed waqt) / (us waqt tak censor na hone ka imkaan), phir
#   psi = mu_a(x) - mu_0(x) + (A/e_a)(Y* - mu_a(x)) - ((1-A)/e_0)(Y* - mu_0(x))
# jahan mu = model ka andaza, e = randomization ka hissa (trial mein pata hai). psi ka average = asar ka
# sahi andaza (model ghalat bhi ho to randomization bachati hai) aur psi ko features par regress karne se CATE.
# ==================================================================
def ipcw_outcome(days, event, tau, fit_days, fit_event):
    kmf = KaplanMeierFitter().fit(fit_days, 1 - fit_event)                # censoring ka KM
    Tt = np.minimum(days, tau)
    observed = ((event == 1) & (days <= tau)) | (days >= tau)             # TAU tak censor nahi hua
    G = kmf.survival_function_at_times(np.maximum(Tt - 1e-6, 0)).values
    return np.where(observed, Tt / np.clip(G, PS_FLOOR, 1.0), 0.0)


def dr_pseudo(arm, a_vec, ydf, mu_cox):
    pos = np.where((a_vec == 0) | (a_vec == arm))[0]
    days, ev = ydf["days"].values[pos], ydf["cens"].values[pos]
    A = (a_vec[pos] == arm).astype(float)
    e_a = A.mean(); e_0 = 1 - e_a
    Ystar = np.zeros(len(pos))
    for g in (0, 1):                                  # censoring ka hissa har arm ka apna (arms mein alag ho sakta hai)
        m_ = A == g
        Ystar[m_] = ipcw_outcome(days[m_], ev[m_], TAU, days[m_], ev[m_])
    m1, m0 = mu_cox[arm][pos], mu_cox[0][pos]
    psi = m1 - m0 + (A / e_a) * (Ystar - m1) - ((1 - A) / e_0) * (Ystar - m0)
    return pos, psi, Ystar, A


def km_rmst(days, ev, tau=TAU):
    k = KaplanMeierFitter().fit(days, ev)
    sf = k.survival_function_
    t, s = sf.index.values, sf.iloc[:, 0].values
    m = t < tau
    return float(np.sum(s[m] * np.diff(np.append(t[m], tau))))


# ------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    Xtr, ytr, ydf_tr, a_tr = load("train")
    Xte, yte, ydf_te, a_te = load("test")
    n, nte = len(Xtr), len(Xte)
    strata = pd.Series(a_tr).astype(str) + "_" + ydf_tr["cens"].astype(str).values
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xtr, strata))

    head("STEP 1: Data aur tareeqa")
    print(f"Train: {n} mareez | Test: {nte} mareez (sirf predictions save hongi, jaanch validation.py mein)")
    print(f"Asar ka naap: RMST ka fark ({TAU} din tak, din mein), har arm ZDV ke muqabley mein.")
    print(f"Cross-fitting: {N_FOLDS}-fold. Cox alpha={COX_ALPHA}, RSF leaf={RSF_LEAF} (models.py wali). "
          f"{'FAST mode' if FAST else 'poora version'}\n")

    # ---------------- Step 2: outcome models (T aur S), out-of-fold ----------------
    head("STEP 2: Outcome models (T-learner, S-learner) out-of-fold")
    mu_oof = {k: {a: np.zeros(n) for a in range(4)} for k in ("cox", "rsf", "scox")}
    mu_te = {k: {a: None for a in range(4)} for k in ("cox", "rsf", "scox")}
    for f, (tr, va) in enumerate(folds, 1):
        for kind in ("cox", "rsf"):
            for a in range(4):
                idx = tr[a_tr[tr] == a]
                m = SurvModel(kind, len(idx), n).fit(Xtr.iloc[idx], ytr[idx])
                mu_oof[kind][a][va] = m.rmst(Xtr.iloc[va])
        Xs_fit = pd.concat([Xtr.iloc[tr]] * 1).copy()
        for a_ in ARMS:
            Xs_fit[f"arm_{a_}"] = (a_tr[tr] == a_).astype(int)
        ms = SurvModel("cox", len(tr), n, alpha=ALPHA_S).fit(Xs_fit, ytr[tr])
        for a in range(4):
            mu_oof["scox"][a][va] = ms.rmst(with_arm(Xtr.iloc[va], a))
        print(f"   fold {f}/{N_FOLDS} mukammal [{time.time()-t0:.0f} sec]")
    # final (poore train par) -> test ke liye
    for kind in ("cox", "rsf"):
        for a in range(4):
            idx = np.where(a_tr == a)[0]
            mu_te[kind][a] = SurvModel(kind, len(idx), n).fit(Xtr.iloc[idx], ytr[idx]).rmst(Xte)
    Xs_all = Xtr.copy()
    for a_ in ARMS:
        Xs_all[f"arm_{a_}"] = (a_tr == a_).astype(int)
    ms_all = SurvModel("cox", n, n, alpha=ALPHA_S).fit(Xs_all, ytr)
    for a in range(4):
        mu_te["scox"][a] = ms_all.rmst(with_arm(Xte, a))
    print(f"Final models (poore train par) bhi taiyar [{time.time()-t0:.0f} sec]\n")

    # ---------------- Step 3: DR pseudo-outcome + final regressors ----------------
    head("STEP 3: Doubly-robust pseudo-outcome aur DR-Ridge / DR-RF")
    oof = {(a, m): np.zeros(n) for a in ARMS for m in METHODS}
    test = {(a, m): None for a in ARMS for m in METHODS}
    coefs, ate_rows = {}, []
    feat_cols = list(Xtr.columns)
    sc_all = StandardScaler().fit(Xtr)
    for arm in ARMS:
        pos, psi, Ystar, A = dr_pseudo(arm, a_tr, ydf_tr, mu_oof["cox"])
        # sanity: IPCW outcome ka average, KM RMST se milna chahiye
        km1 = km_rmst(ydf_tr["days"].values[pos][A == 1], ydf_tr["cens"].values[pos][A == 1])
        km0 = km_rmst(ydf_tr["days"].values[pos][A == 0], ydf_tr["cens"].values[pos][A == 0])
        print(f"{ARM_NAMES[arm]} vs ZDV: KM RMST fark {km1-km0:+.1f} din | DR psi ka average {psi.mean():+.1f} din | "
              f"IPCW check: Y* average ZDV {Ystar[A==0].mean():.0f} (KM {km0:.0f}), {ARM_NAMES[arm]} {Ystar[A==1].mean():.0f} (KM {km1:.0f})")
        ate_rows.append({"arm": ARM_NAMES[arm], "KM_RMST_fark": round(km1 - km0, 1), "DR_psi_mean": round(psi.mean(), 1)})
        # out-of-fold DR final models
        for tr, va in folds:
            use = np.isin(pos, tr)
            Xp, yp = Xtr.iloc[pos[use]], psi[use]
            sc = StandardScaler().fit(Xp)
            r = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc.transform(Xp), yp)
            oof[(arm, "DR-Ridge")][va] = r.predict(sc.transform(Xtr.iloc[va]))
            rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=60, max_features=0.5,
                                       n_jobs=-1, random_state=SEED).fit(Xp, yp)
            oof[(arm, "DR-RF")][va] = rf.predict(Xtr.iloc[va])
        # final DR models (poore train par)
        r = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc_all.transform(Xtr.iloc[pos]), psi)
        test[(arm, "DR-Ridge")] = r.predict(sc_all.transform(Xte))
        coefs[arm] = pd.Series(r.coef_, index=feat_cols)
        rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=60, max_features=0.5,
                                   n_jobs=-1, random_state=SEED).fit(Xtr.iloc[pos], psi)
        test[(arm, "DR-RF")] = rf.predict(Xte)
        # T aur S learners: CATE = mu_arm - mu_ZDV
        for key, nm in (("scox", "S-Cox"), ("cox", "T-Cox"), ("rsf", "T-RSF")):
            oof[(arm, nm)] = mu_oof[key][arm] - mu_oof[key][0]
            test[(arm, nm)] = mu_te[key][arm] - mu_te[key][0]
    print(f"[{time.time()-t0:.0f} sec guzre]\n")

    # ---------------- Step 4: Khulasa ----------------
    head("STEP 4: CATE ka khulasa (train, out-of-fold)")
    rows = []
    for arm in ARMS:
        ate = ate_rows[arm - 1]["KM_RMST_fark"]
        print(f"{ARM_NAMES[arm]} (ZDV ke muqabley mein, poore trial ka average asar {ate:+.1f} din):")
        print(f"   {'tareeqa':10s} {'average':>9s} {'SD':>7s} {'10%':>8s} {'90%':>8s} {'asar > 0 (%)':>13s}")
        for m in METHODS:
            v = oof[(arm, m)]
            rows.append({"arm": ARM_NAMES[arm], "method": m, "mean": round(v.mean(), 1), "SD": round(v.std(), 1),
                         "p10": round(np.percentile(v, 10), 1), "p90": round(np.percentile(v, 90), 1),
                         "share_positive_%": round((v > 0).mean() * 100, 1)})
            print(f"   {m:10s} {v.mean():9.1f} {v.std():7.1f} {np.percentile(v,10):8.1f} {np.percentile(v,90):8.1f} {(v>0).mean()*100:12.1f}%")
        print()
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB_DIR / "cate_summary.csv", index=False)

    print("Tareeqon ke andaze ek dusre se kitne milte hain (Spearman correlation, teeno arms ka average):")
    corr = np.zeros((len(METHODS), len(METHODS)))
    for arm in ARMS:
        corr += pd.DataFrame({m: oof[(arm, m)] for m in METHODS}).corr(method="spearman").values / len(ARMS)
    cdf = pd.DataFrame(corr, index=METHODS, columns=METHODS).round(2)
    print(cdf.to_string(), "\n")

    print("DR-Ridge ke hisaab se asar badalne wale features (din mein, 1 SD barhne par, ZDV ke muqabley mein):")
    cf = pd.DataFrame(coefs)
    cf.columns = [ARM_NAMES[a] for a in ARMS]
    top = cf.abs().mean(axis=1).sort_values(ascending=False).head(6).index
    print(cf.loc[top].round(1).to_string())
    print("(Yeh sirf ishara hai: saaf pakka hone ke liye validation.py ki jaanch chahiye.)\n")

    # ---------------- Step 5: save ----------------
    head("STEP 5: Predictions save")
    pd.DataFrame({f"{ARM_NAMES[a]}__{m}": oof[(a, m)] for a in ARMS for m in METHODS},
                 index=Xtr.index).to_csv(PROCESSED / "cate_oof_train.csv", index_label="row")
    pd.DataFrame({f"{ARM_NAMES[a]}__{m}": test[(a, m)] for a in ARMS for m in METHODS},
                 index=Xte.index).to_csv(PROCESSED / "cate_test.csv", index_label="row")
    print(f"   [save hua] {PROCESSED / 'cate_oof_train.csv'}  ({n} mareez x {len(ARMS)*len(METHODS)} andaze)")
    print(f"   [save hua] {PROCESSED / 'cate_test.csv'}  ({nte} mareez, sirf predictions)")
    print(f"   [save hua] {TAB_DIR / 'cate_summary.csv'}")

    # ---------------- Step 6: graph ----------------
    head("STEP 6: Graph")
    fig = plt.figure(figsize=(20, 11.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    for k_, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs[0, k_])
        data = [oof[(arm, m)] for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color="#222222", lw=1.6))
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(MCOL[m]); patch.set_alpha(0.9)
        ate = ate_rows[arm - 1]["KM_RMST_fark"]
        ax.axhline(ate, color="#333333", ls="--", lw=1.3)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticklabels(METHODS, rotation=20)
        ax.set_ylabel(f"Asar (extra event-free din, {TAU} din tak)")
        ax.set_title(f"{k_+1}. {ARM_NAMES[arm]} vs ZDV: mareezon ka asar")
        if k_ == 0:
            ax.legend(handles=[Line2D([0], [0], color="#333333", ls="--", label="Poore trial ka average asar")], loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(METHODS, rotation=30, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(METHODS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if abs(corr[i, j]) > 0.6 else "#222222")
    ax.spines[:].set_visible(False)
    cb_ = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.03); cb_.set_label("Spearman correlation")
    ax.set_title("4. Tareeqe ek dusre se kitne milte hain")

    ax = fig.add_subplot(gs[1, 1])
    order = cf.abs().mean(axis=1).sort_values().index
    for k_, arm in enumerate(ARMS):
        ax.scatter(cf.loc[order, ARM_NAMES[arm]], range(len(order)), color=ARM_COLORS[arm], s=55,
                   label=ARM_NAMES[arm], zorder=3, alpha=0.9)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    ax.set_xlabel("Din (1 SD barhne par asar mein tabdeeli)")
    ax.legend(loc="lower right"); ax.set_title("5. DR-Ridge: kaun sa feature asar badalta hai")

    ax = fig.add_subplot(gs[1, 2])
    arm = 1
    b = pd.qcut(Xtr["cd40"], 10, duplicates="drop")
    for m in METHODS:
        g = pd.DataFrame({"cd40": Xtr["cd40"], "cate": oof[(arm, m)]}).groupby(b, observed=True).agg(
            x=("cd40", "median"), y=("cate", "mean"))
        ax.plot(g["x"], g["y"], "-o", color=MCOL[m], lw=2, ms=5, label=m)
    ax.axhline(ate_rows[0]["KM_RMST_fark"], color="#333333", ls="--", lw=1.3)
    ax.set_xlabel("cd40 (har point = 10% mareezon ka group)")
    ax.set_ylabel(f"Average asar ({ARM_NAMES[arm]} vs ZDV, din)")
    ax.legend(loc="best", ncol=2); ax.set_title("6. cd40 ke saath asar kaise badalta hai (ZDV+ddI)")

    fig.suptitle("CATE: treatment ka asar har mareez par (ACTG175, RMST din mein)", fontsize=16, fontweight="bold")
    fig.savefig(FIG_DIR / "cate_1_overview.png")
    print(f"   [graph save hua] {FIG_DIR / 'cate_1_overview.png'}")
    print(f"\nCATE mukammal ({time.time()-t0:.0f} sec). Agla qadam: validation.py (kya yeh farq asli hai ya shor?).")
    if SHOW:
        plt.show()