"""
FEATURE ENGINEERING - ACTG175 (sab kuch is ek file mein)
Chalane ka tareeqa (project folder se):
    python src/features.py

Tarteeb:
  Step 1: Train/Test split (sab se pehle)
  Step 2: Columns alag karna aur hatana (+ VIF check)
  Step 3: Encoding
  Step 4: Transform + outlier capping + scaling (Cox ke liye), faisla train par cross-validation se
  Step 5: Final fit (train par) aur test par lagana + checks
  Step 6: Overfit/underfit ka muaina
  Step 7: Save (do versions: trees ke liye raw, Cox ke liye transformed+scaled)

QAIDA: jo cheez data se SEEKHI jaye (mean, cap ki hadd, knots...) wo sirf TRAIN se seekhi jaye.
Test par sirf lagayi jaye. Test data ko dekh kar koi faisla nahi hoga.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIG: faisle yahan badalein (baqi code ko chhene ki zaroorat nahi)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # project ka root (kisi bhi folder se chale)
PROCESSED = ROOT / "data" / "processed"
SEED = 42
TEST_SIZE = 0.20

OUTCOME = ["days", "cens"]                          # outcome (features nahi)
TREATMENT = ["arms", "treat"]                       # treatment (features nahi)
DROP_ALWAYS = ["zprior", "str2"]                    # zprior constant hai, str2 strat se poori tarah tay hota hai
DROP_OVERLAP = ["strat"]                            # preanti se r = 0.96, preanti rakha. (z30 abhi rakha hai)
LEAKAGE_COLS = ["pidnum", "cd420", "cd820", "cd496", "r", "offtrt"]   # sirf check ke liye
KARNOF_MODE = "onehot"                              # "onehot" ya "ordinal"
KARNOF_GROUPS = {70: "70-80", 80: "70-80", 90: "90", 100: "100"}
KARNOF_ORDER = ["70-80", "90", "100"]               # 100 = reference (dummy trap se bachne ke liye)

# --- Step 4 ke liye ---
CONT_COLS = ["age", "wtkg", "preanti", "cd40", "cd80"]   # continuous features (scaling/capping inhi par)
LOG_COLS = ["cd80", "preanti"]                      # skew > 1, log1p lagega
DROP_COX_ONLY = ["oprior"]                          # sirf 47 mareez: Cox/NN version se hatana (trees mein rehta hai)
CAP_Q = 0.99                                        # upar ki capping ki hadd (percentile), train se
N_KNOTS = 3                                         # spline ke knots (3 = 1 extra term)
CV_SPLITS, CV_REPEATS = 5, 3                        # cross-validation (sirf train ke andar)
COX_PENALIZER = 0.01                                # halki ridge penalty (stability ke liye)
# Transform ke variants (aasan se mushkil ki tarteeb mein). Faisla CV C-index se hoga.
VARIANTS = {
    "A": dict(log_cols=(), cd40_mode="none", cap_q=None),
    "B": dict(log_cols=tuple(LOG_COLS), cd40_mode="none", cap_q=None),
    "C": dict(log_cols=tuple(LOG_COLS), cd40_mode="log1p", cap_q=None),
    "E": dict(log_cols=tuple(LOG_COLS), cd40_mode="log1p", cap_q=CAP_Q),
    "D": dict(log_cols=tuple(LOG_COLS), cd40_mode="spline", cap_q=None),
    "F": dict(log_cols=tuple(LOG_COLS), cd40_mode="spline", cap_q=CAP_Q),
}
VARIANT_TEXT = {
    "A": "sirf scaling (koi transform nahi)",
    "B": "log1p (cd80, preanti)",
    "C": "B + cd40 par log1p",
    "E": "C + upar ki capping (99%)",
    "D": "B + cd40 par spline",
    "F": "D + upar ki capping (99%)",
}
COMPLEXITY = ["A", "B", "C", "E", "D", "F"]          # saada se pechida


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


# ==================================================================
# STEP 1: TRAIN / TEST SPLIT
# Test data ko "naye mareez" ki tarah rakhte hain jinhe model ne kabhi nahi dekha.
# Split arms aur event (cens) dono ke hisaab se barabar (stratified) hota hai.
# ==================================================================
def split_train_test(df, test_size=TEST_SIZE, seed=SEED):
    strata = df["arms"].astype(str) + "_" + df["cens"].astype(str)
    tr, te = train_test_split(df.index, test_size=test_size, random_state=seed, stratify=strata)
    return sorted(tr), sorted(te)


def save_split(tr, te, path=PROCESSED / "split_ids.csv"):
    out = pd.concat([pd.DataFrame({"row": tr, "set": "train"}),
                     pd.DataFrame({"row": te, "set": "test"})]).sort_values("row")
    out.to_csv(path, index=False)
    return path


def split_report(train, test):
    rows = []
    for name, d in [("train", train), ("test", test)]:
        r = {"set": name, "mareez": len(d), "events": int(d["cens"].sum()),
             "event_%": round(d["cens"].mean() * 100, 1)}
        for a in range(4):
            r[f"arm{a}_%"] = round((d["arms"] == a).mean() * 100, 1)
        rows.append(r)
    return pd.DataFrame(rows)


# ==================================================================
# STEP 2: COLUMNS ALAG KARNA AUR HATANA
# Outcome (days, cens) aur treatment (arms, treat) features (X) mein shamil nahi hote.
# VIF = ek feature baqi features se kitna "dohra" raha hai. 5 se upar mushkil, 10 se upar
# saaf masla (multicollinearity).
# ==================================================================
def split_columns(d):
    X = d.drop(columns=OUTCOME + TREATMENT)
    return X, d[OUTCOME].copy(), d[TREATMENT].copy()


def vif_table(X):
    """Har column ka VIF (train par). Constant column ho to nikal dete hain."""
    Xn = X.loc[:, X.nunique() > 1].astype(float)
    out = {}
    for c in Xn.columns:
        y = Xn[c].values
        A = np.column_stack([np.ones(len(Xn)), Xn.drop(columns=c).values])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[c] = 1 / max(1 - r2, 1e-9)
    return pd.Series(out).sort_values(ascending=False).round(2)


def drop_columns(X, cols):
    return X.drop(columns=[c for c in cols if c in X.columns])


# ==================================================================
# STEP 3: ENCODING
# karnof ke sirf 4 values hain (70 par sirf 9 mareez), isliye 70 aur 80 ko milakar 3 groups:
# 70-80, 90, 100. Groups domain ke qaide se tay hain (data se seekhe nahi), is liye leakage nahi.
#   onehot : 2 dummy columns (100 reference, dono 0 = karnof 100) - Cox ke liye bhi theek
#   ordinal: ek hi column (0, 1, 2) - kam parameter, kam overfit
# Binary columns pehle se 0/1 hain, unhe nahi chhedte.
# ==================================================================
def encode(X, mode=KARNOF_MODE):
    X = X.copy()
    if "karnof" not in X.columns:
        return X
    grp = pd.Categorical(X["karnof"].map(KARNOF_GROUPS), categories=KARNOF_ORDER)
    pos = X.columns.get_loc("karnof")
    X = X.drop(columns="karnof")
    if mode == "onehot":
        for k, g in enumerate(["70-80", "90"]):
            X.insert(pos + k, f"karnof_{g.replace('-', '_')}", (grp == g).astype(int))
    else:
        X.insert(pos, "karnof_level", pd.Series(grp.codes, index=X.index))     # 70-80:0, 90:1, 100:2
    return X



# ==================================================================
# STEP 4: TRANSFORM + OUTLIER CAPPING + SCALING (Cox / neural network ke liye)
# Tarteeb: log1p -> upar ki capping -> spline (cd40) -> scaling. Sab kuch SIRF TRAIN se seekha jata hai.
#   log1p   : bari values ko dabata hai (zero ko zero hi rakhta hai). cd80/preanti ki lambi dum ke liye
#   capping : 99th percentile se upar ki values ko us hadd par rok dena (outlier hatana nahi)
#   spline  : cd40 ke mor ke liye (restricted cubic spline, 3 knots = linear + 1 extra term)
#   scaling : mean 0, SD 1 (sirf continuous columns par; 0/1 columns jaise hain)
# Trees (RSF, boosting) ko yeh sab nahi chahiye, unke liye raw version alag save hota hai.
# ==================================================================
def rcs_terms(x, knots):
    """Restricted cubic spline ke nonlinear terms (Harrell). 3 knots par 1 term, 4 par 2."""
    t = np.asarray(knots, float); k = len(t)
    pos = lambda u: np.clip(u, 0, None) ** 3
    out = []
    for j in range(k - 2):
        term = (pos(x - t[j])
                - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])) / (t[k - 1] - t[0]) ** 2
        out.append(term)
    return out


class PreprocessCox:
    def __init__(self, log_cols=tuple(LOG_COLS), cd40_mode="log1p", cap_q=None, n_knots=N_KNOTS,
                 drop_cols=tuple(DROP_COX_ONLY), scale=True):
        self.log_cols, self.cd40_mode, self.cap_q = tuple(log_cols), cd40_mode, cap_q
        self.n_knots, self.drop_cols, self.scale = n_knots, tuple(drop_cols), scale
        self.caps_, self.knots_, self.mean_, self.std_ = {}, [], {}, {}

    def _log(self, X):
        Z = X.drop(columns=[c for c in self.drop_cols if c in X.columns]).astype(float)
        cols = list(self.log_cols) + (["cd40"] if self.cd40_mode == "log1p" else [])
        for c in cols:
            if c in Z.columns:
                Z[c] = np.log1p(Z[c])
        return Z

    def _cap(self, Z):
        Z = Z.copy()
        for c, hi in self.caps_.items():
            Z[c] = Z[c].clip(upper=hi)
        return Z

    def _spline(self, Z):
        if self.cd40_mode != "spline":
            return Z
        Z = Z.copy(); pos = Z.columns.get_loc("cd40") + 1
        for i, term in enumerate(rcs_terms(Z["cd40"].values, self.knots_), 1):
            Z.insert(pos + i - 1, f"cd40_s{i}", term)
        return Z

    def fit(self, X):
        Z = self._log(X)
        if self.cap_q:
            self.caps_ = {c: float(Z[c].quantile(self.cap_q)) for c in CONT_COLS if c in Z.columns}
        Z = self._cap(Z)
        if self.cd40_mode == "spline":
            pct = [10, 50, 90] if self.n_knots == 3 else [5, 35, 65, 95]
            self.knots_ = [float(np.percentile(Z["cd40"], p)) for p in pct]
        Z = self._spline(Z)
        if self.scale:
            sc = [c for c in Z.columns if c in CONT_COLS or c.startswith("cd40_s")]
            self.mean_ = {c: float(Z[c].mean()) for c in sc}
            self.std_ = {c: float(Z[c].std(ddof=0)) for c in sc}
        return self

    def transform(self, X):
        Z = self._spline(self._cap(self._log(X)))
        for c in self.mean_:
            Z[c] = (Z[c] - self.mean_[c]) / self.std_[c]
        return Z

    def to_dict(self):
        return dict(log_cols=list(self.log_cols), cd40_mode=self.cd40_mode, cap_q=self.cap_q, n_knots=self.n_knots,
                    drop_cols=list(self.drop_cols), scale=self.scale, caps=self.caps_, knots=self.knots_,
                    mean=self.mean_, std=self.std_)

    @classmethod
    def from_dict(cls, d):
        p = cls(d["log_cols"], d["cd40_mode"], d["cap_q"], d["n_knots"], d["drop_cols"], d["scale"])
        p.caps_, p.knots_, p.mean_, p.std_ = d["caps"], d["knots"], d["mean"], d["std"]
        return p


def cv_cindex(X, y, strata, cfg, seed=SEED):
    """Sirf TRAIN ke andar repeated cross-validation. Preprocessing har fold ke train hisse par
    seekhi jati hai, validation hisse par sirf lagayi jati hai (leakage nahi).
    Do naap: C-index (ranking kitni achhi) aur partial log-likelihood (nonlinear shakal ke liye
    zyada hassas; bara = behtar)."""
    rskf = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=seed)
    scores, pls = [], []
    for a, b in rskf.split(X, strata):
        prep = PreprocessCox(**cfg).fit(X.iloc[a])
        Za, Zb = prep.transform(X.iloc[a]), prep.transform(X.iloc[b])
        d = Za.assign(days=y["days"].iloc[a].values, cens=y["cens"].iloc[a].values)
        cph = CoxPHFitter(penalizer=COX_PENALIZER).fit(d, "days", "cens")
        risk = cph.predict_partial_hazard(Zb).values
        scores.append(concordance_index(y["days"].iloc[b].values, -risk, y["cens"].iloc[b].values))
        db = Zb.assign(days=y["days"].iloc[b].values, cens=y["cens"].iloc[b].values)
        pls.append(cph.score(db, scoring_method="log_likelihood"))     # validation par partial log-likelihood
    return np.array(scores), np.array(pls)


def choose_variant(results):
    """Sab se saada variant jo sab se achhe score se 1 standard error ke andar ho."""
    means = {k: v.mean() for k, v in results.items()}
    best = max(means, key=means.get)
    se_best = results[best].std() / np.sqrt(CV_SPLITS)
    for k in COMPLEXITY:
        if means[k] >= means[best] - se_best:
            return k, best, se_best


# ==================================================================
# CHECKS (har step ke baad)
# ==================================================================
def check_features(Xtr, Xte):
    leak = [c for c in LEAKAGE_COLS + OUTCOME + TREATMENT if c in Xtr.columns]
    ok = {
        "Leakage/outcome/treatment columns nahi": len(leak) == 0,
        "Train missing = 0": int(Xtr.isna().sum().sum()) == 0,
        "Test missing = 0": int(Xte.isna().sum().sum()) == 0,
        "Train aur test ke columns barabar (aur ek tarteeb mein)": list(Xtr.columns) == list(Xte.columns),
        "Sab columns numeric": all(pd.api.types.is_numeric_dtype(t) for t in Xtr.dtypes),
        "Train mein koi constant column nahi": bool((Xtr.nunique() > 1).all()),
    }
    for k, v in ok.items():
        print(f"   [{'OK' if v else 'GADBAD'}] {k}")
    if leak:
        print("   Leakage wale columns mile:", leak)
    return all(ok.values())


# ==================================================================
# CHALANA
# ==================================================================
if __name__ == "__main__":
    df = pd.read_csv(PROCESSED / "actg175_clean.csv")
    print(f"Project root: {ROOT}\nCleaned data: {df.shape[0]} mareez, {df.shape[1]} columns\n")

    # ---- Step 1 ----
    head("STEP 1: Train/Test split")
    tr_idx, te_idx = split_train_test(df)
    p = save_split(tr_idx, te_idx)
    train, test = df.loc[tr_idx], df.loc[te_idx]
    print(split_report(train, test).to_string(index=False))
    print(f"Overlap (0 hona chahiye): {len(set(train.index) & set(test.index))}")
    print(f"   [save hua] {p}\n")

    # ---- Step 2 ----
    head("STEP 2: Columns alag karna aur hatana")
    Xtr, ytr, ttr = split_columns(train)
    Xte, yte, tte = split_columns(test)
    print(f"Outcome: {OUTCOME} | Treatment: {TREATMENT} | Baseline features (X): {Xtr.shape[1]}")
    print(f"X ke columns: {list(Xtr.columns)}\n")

    Xtr, Xte = drop_columns(Xtr, DROP_ALWAYS), drop_columns(Xte, DROP_ALWAYS)
    print(f"Hataye (constant/duplicate): {DROP_ALWAYS}")
    v1 = vif_table(Xtr)
    print("\nVIF (sirf TRAIN par, strat hatane se PEHLE):")
    print(v1.to_string())

    Xtr, Xte = drop_columns(Xtr, DROP_OVERLAP), drop_columns(Xte, DROP_OVERLAP)
    print(f"\nHataye (overlap): {DROP_OVERLAP}")
    v2 = vif_table(Xtr)
    print("VIF (strat hatane ke BAAD):")
    print(v2.to_string())
    print(f"\nSab se bara VIF: pehle {v1.max():.1f} -> ab {v2.max():.1f}")
    print(f"Ab X mein {Xtr.shape[1]} columns: {list(Xtr.columns)}\n")

    # ---- Step 3 ----
    head("STEP 3: Encoding")
    print(f"karnof ke groups (train): {pd.Series(Xtr['karnof'].map(KARNOF_GROUPS)).value_counts().to_dict()}")
    Xtr, Xte = encode(Xtr), encode(Xte)
    print(f"karnof mode: {KARNOF_MODE}")
    print(f"Ab X mein {Xtr.shape[1]} columns: {list(Xtr.columns)}")
    print(f"X_train: {Xtr.shape} | X_test: {Xte.shape}\n")

    # ---- Checks ----
    head("CHECKS")
    good = check_features(Xtr, Xte)
    print("\nSab checks theek hain." if good else "\nKuch check fail hua, upar dekhein.")
    X_trees_tr, X_trees_te = Xtr.copy(), Xte.copy()          # Version A (trees): raw, koi transform nahi

    # ---- Step 4 ----
    head("STEP 4: Transform + capping + scaling (faisla train par cross-validation se)")
    strata = ttr["arms"].astype(str) + "_" + ytr["cens"].astype(str)
    print(f"Cross-validation: {CV_SPLITS}-fold x {CV_REPEATS} baar, sirf TRAIN ke andar. Model: Cox (penalizer {COX_PENALIZER}).")
    print(f"Cox version mein hataya jayega: {DROP_COX_ONLY} (sirf {int(Xtr['oprior'].sum())} mareez)\n")
    results, pl_results = {}, {}
    for k in COMPLEXITY:
        results[k], pl_results[k] = cv_cindex(Xtr, ytr, strata, VARIANTS[k])
    best_k, top_k, se_top = choose_variant(results)
    print("   variant | C-index (mean +- SE) | log-likelihood (mean +- SE) | tafseel")
    for k in COMPLEXITY:
        r, q = results[k], pl_results[k]
        mark = "   <-- chuna gaya" if k == best_k else ("   (sab se achha)" if k == top_k and k != best_k else "")
        print(f"   {k}       | {r.mean():.4f} +- {r.std()/np.sqrt(CV_SPLITS):.4f}   | {q.mean():.4f} +- {q.std()/np.sqrt(CV_SPLITS):.4f}"
              f"         | {VARIANT_TEXT[k]}{mark}")
    print(f"\nFaisla C-index se: sab se saada variant jo sab se achhe ({top_k}) ke 1 standard error ({se_top:.4f}) ke andar ho.")
    print(f"Chuna gaya variant: {best_k} = {VARIANT_TEXT[best_k]}")
    pl_best = max(pl_results, key=lambda k: pl_results[k].mean())
    se_pl = pl_results[pl_best].std() / np.sqrt(CV_SPLITS)
    agree = pl_results[best_k].mean() >= pl_results[pl_best].mean() - se_pl
    print(f"Log-likelihood bhi ek jaisa kehta hai? {'haan' if agree else 'nahi'} "
          f"(sab se achha {pl_best}, chuna hua {best_k} uske 1 SE ke {'andar' if agree else 'bahar'})")
    print("Matlab: transform lagane se (log1p, spline, capping) Cox ko koi saaf fayda nahi hua."
          if best_k == "A" else "Matlab: chuna hua transform lagayenge, baqi nahi.")

    # ---- Step 5 ----
    head("STEP 5: Final fit (TRAIN par) aur test par lagana")
    prep = PreprocessCox(**VARIANTS[best_k]).fit(Xtr)
    Ztr, Zte = prep.transform(Xtr), prep.transform(Xte)
    print("Seekhi hui cheezein (sirf train se):")
    if prep.caps_:
        print("   capping ki upar ki hadd:", {k: round(v, 3) for k, v in prep.caps_.items()})
    if prep.knots_:
        print("   spline ke knots (cd40):", [round(v, 1) for v in prep.knots_])
    print("   scaling (mean, SD):", {k: (round(prep.mean_[k], 2), round(prep.std_[k], 2)) for k in prep.mean_})

    Zraw = prep._log(Xtr)
    if best_k == "A":
        print("\nKoi transform nahi lagaya (variant A). Skewness (train) wahi rehti hai: "
              + ", ".join(f"{c} {skew(Xtr[c]):.2f}" for c in ["cd80", "preanti", "cd40"]))
        print("Outliers jaise hain rakhe gaye (sirf scaling hui).")
    else:
        print("\nSkewness (train): pehle -> baad (cd80, preanti, cd40)")
        for c in ["cd80", "preanti", "cd40"]:
            print(f"   {c}: {skew(Xtr[c]):.2f} -> {skew(Zraw[c]):.2f}")
    if prep.caps_:
        n_tr = {c: int((prep._log(Xtr)[c] > hi).sum()) for c, hi in prep.caps_.items()}
        n_te = {c: int((prep._log(Xte)[c] > hi).sum()) for c, hi in prep.caps_.items()}
        print(f"\nCapping se kitne mareez ki value rok di gayi (train): {n_tr}")
        print(f"Test mein train ki hadd se upar kitne (wohi hadd lagayi): {n_te}")

    print("\nChecks (Cox version):")
    good2 = check_features(Ztr, Zte)
    cont = [c for c in prep.mean_]
    print(f"   [{'OK' if np.allclose(Ztr[cont].mean(), 0, atol=1e-8) and np.allclose(Ztr[cont].std(ddof=0), 1, atol=1e-8) else 'GADBAD'}] Train par scaled columns: mean 0, SD 1")
    print(f"   [{'OK' if np.isfinite(Zte.values).all() and np.isfinite(Ztr.values).all() else 'GADBAD'}] Koi inf/NaN nahi")
    binary_cols = [c for c in Ztr.columns if c not in cont]
    print(f"   [{'OK' if all(set(Ztr[c].unique()) <= {0, 1} for c in binary_cols) else 'GADBAD'}] 0/1 columns scale nahi hue")
    print(f"\nCox version ke columns ({Ztr.shape[1]}): {list(Ztr.columns)}")
    print(f"Trees version ke columns ({X_trees_tr.shape[1]}): {list(X_trees_tr.columns)}")

    # ---- Step 6 ----
    head("STEP 6: Overfit / underfit ka muaina (Cox, chuna hua variant)")
    d_full = Ztr.assign(days=ytr["days"].values, cens=ytr["cens"].values)
    cph = CoxPHFitter(penalizer=COX_PENALIZER).fit(d_full, "days", "cens")
    c_train = concordance_index(ytr["days"].values, -cph.predict_partial_hazard(Ztr).values, ytr["cens"].values)
    c_cv = results[best_k].mean()
    gap = c_train - c_cv
    print(f"Train C-index: {c_train:.4f} | Cross-validation C-index: {c_cv:.4f} | farq: {gap:.4f}")
    if gap > 0.05:
        print("   Farq 0.05 se zyada: overfit ka shak (features/spline kam karein ya penalty barhayein).")
    elif c_cv < 0.6:
        print("   C-index kam hai: underfit ka shak (nonlinear terms ya features badhayein).")
    else:
        print("   Farq chhota hai aur C-index munasib: overfit ya underfit ka saaf nishan nahi.")
    print("   (Test data abhi tak khola nahi gaya. Woh sirf akhri model ke baad ek baar khulega.)")

    # ---- Step 7 ----
    head("STEP 7: Save")
    files = {
        "X_train_trees.csv": X_trees_tr, "X_test_trees.csv": X_trees_te,
        "X_train_cox.csv": Ztr, "X_test_cox.csv": Zte,
        "y_train.csv": ytr, "y_test.csv": yte,
        "treatment_train.csv": ttr, "treatment_test.csv": tte,
    }
    for name, frame in files.items():
        frame.to_csv(PROCESSED / name, index_label="row")
        print(f"   [save hua] {PROCESSED / name}  {frame.shape}")
    cfg_out = dict(variant=best_k, variant_text=VARIANT_TEXT[best_k], seed=SEED, test_size=TEST_SIZE,
                   dropped_always=DROP_ALWAYS, dropped_overlap=DROP_OVERLAP, dropped_cox_only=DROP_COX_ONLY,
                   karnof_mode=KARNOF_MODE, preprocessor=prep.to_dict(),
                   cv={k: dict(cindex=[round(float(results[k].mean()), 4), round(float(results[k].std() / np.sqrt(CV_SPLITS)), 4)],
                               loglik=[round(float(pl_results[k].mean()), 4), round(float(pl_results[k].std() / np.sqrt(CV_SPLITS)), 4)])
                       for k in COMPLEXITY})
    with open(PROCESSED / "preprocess_cox.json", "w", encoding="utf-8") as fh:
        json.dump(cfg_out, fh, indent=2)
    print(f"   [save hua] {PROCESSED / 'preprocess_cox.json'}")
    print("\nFeature engineering mukammal. Agla qadam: notebook 04 (overall effect) aur 05 (prediction models).")