"""
EDA (Step 1 se 10) - ACTG175
  Part 1: Step 1-5  (outcome, distribution, bekaar features, balance, censoring)
  Part 2: Step 6-10 (rishte, Kaplan-Meier, subgroups, patterns)
  Part 3: Step 11-12 (EDA ke saare nateeje + feature engineering ki list + ek-page graph)
Chalane ka tareeqa (project folder se):
    python src/eda.py
Sirf PNG save karni hon (window na khule):  ... eda.py --no-show
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy.stats import chi2_contingency, skew, spearmanr
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# 0. PATH: project ka root khud dhoondhta hai (folder ka path error khatam)
# ------------------------------------------------------------------
def find_root():
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for p in [start, *start.parents]:
        if (p / "data" / "processed" / "actg175_clean.csv").exists():
            return p
    raise FileNotFoundError(
        "data/processed/actg175_clean.csv nahi mili. Pehle cleaning wala code chala kar "
        "file save karein.")


ROOT = find_root()
FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SHOW = "--no-show" not in sys.argv

# ------------------------------------------------------------------
# Style: grid band, saaf aur professional rang
# ------------------------------------------------------------------
plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.frameon": False, "figure.dpi": 100,
    "savefig.dpi": 150, "savefig.bbox": "tight",
})
EVENT_C, CENS_C = "#C8553D", "#5B8DB8"
GOOD_C, BAD_C, NEUTRAL_C = "#3A9D8F", "#C8553D", "#2F5D8A"
FEAT_COLORS = ["#2F5D8A", "#3A9D8F", "#E4893B", "#8E5BA8", "#C8553D"]
ARM_NAMES = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+zal", 3: "ddI"}
ARM_COLORS = {0: "#2F5D8A", 1: "#E4893B", 2: "#3A9D8F", 3: "#8E5BA8"}
GREY_C = "#9AA5B1"
PAL4 = ["#C8553D", "#E4A13B", "#3A9D8F", "#2F5D8A"]   # kam se zyada
PAL3 = ["#C8553D", "#3A9D8F", "#2F5D8A"]
TREAT_C = {0: "#2F5D8A", 1: "#E4893B"}


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path)
    print(f"   [graph save hua] {path}")


# ------------------------------------------------------------------
# Data load
# ------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "processed" / "actg175_clean.csv")
print(f"Project root: {ROOT}")
print(f"Data: {df.shape[0]} mareez, {df.shape[1]} columns\n")

CONT = ["age", "wtkg", "preanti", "cd40", "cd80"]
BINARY = ["hemo", "homo", "drugs", "oprior", "z30", "zprior",
          "race", "gender", "str2", "symptom", "treat"]
CATEG = ["strat", "karnof", "arms"]


# ==================================================================
# STEP 1: OUTCOME KA JAIZA
# Outcome = trial ka nateeja. Yahan "event" ka matlab hai: CD4 mein kami,
# AIDS ki taraf barhna, ya death (in mein se jo pehle ho).
# "Censored" = mareez ke saath event hua hi nahi (ya follow-up beech mein ruk gaya).
# ==================================================================
print("=" * 60, "\nSTEP 1: Outcome ka jaiza\n" + "=" * 60)
n_ev = int(df["cens"].sum()); n_ce = len(df) - n_ev
print(f"Event hua:     {n_ev} ({n_ev/len(df)*100:.1f}%)")
print(f"Censored:      {n_ce} ({n_ce/len(df)*100:.1f}%)")
print(f"Follow-up days: min {df['days'].min()}, median {df['days'].median():.0f}, max {df['days'].max()}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
ax.hist([df.loc[df.cens == 1, "days"], df.loc[df.cens == 0, "days"]], bins=30, stacked=True,
        color=[EVENT_C, CENS_C], edgecolor="white", label=["Event hua", "Censored"])
ax.axvline(df["days"].median(), color="#333333", ls="--", lw=1.2,
           label=f"Median = {df['days'].median():.0f} din")
ax.set_xlabel("Follow-up (din)"); ax.set_ylabel("Mareezon ki tadaad")
ax.set_title("Follow-up ka waqt"); ax.legend(loc="upper left")

ax = axes[1]
vals = [n_ev, n_ce]
ax.barh(["Event hua", "Censored"], vals, color=[EVENT_C, CENS_C], height=0.55)
for i, v in enumerate(vals):
    ax.text(v + max(vals) * 0.02, i, f"{v}  ({v/len(df)*100:.1f}%)", va="center")
ax.set_xlim(0, max(vals) * 1.35); ax.set_xlabel("Mareezon ki tadaad")
ax.set_title("Event vs Censored"); ax.invert_yaxis()
fig.suptitle("Step 1: Outcome ka jaiza", fontsize=14, fontweight="bold")
save(fig, "eda_1_outcome.png")


# ==================================================================
# STEP 2: HAR FEATURE KA DISTRIBUTION AUR SKEWNESS
# Distribution = values kahan zyada hain, kahan kam.
# Skewness = tarazu ek taraf jhuka hua hai ya nahi. 0 = barabar, +1 se zyada = lambi
# dum right taraf (kuch bohot bari values), -1 se kam = lambi dum left taraf.
# ==================================================================
print("=" * 60, "\nSTEP 2: Features ka distribution\n" + "=" * 60)
sk = pd.Series({c: skew(df[c]) for c in CONT}).sort_values()
print("Skewness (continuous):\n" + sk.round(2).to_string() + "\n")

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for ax, c, col in zip(axes.ravel()[:5], CONT, FEAT_COLORS):
    ax.hist(df[c], bins=30, color=col, edgecolor="white")
    ax.axvline(df[c].median(), color="#333333", ls="--", lw=1.2, label=f"Median = {df[c].median():.0f}")
    ax.set_title(f"{c}  (skew {skew(df[c]):.2f})")
    ax.set_xlabel(c); ax.set_ylabel("Mareezon ki tadaad"); ax.legend(loc="best")
ax = axes.ravel()[5]
ax.barh(sk.index, sk.values, color=[BAD_C if abs(v) > 1 else GOOD_C for v in sk.values], height=0.55)
for i, v in enumerate(sk.values):
    ax.text(v + (0.08 if v >= 0 else -0.08), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right")
ax.axvline(0, color="#333333", lw=0.8)
ax.set_xlim(min(sk.min() - 0.6, -0.6), sk.max() + 1.9)
ax.set_title("Skewness ek nazar mein"); ax.set_xlabel("Skewness")
ax.legend(handles=[Patch(color=BAD_C, label="|skew| > 1 (zyada jhuka hua)"),
                   Patch(color=GOOD_C, label="|skew| <= 1 (theek)")], loc="lower right")
fig.suptitle("Step 2a: Continuous features ka distribution", fontsize=14, fontweight="bold")
save(fig, "eda_2a_continuous.png")

# binary + categorical
fig = plt.figure(figsize=(15, 8), constrained_layout=True)
gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])
ax = fig.add_subplot(gs[:, 0])
pct1 = (df[BINARY].mean() * 100).sort_values()
ax.barh(pct1.index, 100 - pct1.values, color=CENS_C, height=0.6, label="Value = 0")
ax.barh(pct1.index, pct1.values, left=100 - pct1.values, color=EVENT_C, height=0.6, label="Value = 1")
for i, v in enumerate(pct1.values):
    ax.text(100 - v / 2 if v > 6 else 100 - v - 1.5, i, f"{v:.1f}%",
            va="center", ha="center" if v > 6 else "right",
            color="white" if v > 6 else "#333333", fontsize=9)
ax.set_xlim(0, 100); ax.set_xlabel("Mareezon ka hissa (%)")
ax.set_title("Binary features: 0 aur 1 ka hissa")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

for k, (c, col) in enumerate(zip(CATEG, [NEUTRAL_C, GOOD_C, "#E4893B"])):
    ax = fig.add_subplot(gs[k, 1])
    vc = df[c].value_counts().sort_index()
    labels = [ARM_NAMES.get(i, str(i)) if c == "arms" else str(i) for i in vc.index]
    ax.bar(labels, vc.values, color=col, width=0.6)
    for i, v in enumerate(vc.values):
        ax.text(i, v + vc.max() * 0.02, str(v), ha="center", fontsize=9)
    ax.set_ylim(0, vc.max() * 1.18)
    ax.set_title(f"{c}: kitne mareez har value par"); ax.set_ylabel("Tadaad")
fig.suptitle("Step 2b: Binary aur categorical features", fontsize=14, fontweight="bold")
save(fig, "eda_2b_binary_categorical.png")


# ==================================================================
# STEP 3: BEKAAR YA DUPLICATE FEATURES
# Bekaar = lagbhag sab mareezon ki ek hi value (model ko kuch sikha nahi sakta).
# Duplicate = ek feature doosre se poori tarah tay ho jata hai (dono rakhna fuzool).
# ==================================================================
print("=" * 60, "\nSTEP 3: Bekaar / duplicate features\n" + "=" * 60)
feats = [c for c in df.columns if c not in ("days", "cens")]
rows = []
for c in feats:
    vc = df[c].value_counts(normalize=True)
    rows.append({"feature": c, "unique_values": df[c].nunique(),
                 "sab_se_aam_value_%": round(vc.iloc[0] * 100, 1)})
dom = pd.DataFrame(rows).sort_values("sab_se_aam_value_%", ascending=False).reset_index(drop=True)
dom["flag"] = np.where(dom["unique_values"] == 1, "CONSTANT",
                       np.where(dom["sab_se_aam_value_%"] >= 95, "lagbhag constant", "theek"))
print(dom.to_string(index=False))

print("\nDuplicate check (A se B poori tarah tay ho jata hai):")
small = [c for c in feats if 1 < df[c].nunique() <= 10]
dups = []
for a in small:
    for b in small:
        if a != b and (df.groupby(a)[b].nunique() == 1).all():
            dups.append((a, b))
for a, b in dups:
    print(f"   {a}  ->  {b}  (agar {a} pata ho to {b} khud pata chal jata hai)")
if not dups:
    print("   koi nahi mila")
print("\nCrosstab str2 vs strat:")
print(pd.crosstab(df["strat"], df["str2"]).to_string() + "\n")

dsort = dom.sort_values("sab_se_aam_value_%")
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
cols = [BAD_C if f in ("CONSTANT", "lagbhag constant") else NEUTRAL_C for f in dsort["flag"]]
ax.barh(dsort["feature"], dsort["sab_se_aam_value_%"], color=cols, height=0.65)
ax.axvline(95, color="#333333", ls="--", lw=1.2)
for i, v in enumerate(dsort["sab_se_aam_value_%"]):
    ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlim(0, 112); ax.set_xlabel("Sab se aam value ka hissa (%)")
ax.set_title("Step 3: Kaun sa feature lagbhag constant hai")
ax.legend(handles=[Patch(color=BAD_C, label="95% ya zyada ek hi value (bekaar ka shak)"),
                   Patch(color=NEUTRAL_C, label="theek"),
                   plt.Line2D([0], [0], color="#333333", ls="--", label="95% ki hadd")],
          loc="lower right")
save(fig, "eda_3_useless_features.png")


# ==================================================================
# STEP 4: BASELINE BALANCE (TABLE 1)
# Trial mein mareez randomly arms mein baante jate hain, to shuru mein chaaron arms
# lagbhag ek jaise hone chahiye. Yeh check wahi hai.
# SMD (standardized difference) = arm ka average aur sab ka average kitna door hai,
# SD ke hisaab se. 0.1 se kam = barabar, 0.1 se zyada = thoda fark.
# ==================================================================
print("=" * 60, "\nSTEP 4: Baseline balance (Table 1)\n" + "=" * 60)
base = df.drop(columns=["days", "cens", "arms", "treat"]).copy()
base = pd.concat([base.drop(columns=["strat"]),
                  pd.get_dummies(df["strat"], prefix="strat").astype(int)], axis=1)
base = base.loc[:, base.nunique() > 1]            # constant (zprior) nikal diya
arm = df["arms"]

t1 = {}
for c in base.columns:
    if c in CONT or c == "karnof":
        t1[c] = [f"{base.loc[arm == a, c].mean():.1f} ({base.loc[arm == a, c].std():.1f})" for a in range(4)]
    else:
        t1[c] = [f"{int(base.loc[arm == a, c].sum())} ({base.loc[arm == a, c].mean()*100:.0f}%)" for a in range(4)]
table1 = pd.DataFrame(t1, index=[f"{ARM_NAMES[a]} (n={int((arm == a).sum())})" for a in range(4)]).T
print("Continuous: mean (SD) | Binary: n (%)\n")
print(table1.to_string())
table1.to_csv(TAB_DIR / "eda_table1_baseline.csv")

smd = pd.DataFrame({a: [abs(base.loc[arm == a, c].mean() - base[c].mean()) / base[c].std()
                        for c in base.columns] for a in range(4)}, index=base.columns)
order = smd.max(axis=1).sort_values().index
smd = smd.loc[order]
print(f"\nSab se bara SMD: {smd.max().max():.3f}  ({smd.max(axis=1).idxmax()})")
print("SMD > 0.1 wale features:", list(smd.index[smd.max(axis=1) > 0.1]) or "koi nahi")

fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
for a in range(4):
    ax.scatter(smd[a], range(len(smd)), s=55, color=ARM_COLORS[a], label=ARM_NAMES[a], zorder=3, alpha=0.9)
ax.set_yticks(range(len(smd))); ax.set_yticklabels(smd.index)
ax.axvline(0.1, color="#333333", ls="--", lw=1.2, label="0.1 ki hadd")
ax.set_xlabel("Standardized difference (SMD)  -  chhota = arms barabar")
ax.set_title("Step 4: Baseline balance (har arm sab mareezon ke average se kitna door)")
ax.legend(loc="lower right", ncol=1)
save(fig, "eda_4_baseline_balance.png")


# ==================================================================
# STEP 5: CENSORING BY ARM
# Dekhna hai ke har arm mein kitne mareez bina event ke (censored) rahe aur woh kitne din
# follow hue. Agar kisi arm ke mareez bohot jaldi gayab ho jayein to muqabla na-insafi
# ban sakta hai. (Event ka hissa arms mein alag hona treatment ka asar bhi ho sakta hai.)
# ==================================================================
print("=" * 60, "\nSTEP 5: Censoring by arm\n" + "=" * 60)
g = df.groupby("arms")
cb = pd.DataFrame({
    "arm": [ARM_NAMES[a] for a in g.size().index],
    "n": g.size().values,
    "event": g["cens"].sum().values,
})
cb["censored"] = cb["n"] - cb["event"]
cb["censored_%"] = (cb["censored"] / cb["n"] * 100).round(1)
cb["median_followup_days"] = g["days"].median().values
cb["median_followup_censored"] = df[df.cens == 0].groupby("arms")["days"].median().values
print(cb.to_string(index=False))
chi, p, _, _ = chi2_contingency(pd.crosstab(df["arms"], df["cens"]))
print(f"\nChi-square p-value (kya arms mein event ka hissa alag hai?): {p:.4f}")
print("   Dhyan: yahan 'censored %' = 100 minus 'event %'. Agar arms mein yeh alag hai to")
print("   iska matlab hai ke treatment ka asar alag hai (ZDV mein zyada events), yeh dropout ki")
print("   shikayat nahi. Asal dropout ka ishara 'censored mareez kitne din follow hue' hai.")
print("   Censored mareezon ka median follow-up arms mein kitna alag hai:",
      f"{cb['median_followup_censored'].min():.0f} se {cb['median_followup_censored'].max():.0f} din")
cb.to_csv(TAB_DIR / "eda_censoring_by_arm.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
ax = axes[0]
x = np.arange(4); ev_pct = 100 - cb["censored_%"].values; ce_pct = cb["censored_%"].values
ax.bar(x, ev_pct, color=EVENT_C, width=0.6, label="Event hua")
ax.bar(x, ce_pct, bottom=ev_pct, color=CENS_C, width=0.6, label="Censored")
for i in range(4):
    ax.text(i, ev_pct[i] / 2, f"{ev_pct[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.text(i, ev_pct[i] + ce_pct[i] / 2, f"{ce_pct[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(cb["arm"]); ax.set_ylim(0, 100)
ax.set_ylabel("Mareezon ka hissa (%)"); ax.set_title("Har arm mein event vs censored")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

ax = axes[1]
data = [df.loc[(df.arms == a) & (df.cens == 0), "days"] for a in range(4)]
bp = ax.boxplot(data, patch_artist=True, widths=0.5, medianprops=dict(color="#333333", lw=1.5),
                flierprops=dict(marker="o", markersize=3, alpha=0.4))
for patch, a in zip(bp["boxes"], range(4)):
    patch.set_facecolor(ARM_COLORS[a]); patch.set_alpha(0.85)
ax.set_xticklabels(cb["arm"]); ax.set_ylabel("Follow-up (din)")
ax.set_title("Censored mareez kitne din tak follow hue")
ax.legend(handles=[Patch(color=ARM_COLORS[a], label=ARM_NAMES[a]) for a in range(4)], loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
fig.suptitle("Step 5: Censoring by arm", fontsize=14, fontweight="bold")
save(fig, "eda_5_censoring_by_arm.png")


# ##################################################################
#                    PART 2: STEP 6 SE 10
# ##################################################################
OVERALL = df["cens"].mean() * 100


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


def logrank_p(groups):
    return multivariate_logrank_test(df.loc[groups.index, "days"], groups, df.loc[groups.index, "cens"]).p_value


# ==================================================================
# STEP 6a: FEATURES KA AAPAS KA RISHTA (CORRELATION)
# Correlation = do cheezein saath saath barhti/ghatti hain ya nahi. +1 = dono saath barhein,
# -1 = ek barhe to doosri ghate, 0 = koi rishta nahi. (Spearman: values ki tarteeb dekhta hai,
# is liye bari values se zyada mutassir nahi hota.)
# ==================================================================
head("STEP 6a: Features ka aapas ka rishta (correlation)")
SP = ["age", "wtkg", "karnof", "preanti", "cd40", "cd80", "hemo", "homo", "drugs",
      "oprior", "z30", "race", "gender", "symptom", "strat"]
corr = df[SP].corr(method="spearman")
mask = np.triu(np.ones_like(corr, dtype=bool))
pairs = (corr.where(~mask).stack().rename("r").reset_index()
         .rename(columns={"level_0": "a", "level_1": "b"}))
pairs["abs"] = pairs["r"].abs()
top = pairs.sort_values("abs", ascending=False).head(8).iloc[::-1]
print("Sab se zyada juray hue 8 jore (Spearman r):")
print(top.sort_values("abs", ascending=False)[["a", "b", "r"]].round(2).to_string(index=False), "\n")

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
data = np.ma.array(corr.values, mask=mask)
im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(SP))); ax.set_xticklabels(SP, rotation=45, ha="right")
ax.set_yticks(range(len(SP))); ax.set_yticklabels(SP)
for i in range(len(SP)):
    for j in range(i):
        v = corr.values[i, j]
        if abs(v) >= 0.2:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "#222222")
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("Spearman correlation")
ax.set_title("Sab features ka heatmap (sirf 0.2 se bade number likhe hain)")

ax = axes[1]
cols = [GOOD_C if r > 0 else BAD_C for r in top["r"]]
labels = [f"{a}  &  {b}" for a, b in zip(top["a"], top["b"])]
ax.barh(labels, top["r"], color=cols, height=0.6)
for i, v in enumerate(top["r"]):
    ax.text(v + (0.02 if v >= 0 else -0.02), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right")
ax.axvline(0, color="#333333", lw=0.8)
lim = max(0.15, top["abs"].max() * 1.25); ax.set_xlim(-lim, lim)
ax.set_xlabel("Correlation"); ax.set_title("Sab se mazboot 8 jore")
ax.legend(handles=[Patch(color=GOOD_C, label="Saath saath barhte hain (+)"),
                   Patch(color=BAD_C, label="Ek barhe to doosra ghatta hai (-)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1)
fig.suptitle("Step 6a: Features ka aapas ka rishta", fontsize=14, fontweight="bold")
save(fig, "eda_6a_correlation.png")


# ==================================================================
# STEP 6b: BINARY / CATEGORICAL FEATURES KA ASSOCIATION (Cramer's V)
# Jab features 0/1 ya categories hon to correlation ki jagah Cramer's V use hota hai.
# 0 = bilkul koi rishta nahi, 1 = ek feature doosre ko poori tarah tay kar deta hai.
# ==================================================================
head("STEP 6b: Binary/categorical features ka association (Cramer's V)")


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); phi2 = chi2 / n; r, k = ct.shape
    phi2c = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rc = r - ((r - 1) ** 2) / (n - 1); kc = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2c / max(min(kc - 1, rc - 1), 1e-9))


CAT = ["hemo", "homo", "drugs", "oprior", "z30", "race", "gender", "symptom", "strat", "karnof", "arms"]
V = pd.DataFrame(index=CAT, columns=CAT, dtype=float)
for a in CAT:
    for b in CAT:
        V.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
vm = np.triu(np.ones_like(V, dtype=bool))
vp = (V.where(~vm).stack().rename("v").reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
print("Sab se zyada juray hue 5 jore (Cramer's V):")
print(vp.sort_values("v", ascending=False).head(5).round(2).to_string(index=False))
arms_v = V.loc["arms", [c for c in CAT if c != "arms"]]
print(f"\n'arms' ka baaki features se sab se bara V: {arms_v.max():.3f} ({arms_v.idxmax()})")
print("   (0 ke qareeb hona achi baat hai: randomization ne arms ko features se alag rakha)\n")

fig, ax = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
im = ax.imshow(np.ma.array(V.values, mask=vm), cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(len(CAT))); ax.set_xticklabels(CAT, rotation=45, ha="right")
ax.set_yticks(range(len(CAT))); ax.set_yticklabels(CAT)
for i in range(len(CAT)):
    for j in range(i):
        v = V.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.5 else "#222222")
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("Cramer's V")
ax.set_title("Step 6b: Binary aur categorical features ka association")
save(fig, "eda_6b_cramers_v.png")


# ==================================================================
# STEP 7a: PEHLA KAPLAN-MEIER CURVE (ARMS KE HISAAB SE)
# Kaplan-Meier = ek curve jo dikhata hai ke waqt guzarne ke saath kitne mareez "event se bache
# hue" hain. Curve jitna oopar rahe, utna achha. Curves alag alag ho jayein to arms ka nateeja alag.
# Log-rank test = poochta hai ke curves ka fark ittifaq hai ya asli (p chhota = asli fark).
# ==================================================================
head("STEP 7a: Kaplan-Meier curve (arms ke hisaab se)")
T0 = 900
rows = []
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
ax = axes[0]
for a in range(4):
    m = df["arms"] == a
    k = KaplanMeierFitter().fit(df.loc[m, "days"], df.loc[m, "cens"], label=f"{ARM_NAMES[a]} (n={m.sum()})")
    k.plot_survival_function(ax=ax, color=ARM_COLORS[a], ci_alpha=0.10, lw=2)
    rows.append({"arm": ARM_NAMES[a], f"event_se_bachne_ka_imkaan_{T0}_din": round(float(k.predict(T0)), 3)})
p_arms = logrank_p(df["arms"])
ax.set_ylim(0.5, 1.02); ax.set_xlabel("Din (follow-up)"); ax.set_ylabel("Event se bache rehne ka imkaan")
ax.set_title(f"Chaaron arms (log-rank p = {p_arms:.2g})"); ax.legend(loc="lower left")

ax = axes[1]
for t, lab in [(0, "Sirf ZDV"), (1, "Koi doosra treatment")]:
    m = df["treat"] == t
    k = KaplanMeierFitter().fit(df.loc[m, "days"], df.loc[m, "cens"], label=f"{lab} (n={m.sum()})")
    k.plot_survival_function(ax=ax, color=TREAT_C[t], ci_alpha=0.12, lw=2)
p_tr = logrank_p(df["treat"])
ax.set_ylim(0.5, 1.02); ax.set_xlabel("Din (follow-up)"); ax.set_ylabel("Event se bache rehne ka imkaan")
ax.set_title(f"ZDV akela vs baqi sab (log-rank p = {p_tr:.2g})"); ax.legend(loc="lower left")
fig.suptitle("Step 7a: Kaplan-Meier curves", fontsize=14, fontweight="bold")
save(fig, "eda_7a_km_arms.png")
km_tab = pd.DataFrame(rows)
print(km_tab.to_string(index=False))
print(f"Log-rank p (4 arms): {p_arms:.3g} | (ZDV vs baqi): {p_tr:.3g}\n")
km_tab.to_csv(TAB_DIR / "eda_km_arms_900days.csv", index=False)


# ==================================================================
# STEP 7b: FEATURE AUR OUTCOME KA RISHTA (KM GROUPS KE HISAAB SE)
# Har feature ko 4 (ya 3) groups mein baant kar dekhte hain ke kis group ka survival behtar hai.
# Rang sirf tarteeb dikhata hai (laal = sab se kam value wala group, neela = sab se zyada).
# ==================================================================
head("STEP 7b: Feature aur outcome ka rishta (KM groups)")
g_cd40 = pd.qcut(df["cd40"], 4, labels=["Q1 (sab se kam)", "Q2", "Q3", "Q4 (sab se zyada)"])
g_cd80 = pd.qcut(df["cd80"], 4, labels=["Q1 (sab se kam)", "Q2", "Q3", "Q4 (sab se zyada)"])
g_age = pd.qcut(df["age"], 4, labels=["Q1 (jawan)", "Q2", "Q3", "Q4 (bade)"])
g_wt = pd.qcut(df["wtkg"], 4, labels=["Q1 (halka)", "Q2", "Q3", "Q4 (bhari)"])
g_pre = pd.cut(df["preanti"], [-1, 0, 365, 730, np.inf], labels=["0 din (pehle koi nahi)", "1-365 din", "366-730 din", "730+ din"])
g_kar = df["karnof"].map({70: "70-80", 80: "70-80", 90: "90", 100: "100"})
panels = [("cd40", g_cd40), ("cd80", g_cd80), ("preanti (pehle ki therapy ke din)", g_pre),
          ("karnof (roz ke kaam ki salahiyat)", g_kar), ("age", g_age), ("wtkg", g_wt)]
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
for ax, (name, g) in zip(axes.ravel(), panels):
    g = g.astype(str) if name.startswith("karnof") else g
    levels = ["70-80", "90", "100"] if name.startswith("karnof") else list(g.cat.categories)
    pal = PAL3 if len(levels) == 3 else PAL4
    gs = g.astype(str)
    for lv, col in zip(levels, pal):
        m = gs == lv
        if m.sum() == 0:
            continue
        k = KaplanMeierFitter().fit(df.loc[m, "days"], df.loc[m, "cens"], label=f"{lv} (n={m.sum()})")
        k.plot_survival_function(ax=ax, color=col, ci_show=False, lw=2)
    ax.set_ylim(0.4, 1.02); ax.set_xlabel("Din"); ax.set_ylabel("Event se bache rehne ka imkaan")
    ax.set_title(f"{name}  (p = {logrank_p(gs):.2g})"); ax.legend(loc="lower left", fontsize=8.5)
fig.suptitle("Step 7b: Feature ke groups ke hisaab se Kaplan-Meier (p = log-rank test)",
             fontsize=14, fontweight="bold")
save(fig, "eda_7b_km_by_feature.png")
for name, g in panels:
    gs = g.astype(str)
    r = df.groupby(gs)["cens"].agg(["size", "mean"])
    print(f"{name}: event rate % ->", {k: f"{v*100:.1f}% (n={n})" for k, (n, v) in r.iterrows()})
print()


# ==================================================================
# STEP 8: BINARY FEATURES KA EVENT RATE
# Har 0/1 feature ke liye: jin mareezon mein feature = 0 tha unmein kitne % par event hua,
# aur jin mein feature = 1 tha unmein kitne % par. Dono ke beech ka fasla jitna bara, utna
# feature outcome se juda hua.
# ==================================================================
head("STEP 8: Binary features aur event rate")
BIN = ["hemo", "homo", "drugs", "oprior", "z30", "race", "gender", "symptom"]
er = []
for c in BIN:
    r0 = df.loc[df[c] == 0, "cens"].mean() * 100; r1 = df.loc[df[c] == 1, "cens"].mean() * 100
    er.append({"feature": c, "n0": int((df[c] == 0).sum()), "n1": int((df[c] == 1).sum()),
               "rate_0": r0, "rate_1": r1, "diff": r1 - r0})
er = pd.DataFrame(er).sort_values("diff").reset_index(drop=True)
print(er.round(1).to_string(index=False), "\n")
er.round(2).to_csv(TAB_DIR / "eda_event_rate_binary.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
for i, r in er.iterrows():
    ax.plot([r["rate_0"], r["rate_1"]], [i, i], color=GREY_C, lw=2, zorder=1)
    ax.scatter(r["rate_0"], i, color=CENS_C, s=90, zorder=3)
    ax.scatter(r["rate_1"], i, color=EVENT_C, s=90, zorder=3)
    ax.text(max(r["rate_0"], r["rate_1"]) + 0.9, i, f"{r['diff']:+.1f} point", va="center", fontsize=9)
ax.axvline(OVERALL, color="#333333", ls="--", lw=1.2)
ax.set_yticks(range(len(er)))
ax.set_yticklabels([f"{r.feature}  (0: {r.n0}, 1: {r.n1})" for r in er.itertuples()])
ax.set_xlim(er[["rate_0", "rate_1"]].min().min() - 3, er[["rate_0", "rate_1"]].max().max() + 8)
ax.set_xlabel("Event hone wale mareez (%)")
ax.set_title("Step 8: Feature = 0 vs feature = 1 par event ka hissa")
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CENS_C, markersize=10, label="Feature = 0"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor=EVENT_C, markersize=10, label="Feature = 1"),
                   Line2D([0], [0], color="#333333", ls="--", label=f"Sab mareezon ka average ({OVERALL:.1f}%)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
save(fig, "eda_8_event_rate_binary.png")


# ==================================================================
# STEP 9a: TREATMENT KE HISAAB SE SUBGROUP PATTERNS (KM)
# Mareezon ko subgroups mein baant kar dekhna ke kya treatment ka asar har subgroup mein ek jaisa hai.
# ==================================================================
head("STEP 9a: Treatment ka asar subgroups mein (KM)")
med_cd40, med_age = df["cd40"].median(), df["age"].median()
sub = {
    "symptom = 0 (koi symptom nahi)": df["symptom"] == 0,
    "symptom = 1 (symptom hain)": df["symptom"] == 1,
    f"cd40 <= {med_cd40:.0f} (kam)": df["cd40"] <= med_cd40,
    f"cd40 > {med_cd40:.0f} (zyada)": df["cd40"] > med_cd40,
    f"age <= {med_age:.0f} (jawan)": df["age"] <= med_age,
    f"age > {med_age:.0f} (bade)": df["age"] > med_age,
}
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
order9 = [list(sub)[0], list(sub)[2], list(sub)[4], list(sub)[1], list(sub)[3], list(sub)[5]]
for ax, name in zip(axes.ravel(), order9):
    m = sub[name]
    d = df[m]
    for t, lab in [(0, "ZDV akela"), (1, "Koi doosra")]:
        mm = d["treat"] == t
        k = KaplanMeierFitter().fit(d.loc[mm, "days"], d.loc[mm, "cens"], label=f"{lab} (n={mm.sum()})")
        k.plot_survival_function(ax=ax, color=TREAT_C[t], ci_alpha=0.12, lw=2)
    p = multivariate_logrank_test(d["days"], d["treat"], d["cens"]).p_value
    ax.set_ylim(0.4, 1.02); ax.set_xlabel("Din"); ax.set_ylabel("Event se bache rehne ka imkaan")
    ax.set_title(f"{name}  (p = {p:.2g})"); ax.legend(loc="lower left")
fig.suptitle("Step 9a: ZDV akela vs baqi treatments, alag alag subgroups mein",
             fontsize=14, fontweight="bold")
save(fig, "eda_9a_km_treat_subgroups.png")


# ==================================================================
# STEP 9b: FOREST PLOT (har subgroup mein treatment ka asar ek nazar mein)
# Hazard ratio (HR) = "doosre treatments mein event ka khatra, ZDV akele ke muqabley mein kitna guna".
# HR = 1 matlab koi fark nahi, HR < 1 matlab doosre treatments behtar (khatra kam).
# Lakeer = 95% bharosa ka daira. Agar daira 1 ko na chhuye to fark pakka lagta hai.
# ==================================================================
head("STEP 9b: Forest plot (hazard ratio)")
SUB = [("Sab mareez", pd.Series(True, index=df.index))]
for c, nm in [("symptom", "symptom"), ("gender", "gender"), ("race", "race"),
              ("drugs", "drugs"), ("homo", "homo")]:
    SUB += [(f"{nm} = 0", df[c] == 0), (f"{nm} = 1", df[c] == 1)]
SUB += [(f"cd40 <= {med_cd40:.0f}", df["cd40"] <= med_cd40), (f"cd40 > {med_cd40:.0f}", df["cd40"] > med_cd40),
        (f"age <= {med_age:.0f}", df["age"] <= med_age), (f"age > {med_age:.0f}", df["age"] > med_age),
        ("karnof = 100", df["karnof"] == 100), ("karnof < 100", df["karnof"] < 100),
        ("preanti = 0 (naive)", df["preanti"] == 0), ("preanti > 0", df["preanti"] > 0)]
res = []
for nm, m in SUB:
    d = df.loc[m, ["treat", "days", "cens"]]
    c = CoxPHFitter().fit(d, "days", "cens")
    s = c.summary.loc["treat"]
    res.append({"subgroup": nm, "n": int(m.sum()), "events": int(d["cens"].sum()),
                "HR": float(np.exp(s["coef"])), "lo": float(np.exp(s["coef lower 95%"])),
                "hi": float(np.exp(s["coef upper 95%"]))})
res = pd.DataFrame(res)
res.round(3).to_csv(TAB_DIR / "eda_forest_treat_subgroups.csv", index=False)
print(res.round(2).to_string(index=False), "\n")

fig = plt.figure(figsize=(12, 9.5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.1])
ax = fig.add_subplot(gs[0, 0]); axt = fig.add_subplot(gs[0, 1], sharey=ax)
n = len(res); ys = np.arange(n)[::-1]
for y, r in zip(ys, res.itertuples()):
    col = GOOD_C if r.hi < 1 else (BAD_C if r.lo > 1 else GREY_C)
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=2.2, zorder=2)
    ax.scatter(r.HR, y, color=col, s=70 if r.subgroup != "Sab mareez" else 130,
               marker="o" if r.subgroup != "Sab mareez" else "D", zorder=3)
    axt.text(0, y, f"{r.HR:.2f}  ({r.lo:.2f} - {r.hi:.2f})", va="center", fontsize=9.5)
ax.axvline(1, color="#333333", ls="--", lw=1.2)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
xt = [0.3, 0.5, 0.7, 1, 1.5]; ax.set_xticks(xt); ax.set_xticklabels([str(x) for x in xt])
ax.set_xlim(min(0.25, res["lo"].min() * 0.9), max(1.8, res["hi"].max() * 1.1))
ax.set_yticks(ys); ax.set_yticklabels([f"{r.subgroup}  (n={r.n})" for r in res.itertuples()])
ax.set_xlabel("Hazard ratio (log scale)   <-- doosre treatments behtar   |   ZDV akela behtar -->")
ax.set_title("Step 9b: Treatment ka asar har subgroup mein")
ax.legend(handles=[Patch(color=GOOD_C, label="Daira 1 se door: doosre treatments behtar"),
                   Patch(color=GREY_C, label="Daira 1 ko chhuta hai: fark pakka nahi"),
                   Patch(color=BAD_C, label="Daira 1 se door: ZDV akela behtar")],
          loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1)
axt.set_xlim(0, 1); axt.axis("off"); axt.set_title("HR (95% daira)", fontsize=10)
save(fig, "eda_9b_forest_treatment.png")


# ==================================================================
# STEP 10: PATTERNS KI TALASH (nonlinear, zero ke dher, do features ka rishta)
# Feature ko barabar groups (deciles) mein baant kar har group ka event rate dekhte hain.
# Agar line seedhi na ho (mur jaye) to rishta nonlinear hai.
# ==================================================================
head("STEP 10: Patterns")
fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
colors = ["#2F5D8A", "#3A9D8F", "#8E5BA8", "#E4893B"]
for ax, c, col in zip(axes.ravel()[:4], ["cd40", "cd80", "age", "wtkg"], colors):
    b = pd.qcut(df[c], 10, duplicates="drop")
    gx = df.groupby(b, observed=True).agg(x=(c, "median"), rate=("cens", "mean"), n=("cens", "size"))
    ax.plot(gx["x"], gx["rate"] * 100, "-o", color=col, lw=2, ms=7, label="Har group ka event rate")
    ax.axhline(OVERALL, color="#333333", ls="--", lw=1.1, label=f"Average ({OVERALL:.1f}%)")
    ax.set_xlabel(f"{c} (har point = 10% mareezon ka group)"); ax.set_ylabel("Event hone wale mareez (%)")
    ax.set_title(f"{c}: kya rishta seedha hai?"); ax.legend(loc="best")
    print(f"{c}: pehle group {gx['rate'].iloc[0]*100:.1f}% -> aakhri group {gx['rate'].iloc[-1]*100:.1f}%")

ax = axes.ravel()[4]
pg = df.groupby(g_pre, observed=True).agg(rate=("cens", "mean"), n=("cens", "size"))
ax.bar(range(len(pg)), pg["rate"] * 100, color="#C8553D", width=0.6, label="Event rate")
for i, (rt, nn) in enumerate(zip(pg["rate"], pg["n"])):
    ax.text(i, rt * 100 + 0.6, f"{rt*100:.1f}%\n(n={nn})", ha="center", fontsize=9)
ax.axhline(OVERALL, color="#333333", ls="--", lw=1.1, label=f"Average ({OVERALL:.1f}%)")
ax.set_xticks(range(len(pg))); ax.set_xticklabels(["0 din", "1-365", "366-730", "730+"])
ax.set_ylim(0, pg["rate"].max() * 100 * 1.3); ax.set_xlabel("preanti (pehle ki therapy ke din)")
ax.set_ylabel("Event hone wale mareez (%)"); ax.set_title("preanti: zero ka dher"); ax.legend(loc="upper left")

ax = axes.ravel()[5]
hb = ax.hexbin(df["cd40"], df["cd80"], gridsize=28, cmap="viridis", mincnt=1)
cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02); cb.set_label("Mareezon ki tadaad")
rho = spearmanr(df["cd40"], df["cd80"])[0]
ax.set_xlabel("cd40"); ax.set_ylabel("cd80"); ax.set_title(f"cd40 aur cd80 (Spearman r = {rho:.2f})")
fig.suptitle("Step 10: Patterns ki talash", fontsize=14, fontweight="bold")
save(fig, "eda_10_patterns.png")

# ##################################################################
#            PART 3: STEP 11 (EDA KE NATEEJE + FEATURE ENGINEERING KI LIST)
# ##################################################################

# ==================================================================
# STEP 11: EDA KE SAARE NATEEJE EK JAGAH + FEATURE ENGINEERING MEIN TAY KARNE WALI CHEEZEIN
# Hissa A: Step 1 se 10 mein jo kuch mila (data se nikle hue numbers).
# Hissa B: Wo cheezein jo feature engineering (notebook 03) mein tay hongi.
# Dhyan: koi feature outcome dekh kar nahi chuna ya hataya jayega.
# ==================================================================
head("STEP 11: EDA ke nateeje aur feature engineering ki list")

n_all, n_ev = len(df), int(df["cens"].sum())
str2_dup = bool((df.groupby("strat")["str2"].nunique() == 1).all())
treat_dup = bool((df.groupby("arms")["treat"].nunique() == 1).all())
r_sp, r_zp = corr.loc["strat", "preanti"], corr.loc["z30", "preanti"]
pre_zero = (df["preanti"] == 0).mean() * 100
n_oprior = int(df["oprior"].sum())
n_k70 = int((df["karnof"] == 70).sum()); n_k7080 = int((df["karnof"] <= 80).sum())
out_pct = {}
for c in CONT:
    q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
    out_pct[c] = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).mean() * 100
arm_ev = df.groupby("arms")["cens"].mean() * 100
arm_cfu = df[df.cens == 0].groupby("arms")["days"].median()
p_chi = chi2_contingency(pd.crosstab(df["arms"], df["cens"]))[1]
km_vals = km_tab.iloc[:, 1].values * 100
hr_all = res.iloc[0]
hr_sub = res.iloc[1:]
wide_ci = res[(res["lo"] <= 1) & (res["hi"] >= 1)]
b40 = pd.qcut(df["cd40"], 10, duplicates="drop")
r40 = df.groupby(b40, observed=True)["cens"].mean() * 100
b80 = pd.qcut(df["cd80"], 10, duplicates="drop")
r80 = df.groupby(b80, observed=True)["cens"].mean() * 100
n_feat_left = len([c for c in df.columns if c not in ("days", "cens", "arms", "treat", "zprior", "str2")])


def grp_line(name, g):
    gs = g.astype(str)
    order = ["70-80", "90", "100"] if name.startswith("karnof") else [str(x) for x in g.cat.categories]
    rt = df.groupby(gs)["cens"].mean() * 100
    return (f"{name.split(' ')[0]}: log-rank p = {logrank_p(gs):.2g}; event rate pehle group mein "
            f"{rt[order[0]]:.1f}%, aakhri group mein {rt[order[-1]]:.1f}%")


top4 = top.sort_values("abs", ascending=False).head(4)
vtop = vp.sort_values("v", ascending=False).head(3)
er_sorted = er.sort_values("diff", ascending=False)

eda_results = [
    ("STEP 1: Outcome", [
        f"{n_all} mareez, {n_ev} par event ({n_ev/n_all*100:.1f}%), {n_all-n_ev} censored. "
        f"Follow-up min {df['days'].min()}, median {df['days'].median():.0f}, max {df['days'].max()} din.",
    ]),
    ("STEP 2: Features ka distribution", [
        "Skewness: " + ", ".join(f"{k} {v:.2f}" for k, v in sk.sort_values(ascending=False).items())
        + ". Sirf cd80 aur preanti ki skew 1 se zyada.",
        f"preanti: {pre_zero:.0f}% mareez 0 par (zero ka dher).",
        "Binary features mein 1 ka hissa: " + ", ".join(f"{c} {df[c].mean()*100:.1f}%" for c in BINARY) + ".",
        f"karnof: sirf 4 values (70: {n_k70}, 80: {int((df['karnof']==80).sum())}, 90: {int((df['karnof']==90).sum())}, 100: {int((df['karnof']==100).sum())}). "
        f"strat: " + ", ".join(f"{k}: {v}" for k, v in df['strat'].value_counts().sort_index().items()) + ".",
        "Arms: " + ", ".join(f"{ARM_NAMES[a]} {int((df['arms']==a).sum())}" for a in range(4)) + ".",
    ]),
    ("STEP 3: Bekaar / duplicate features", [
        f"zprior: sab {n_all} mareezon ki ek hi value (constant). oprior: {100-n_oprior/n_all*100:.1f}% mareezon par 0 (lagbhag constant).",
        f"strat se str2 poori tarah tay hota hai: {'haan' if str2_dup else 'nahi'}. arms se treat poori tarah tay hota hai: {'haan' if treat_dup else 'nahi'}.",
    ]),
    ("STEP 4: Baseline balance", [
        f"Sab se bara SMD {smd.max().max():.3f} ({smd.max(axis=1).idxmax()}). Koi feature 0.1 se upar nahi: arms shuru mein barabar.",
    ]),
    ("STEP 5: Censoring by arm", [
        "Event ka hissa: " + ", ".join(f"{ARM_NAMES[a]} {arm_ev[a]:.1f}%" for a in range(4)) + f" (chi-square p = {p_chi:.2g}).",
        "Censored mareezon ka median follow-up: " + ", ".join(f"{ARM_NAMES[a]} {arm_cfu[a]:.0f}" for a in range(4))
        + " din (sab arms mein qareeb, dropout ka bias nazar nahi aata).",
    ]),
    ("STEP 6: Features ka aapas ka rishta", [
        "Sab se mazboot Spearman jore: " + ", ".join(f"{r.a}-{r.b} {r.r:.2f}" for r in top4.itertuples()) + ".",
        "Cramer's V ke sab se mazboot jore: " + ", ".join(f"{r.a}-{r.b} {r.v:.2f}" for r in vtop.itertuples()) + ".",
        f"arms ka kisi bhi feature se sab se bara V {arms_v.max():.3f} (randomization theek).",
    ]),
    ("STEP 7: Kaplan-Meier", [
        "900 din par event se bache hue: " + ", ".join(f"{ARM_NAMES[a]} {km_vals[a]:.1f}%" for a in range(4))
        + f". Log-rank p: 4 arms {p_arms:.2g}, ZDV vs baqi {p_tr:.2g}.",
    ] + [grp_line(n_, g_) for n_, g_ in panels]),
    ("STEP 8: Binary features ka event rate (feature = 1 minus feature = 0, point mein)", [
        ", ".join(f"{r.feature} {r.diff:+.1f}" for r in er_sorted.itertuples()) + f". (oprior mein sirf {n_oprior} mareez.)",
    ]),
    ("STEP 9: Treatment ka asar subgroups mein", [
        f"Sab mareez: HR {hr_all.HR:.2f} ({hr_all.lo:.2f} - {hr_all.hi:.2f}). Subgroups mein HR {hr_sub.HR.min():.2f} se {hr_sub.HR.max():.2f}, "
        "sab 1 se neeche (doosre treatments ki taraf).",
        "Jahan 95% daira 1 ko chhuta hai: " + ", ".join(f"{r.subgroup} (n={r.n}, HR {r.HR:.2f})" for r in wide_ci.itertuples())
        + ". Yeh chhote groups hain jahan events kam hain.",
        "karnof = 100 mein HR " + f"{res.loc[res.subgroup=='karnof = 100','HR'].iloc[0]:.2f}" + " aur karnof < 100 mein "
        + f"{res.loc[res.subgroup=='karnof < 100','HR'].iloc[0]:.2f}" + ": halka ishara, daire aapas mein milte hain.",
    ]),
    ("STEP 10: Patterns", [
        f"cd40 (deciles): event rate {r40.iloc[0]:.1f}% se girta hai {r40.iloc[-1]:.1f}% tak. cd80: {r80.iloc[0]:.1f}% se {r80.iloc[-1]:.1f}% tak.",
        f"preanti: event rate {pg['rate'].iloc[0]*100:.1f}% (0 din) se {pg['rate'].iloc[-1]*100:.1f}% (730+ din) tak barhta hai.",
        "age aur wtkg ke deciles mein sirf halka utar chadhao (shor jaisa).",
        f"cd40 aur cd80 ka Spearman rishta {rho:.2f} (halka).",
    ]),
    ("PEHLE KE CHECKS (cleaning, linearity, outliers)", [
        "Missing values: koi nahi (cd496 hatane ke baad). Hatayi gayi columns: cd496, cd420, cd820, r, offtrt, pidnum.",
        "Linearity (martingale residual graph): cd40 mein mor nazar aaya; age, wtkg, preanti, cd80 taqriban seedhe.",
        "Outliers (IQR rule): " + ", ".join(f"{k} {v:.1f}%" for k, v in out_pct.items()) + ". Asli clinical values lagti hain.",
    ]),
]

fe_todo = [
    ("zprior, str2",
     "zprior constant hai; str2 strat se poori tarah tay hota hai.",
     "Dono ko hatana."),
    ("arms, treat",
     f"arms se treat khud nikalta hai; treatment features (X) ka hissa nahi.",
     "Treatment column alag rakhna. arms rakhein (4 arms), treat usi se nikal sakta hai."),
    ("strat, preanti, z30",
     f"strat-preanti r = {r_sp:.2f}, z30-preanti r = {r_zp:.2f}: teeno ek hi baat dohra rahe hain.",
     "Inmein se ek ya do rakhna."),
    ("oprior",
     f"Sirf {n_oprior} mareez ({n_oprior/n_all*100:.1f}%) par 1.",
     "Rakhna ya hatana tay karna."),
    ("preanti",
     f"{pre_zero:.0f}% mareez 0 par, skew {sk['preanti']:.2f}.",
     "log1p dekhna (ya 0 / >0 flag)."),
    ("cd80",
     f"skew {sk['cd80']:.2f}, sab se zyada jhuka hua.",
     "log1p transform dekhna."),
    ("cd40",
     "Linearity graph mein mor; event rate shuru mein tezi se girta hai phir chapta hota hai; cd40 = 0 bhi hai.",
     "log1p ya spline dekhna (log nahi, zero ki wajah se)."),
    ("karnof",
     f"Sirf 4 values; 70 par sirf {n_k70} mareez, 70 aur 80 milakar {n_k7080}.",
     "Categories banana: 70-80, 90, 100."),
    ("age, wtkg, binary features",
     "Skew 1 se kam, koi masla nahi.",
     "Jaisa hai rakhna."),
    ("Outliers",
     f"Sab features mein {max(out_pct.values()):.1f}% se kam, asli values.",
     "Rakhna, hatana nahi."),
    ("Missing values",
     "Koi nahi.",
     "Imputation ki zaroorat nahi."),
    ("Qaida (hamesha)",
     "Poore data par seekhne se leakage hoti hai.",
     "Transform aur scaling sirf cross-validation ke training fold ke andar seekhna."),
]
todo = pd.DataFrame(fe_todo, columns=["feature", "eda_ne_kya_dikhaya", "feature_engineering_mein_kya_karna"])

print("\n" + "#" * 62 + "\nHISSA A: EDA KE NATEEJE\n" + "#" * 62)
for title, lines in eda_results:
    print(f"\n{title}")
    for ln in lines:
        print("   - " + ln)

print("\n" + "#" * 62 + "\nHISSA B: FEATURE ENGINEERING MEIN TAY KARNE WALI CHEEZEIN\n" + "#" * 62)
for i, r in enumerate(todo.itertuples(), 1):
    print(f"\n {i}. {r.feature}\n      EDA ne dikhaya: {r.eda_ne_kya_dikhaya}\n      Feature engineering mein: {r.feature_engineering_mein_kya_karna}")

todo.to_csv(TAB_DIR / "eda_feature_decisions.csv", index=False, encoding="utf-8-sig")
summ_path = ROOT / "results" / "eda_summary.txt"
with open(summ_path, "w", encoding="utf-8") as fh:
    fh.write("EDA KE NATEEJE (ACTG175)\n" + "=" * 60 + "\n")
    for title, lines in eda_results:
        fh.write(f"\n{title}\n")
        for ln in lines:
            fh.write(f"  - {ln}\n")
    fh.write("\n\nFEATURE ENGINEERING MEIN TAY KARNE WALI CHEEZEIN\n" + "=" * 60 + "\n")
    for i, r in enumerate(todo.itertuples(), 1):
        fh.write(f"\n{i}. {r.feature}\n   EDA ne dikhaya: {r.eda_ne_kya_dikhaya}\n"
                 f"   Feature engineering mein: {r.feature_engineering_mein_kya_karna}\n")
print(f"\n   [save hua] {summ_path}")
print(f"   [save hua] {TAB_DIR / 'eda_feature_decisions.csv'}")

# ==================================================================
# STEP 12: EDA KA EK PAGE KA GRAPH (summary dashboard)
# Pure EDA ke sab se aham nateeje ek hi graph mein (9 chhote panels).
# ==================================================================
head("STEP 12: EDA ka ek-page summary graph")

fig = plt.figure(figsize=(20, 14.5), constrained_layout=True)
gs = fig.add_gridspec(3, 3)

# 1. Outcome (donut)
ax = fig.add_subplot(gs[0, 0])
ax.pie([n_ev, n_all - n_ev], colors=[EVENT_C, CENS_C], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.38, edgecolor="white"))
ax.text(0, 0, f"{n_all}\nmareez", ha="center", va="center", fontsize=15, fontweight="bold")
ax.legend(handles=[Patch(color=EVENT_C, label=f"Event hua: {n_ev} ({n_ev/n_all*100:.1f}%)"),
                   Patch(color=CENS_C, label=f"Censored: {n_all-n_ev} ({(n_all-n_ev)/n_all*100:.1f}%)")],
          loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1)
ax.set_title(f"1. Outcome: {n_ev/n_all*100:.0f}% mareezon par event")

# 2. Randomization (SMD)
ax = fig.add_subplot(gs[0, 1])
s8 = smd.max(axis=1).sort_values().tail(8)
ax.barh(s8.index, s8.values, color="#E4893B", height=0.6, label="Sab se bara SMD (4 arms mein)")
for i, v in enumerate(s8.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
ax.axvline(0.1, color="#333333", ls="--", lw=1.2, label="0.1 ki hadd")
ax.set_xlim(0, 0.115); ax.set_xlabel("SMD (chhota = arms barabar)")
ax.legend(loc="lower right")
ax.set_title(f"2. Randomization theek: sab SMD {smd.max().max():.3f} (< 0.1)")

# 3. Kaplan-Meier (arms)
ax = fig.add_subplot(gs[0, 2])
for a in range(4):
    m = df["arms"] == a
    KaplanMeierFitter().fit(df.loc[m, "days"], df.loc[m, "cens"], label=ARM_NAMES[a]).plot_survival_function(
        ax=ax, ci_show=False, color=ARM_COLORS[a], lw=2)
ax.set_ylim(0.5, 1.02); ax.set_xlabel("Din"); ax.set_ylabel("Event se bache rehne ka imkaan")
ax.legend(loc="lower left")
ax.set_title(f"3. ZDV akela neeche: 900 din par {km_vals[0]:.0f}% vs baqi {km_vals[1:].min():.0f}-{km_vals[1:].max():.0f}%")

# 4. Kaun sa feature outcome se juda hai
ax = fig.add_subplot(gs[1, 0])
pv = {n_.split(" ")[0]: logrank_p(g_.astype(str)) for n_, g_ in panels}
strength = pd.Series({k: -np.log10(v) for k, v in pv.items()}).sort_values()
thr = -np.log10(0.05)
ax.barh(strength.index, strength.values, color=[GOOD_C if v > thr else GREY_C for v in strength.values], height=0.6)
for i, (k, v) in enumerate(strength.items()):
    ax.text(max(v, thr) + strength.max() * 0.02, i, f"p = {pv[k]:.2g}", va="center", fontsize=9)
ax.axvline(thr, color="#333333", ls="--", lw=1.2)
ax.set_xlim(0, strength.max() * 1.3); ax.set_xlabel("Rishte ki taqat (bara = mazboot)")
ax.legend(handles=[Patch(color=GOOD_C, label="p < 0.05 (fark pakka)"), Patch(color=GREY_C, label="p >= 0.05"),
                   Line2D([0], [0], color="#333333", ls="--", label="0.05 ki hadd")], loc="lower right")
ax.set_title(f"4. Outcome se sab se juda feature: {strength.index[-1]}")

# 5. Overlap wale features
ax = fig.add_subplot(gs[1, 1])
nm = ["strat", "preanti", "z30"]
M = corr.loc[nm, nm].values
im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(nm); ax.set_yticks(range(3)); ax.set_yticklabels(nm)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if abs(M[i, j]) > 0.6 else "#222222")
ax.spines[:].set_visible(False)
cbr = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03); cbr.set_label("Spearman correlation")
ax.set_title("5. strat, preanti, z30: lagbhag ek hi cheez")

# 6. Forest (chhota)
ax = fig.add_subplot(gs[1, 2])
sel = ["Sab mareez", "symptom = 0", "symptom = 1", "karnof = 100", "karnof < 100",
       f"cd40 <= {med_cd40:.0f}", f"cd40 > {med_cd40:.0f}", f"age <= {med_age:.0f}", f"age > {med_age:.0f}"]
rs = res.set_index("subgroup").loc[sel].reset_index()
ys = np.arange(len(rs))[::-1]
for y, r in zip(ys, rs.itertuples()):
    col = GOOD_C if r.hi < 1 else GREY_C
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=2.2)
    ax.scatter(r.HR, y, color=col, s=110 if r.subgroup == "Sab mareez" else 60,
               marker="D" if r.subgroup == "Sab mareez" else "o", zorder=3)
ax.axvline(1, color="#333333", ls="--", lw=1.2)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([0.3, 0.5, 0.7, 1, 1.5]); ax.set_xticklabels(["0.3", "0.5", "0.7", "1", "1.5"])
ax.set_xlim(0.25, 1.6)
ax.set_yticks(ys); ax.set_yticklabels([f"{r.subgroup} (n={r.n})" for r in rs.itertuples()])
ax.set_xlabel("Hazard ratio (1 se neeche = doosre treatments behtar)")
ax.legend(handles=[Patch(color=GOOD_C, label="Daira 1 se door"), Patch(color=GREY_C, label="Daira 1 ko chhuta hai")],
          loc="lower right")
ax.set_title(f"6. Har subgroup mein HR < 1 (sab mareez: {hr_all.HR:.2f})")

# 7. Skewness
ax = fig.add_subplot(gs[2, 0])
skv = sk.sort_values()
ax.barh(skv.index, skv.values, color=[BAD_C if v > 1 else GOOD_C for v in skv.values], height=0.6)
for i, v in enumerate(skv.values):
    ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=9)
ax.axvline(1, color="#333333", ls="--", lw=1.2)
ax.set_xlim(0, skv.max() * 1.25); ax.set_xlabel("Skewness")
ax.legend(handles=[Patch(color=BAD_C, label="skew > 1 (zyada jhuka hua)"), Patch(color=GOOD_C, label="skew <= 1"),
                   Line2D([0], [0], color="#333333", ls="--", label="1 ki hadd")], loc="lower right")
ax.set_title("7. Sirf " + " aur ".join(skv[skv > 1].index[::-1]) + " zyada jhuke hue")

# 8. cd40 ka pattern
ax = fig.add_subplot(gs[2, 1])
gx = df.groupby(b40, observed=True).agg(x=("cd40", "median"), rate=("cens", "mean"))
ax.plot(gx["x"], gx["rate"] * 100, "-o", color="#8E5BA8", lw=2.2, ms=7, label="Har group ka event rate")
ax.axhline(OVERALL, color="#333333", ls="--", lw=1.1, label=f"Average ({OVERALL:.1f}%)")
ax.set_xlabel("cd40 (har point = 10% mareezon ka group)"); ax.set_ylabel("Event hone wale mareez (%)")
ax.legend(loc="upper right")
ax.set_title(f"8. cd40 jitna kam, event utna zyada ({r40.iloc[0]:.0f}% se {r40.iloc[-1]:.0f}%)")

# 9. Feature engineering ki list
ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
ax.text(0, 1.0, "9. Feature engineering mein kya hoga", fontsize=12, fontweight="bold", va="top",
        transform=ax.transAxes)
items = ["zprior, str2 hatana",
         "arms (treatment) alag, features (X) mein nahi",
         "strat / preanti / z30: ek ya do rakhna",
         "oprior: rakhna ya hatana tay karna",
         "preanti, cd80: log1p dekhna",
         "cd40: log1p ya spline dekhna",
         "karnof: categories 70-80 / 90 / 100",
         "outliers rakhna, hatana nahi; missing koi nahi",
         "transform sirf training fold ke andar seekhna"]
for i, t in enumerate(items):
    ax.text(0.02, 0.89 - i * 0.092, f"-  {t}", fontsize=11, va="top", transform=ax.transAxes)

fig.suptitle(f"EDA ka ek page ka khulasa  -  ACTG175, {n_all} mareez", fontsize=18, fontweight="bold")
save(fig, "eda_11_summary.png")

print("\nEDA mukammal (Step 1 se 12). Graphs: results/figures, tables: results/tables")
if SHOW:
    plt.show()