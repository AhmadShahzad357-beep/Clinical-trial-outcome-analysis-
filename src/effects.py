"""
EFFECTS - ACTG175: treatment ka overall asar (Kaplan-Meier, Cox, RMST)
Chalane ka tareeqa (project folder se):
    python src/effects.py
Sirf PNG save karni hon (window na khule):  ... effects.py --no-show

Sawal: ZDV akele ke muqabley mein baqi teen arms ne outcome (event) par kitna fark dala?
Data: poore 2139 mareez (randomized trial ka asar naapne mein koi model tune nahi hota, is liye
test data ko alag karne ki zaroorat nahi). Train-only ka muqabla bhi ek line mein dikhaya gaya hai.
"""
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

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
SHOW = "--no-show" not in sys.argv

TAU = 900                  # RMST aur "event se bache hue" ka waqt (din)
TAU2 = 600                 # sensitivity (doosra waqt)
N_BOOT = 500               # bootstrap ki tadaad (RMST aur risk fark ke daire ke liye)
SEED = 42
ARM_NAMES = {0: "ZDV", 1: "ZDV+ddI", 2: "ZDV+zal", 3: "ddI"}
ARM_COLORS = {0: "#2F5D8A", 1: "#E4893B", 2: "#3A9D8F", 3: "#8E5BA8"}
GOOD_C, GREY_C, EVENT_C = "#3A9D8F", "#9AA5B1", "#C8553D"
COVARIATES = ["age", "wtkg", "hemo", "homo", "drugs", "karnof_70_80", "karnof_90", "z30",
              "preanti", "race", "gender", "symptom", "cd40", "cd80"]     # Cox version jaise (oprior nahi)

plt.rcParams.update({
    "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


def holm(pvals):
    """Holm correction: ek se zyada comparisons par jhoote 'fark' se bachne ke liye."""
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
df = pd.read_csv(PROCESSED / "actg175_clean.csv")
df["karnof_70_80"] = (df["karnof"] <= 80).astype(int)
df["karnof_90"] = (df["karnof"] == 90).astype(int)
for a in (1, 2, 3):
    df[f"arm_{a}"] = (df["arms"] == a).astype(int)
ARM_COLS = ["arm_1", "arm_2", "arm_3"]
print(f"Project root: {ROOT}\nData: {len(df)} mareez, {int(df['cens'].sum())} events\n")


def km_fit(d, label=None):
    return KaplanMeierFitter().fit(d["days"], d["cens"], label=label)


def rmst(kmf, tau):
    """Restricted mean survival time: KM curve ke neeche ka raqba tau tak (din mein)."""
    sf = kmf.survival_function_
    t, s = sf.index.values, sf.iloc[:, 0].values
    m = t < tau
    return float(np.sum(s[m] * np.diff(np.append(t[m], tau))))


# ==================================================================
# STEP 1: KAPLAN-MEIER (har arm ka "event se bache rehne ka imkaan")
# KM curve = waqt ke saath kitne mareez abhi tak event se bache hue hain. Ooncha = behtar.
# ==================================================================
head("STEP 1: Kaplan-Meier (har arm)")
km, rows = {}, []
for a in range(4):
    d = df[df.arms == a]
    km[a] = km_fit(d, f"{ARM_NAMES[a]} (n={len(d)})")
    ci = km[a].confidence_interval_
    row = {"arm": ARM_NAMES[a], "n": len(d), "events": int(d["cens"].sum())}
    for t in (300, 600, TAU):
        row[f"bache_{t}din_%"] = round(float(km[a].predict(t)) * 100, 1)
    rows.append(row)
km_tab = pd.DataFrame(rows)
print("Event se bache hue mareez (%), alag alag waqt par:")
print(km_tab.to_string(index=False))

print("\nPairwise log-rank test (Holm correction ke baad):")
pairs = list(combinations(range(4), 2))
raw_p = [logrank_test(df.loc[df.arms == i, "days"], df.loc[df.arms == j, "days"],
                      df.loc[df.arms == i, "cens"], df.loc[df.arms == j, "cens"]).p_value for i, j in pairs]
adj_p = holm(raw_p)
pw = pd.DataFrame({"arm_A": [ARM_NAMES[i] for i, j in pairs], "arm_B": [ARM_NAMES[j] for i, j in pairs],
                   "p_adjusted": [fmt_p(p) for p in adj_p]})
print(pw.to_string(index=False))
pw.to_csv(TAB_DIR / "effects_pairwise_logrank.csv", index=False)
print("Matlab: ZDV ke saath har jore ka p chhota hai (asli fark). Baqi teeno arms ka aapas ka p bara hai:")
print("        yani teeno combination arms ek dusre se saaf alag nahi.\n")


# ==================================================================
# STEP 2: COX MODEL (hazard ratio)
# Hazard ratio (HR) = ZDV akele ke muqabley mein doosre arm mein event ka khatra kitna guna.
#   HR = 1 : koi fark nahi | HR = 0.5 : khatra aadha | HR > 1 : khatra zyada
# Unadjusted: sirf arms. Adjusted: arms + baseline features (age, cd40, karnof...). Randomized trial
# mein dono qareeb hone chahiye; adjusted se andaza thoda pakka (daira chhota) hota hai.
# ==================================================================
head("STEP 2: Cox model (hazard ratio, ZDV ke muqabley mein)")
cph_u = CoxPHFitter().fit(df[["days", "cens"] + ARM_COLS], "days", "cens")
cph_a = CoxPHFitter().fit(df[["days", "cens"] + ARM_COLS + COVARIATES], "days", "cens")


def hr_table(cph):
    s = cph.summary.loc[ARM_COLS]
    return pd.DataFrame({"HR": np.exp(s["coef"]), "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                         "p": s["p"]})


hu, ha = hr_table(cph_u), hr_table(cph_a)
hu["p_holm"], ha["p_holm"] = holm(hu["p"].values), holm(ha["p"].values)
for name, h in [("Unadjusted (sirf arms)", hu), ("Adjusted (arms + baseline features)", ha)]:
    print(name + ":")
    out = pd.DataFrame({"arm vs ZDV": [ARM_NAMES[i] for i in (1, 2, 3)],
                        "HR": h["HR"].round(2).values,
                        "95% daira": [f"{lo:.2f} - {hi:.2f}" for lo, hi in zip(h['lo'], h['hi'])],
                        "p (Holm)": [fmt_p(p) for p in h["p_holm"]]})
    print(out.to_string(index=False), "\n")

# sab "doosre treatments" ek saath (treat = 0/1)
c_t = CoxPHFitter().fit(df[["days", "cens", "treat"]], "days", "cens").summary.loc["treat"]
print(f"Sab doosre treatments ek saath vs ZDV: HR {np.exp(c_t['coef']):.2f} "
      f"({c_t['exp(coef) lower 95%']:.2f} - {c_t['exp(coef) upper 95%']:.2f}), p {fmt_p(c_t['p'])}")

# train-only se muqabla (test ko dekhe baghair)
sid = pd.read_csv(PROCESSED / "split_ids.csv")
tr_rows = sid.loc[sid["set"] == "train", "row"]
dtr = df.loc[tr_rows]
h_tr = hr_table(CoxPHFitter().fit(dtr[["days", "cens"] + ARM_COLS], "days", "cens"))
print("Sirf train data par (muqabla): " + ", ".join(f"{ARM_NAMES[i]} {h_tr.loc[f'arm_{i}', 'HR']:.2f}" for i in (1, 2, 3)))
print("Matlab: teeno arms mein khatra ZDV ka taqriban aadha (HR 0.5 ke qareeb). Adjusted aur unadjusted qareeb: theek.\n")

# Proportional hazards assumption
print("Proportional hazards check (Cox ka ek qaida: HR waqt ke saath ek jaisa rahe):")
ph = proportional_hazard_test(cph_a, df[["days", "cens"] + ARM_COLS + COVARIATES], time_transform="rank").summary
ph_arm = ph.loc[ARM_COLS, "p"]
print("   arms ke p-values:", {ARM_NAMES[i]: round(float(ph_arm[f"arm_{i}"]), 3) for i in (1, 2, 3)})
bad = [c for c in ph.index if ph.loc[c, "p"] < 0.05]
print(f"   p < 0.05 wale (qaida toota): {bad if bad else 'koi nahi'}")
print("   Matlab: p bara (0.05 se upar) = HR waqt ke saath ek jaisa. p chhota = HR waqt ke saath badalta hai.")
print("           Aisa ho to ek HR poore trial ka sirf average hai; RMST aur 'bache hue %' par zyada bharosa karein.\n")


# ==================================================================
# STEP 3: RMST AUR ABSOLUTE FARK (bootstrap se daira)
# RMST = pehle TAU din mein mareez ne average kitne din "event ke baghair" guzare.
#   RMST ka fark = doosre arm mein ZDV ke muqabley mein kitne din zyada event-free.
# Risk fark = TAU din par event se bache hue mareezon ke hisse ka fark (percentage point).
# NNT = kitne mareezon ko ZDV akele ki jagah yeh arm dein to ek ka event bache (1 / risk fark).
# Daira bootstrap se: har arm ke mareezon ko dobara-dobara chun kar 500 baar naap kar.
# ==================================================================
head(f"STEP 3: RMST aur absolute fark ({TAU} din)")
rng = np.random.default_rng(SEED)
groups = {a: df.loc[df.arms == a, ["days", "cens"]].reset_index(drop=True) for a in range(4)}
pt_rmst = {a: rmst(km[a], TAU) for a in range(4)}
pt_rmst2 = {a: rmst(km[a], TAU2) for a in range(4)}
pt_surv = {a: float(km[a].predict(TAU)) for a in range(4)}
boot = {"rmst": {a: [] for a in range(4)}, "rmst2": {a: [] for a in range(4)}, "surv": {a: [] for a in range(4)}}
for _ in range(N_BOOT):
    for a in range(4):
        g = groups[a].iloc[rng.integers(0, len(groups[a]), len(groups[a]))]
        k = KaplanMeierFitter().fit(g["days"], g["cens"])
        boot["rmst"][a].append(rmst(k, TAU)); boot["rmst2"][a].append(rmst(k, TAU2))
        boot["surv"][a].append(float(k.predict(TAU)))
boot = {m: {a: np.array(v) for a, v in d.items()} for m, d in boot.items()}


def diff_ci(metric, a, pt):
    dd = boot[metric][a] - boot[metric][0]
    return pt[a] - pt[0], np.percentile(dd, 2.5), np.percentile(dd, 97.5)


res_rows = []
for a in (1, 2, 3):
    r_d, r_lo, r_hi = diff_ci("rmst", a, pt_rmst)
    r2_d, r2_lo, r2_hi = diff_ci("rmst2", a, pt_rmst2)
    s_d, s_lo, s_hi = diff_ci("surv", a, pt_surv)
    res_rows.append({
        "arm": ARM_NAMES[a],
        "HR_unadj": round(hu.loc[f"arm_{a}", "HR"], 2),
        "HR_adj": round(ha.loc[f"arm_{a}", "HR"], 2),
        f"event_free_{TAU}d_%": round(pt_surv[a] * 100, 1),
        f"risk_fark_{TAU}d_pt": round(s_d * 100, 1),
        "risk_fark_95%": f"{s_lo*100:.1f} - {s_hi*100:.1f}",
        "NNT": round(1 / s_d, 1) if s_d > 0 else np.nan,
        f"RMST_fark_{TAU}d_din": round(r_d, 1),
        "RMST_95%": f"{r_lo:.1f} - {r_hi:.1f}",
        f"RMST_fark_{TAU2}d_din": round(r2_d, 1),
        "RMST_600_95%": f"{r2_lo:.1f} - {r2_hi:.1f}",
    })
res_tab = pd.DataFrame(res_rows)
print(f"ZDV ka RMST ({TAU} din tak): {pt_rmst[0]:.1f} din | event se bachne ka imkaan: {pt_surv[0]*100:.1f}%\n")
for _, r in res_tab.iterrows():
    print(f"{r['arm']} (ZDV ke muqabley mein):")
    print(f"   event se bache hue ({TAU} din): {r[f'event_free_{TAU}d_%']}%  |  fark {r[f'risk_fark_{TAU}d_pt']} point "
          f"(95% daira {r['risk_fark_95%']})  |  NNT {r['NNT']}")
    print(f"   RMST fark: +{r[f'RMST_fark_{TAU}d_din']} din ({TAU} din tak), 95% daira {r['RMST_95%']}")
    print(f"   RMST fark ({TAU2} din tak): +{r[f'RMST_fark_{TAU2}d_din']} din, 95% daira {r['RMST_600_95%']}")
res_tab.to_csv(TAB_DIR / "effects_summary.csv", index=False)
print("\nMatlab: RMST fark = in teeno arms ke mareez ZDV walon se average itne din zyada event ke baghair rahe.")
print("        NNT jitna chhota, utna behtar (is ke saath ek mareez ka event bachta hai).")
print(f"        {TAU2} din ka RMST fark sensitivity ke liye hai: tasveer wahi rehni chahiye.\n")


# ==================================================================
# STEP 4: EK PAGE KA GRAPH
# ==================================================================
head("STEP 4: Graph")
fig = plt.figure(figsize=(19, 11.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)

# 1. Kaplan-Meier
ax = fig.add_subplot(gs[0, 0])
for a in range(4):
    km[a].plot_survival_function(ax=ax, color=ARM_COLORS[a], ci_alpha=0.10, lw=2)
ax.axvline(TAU, color="#333333", ls=":", lw=1)
ax.set_ylim(0.5, 1.02); ax.set_xlabel("Din"); ax.set_ylabel("Event se bache rehne ka imkaan")
ax.legend(loc="lower left"); ax.set_title("1. Kaplan-Meier (daira = 95% bharosa)")

# 2. Forest: HR unadjusted vs adjusted
ax = fig.add_subplot(gs[0, 1])
ys = {1: 2.0, 2: 1.0, 3: 0.0}
for a in (1, 2, 3):
    for off, h, col in [(0.16, hu, "#2F5D8A"), (-0.16, ha, "#E4893B")]:
        r = h.loc[f"arm_{a}"]
        ax.plot([r.lo, r.hi], [ys[a] + off] * 2, color=col, lw=2.2)
        ax.scatter(r.HR, ys[a] + off, color=col, s=55, zorder=3)
        ax.text(r.hi * 1.04, ys[a] + off, f"{r.HR:.2f} ({r.lo:.2f}-{r.hi:.2f})", va="center", fontsize=8.5)
ax.axvline(1, color="#333333", ls="--", lw=1.2)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([0.3, 0.5, 0.7, 1]); ax.set_xticklabels(["0.3", "0.5", "0.7", "1"])
ax.set_xlim(0.28, 1.6); ax.set_ylim(-1.0, 2.5)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels([f"{ARM_NAMES[a]}\nvs ZDV" for a in (1, 2, 3)])
ax.set_xlabel("Hazard ratio (1 se neeche = ZDV akele se behtar)")
ax.legend(handles=[Patch(color="#2F5D8A", label="Unadjusted"), Patch(color="#E4893B", label="Adjusted (baseline features ke saath)")],
          loc="lower left")
ax.set_title("2. Hazard ratio (khatra ZDV ka kitna guna)")

# 3. RMST fark
ax = fig.add_subplot(gs[0, 2])
xs = np.arange(3)
for k_, a in enumerate((1, 2, 3)):
    d_, lo_, hi_ = diff_ci("rmst", a, pt_rmst)
    ax.bar(k_, d_, color=ARM_COLORS[a], width=0.55)
    ax.errorbar(k_, d_, yerr=[[d_ - lo_], [hi_ - d_]], color="#222222", capsize=5, lw=1.6)
    ax.text(k_, hi_ + 1.5, f"+{d_:.0f} din", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([ARM_NAMES[a] for a in (1, 2, 3)])
ax.set_ylim(0, max(diff_ci("rmst", a, pt_rmst)[2] for a in (1, 2, 3)) * 1.25)
ax.set_ylabel(f"Extra event-free din ({TAU} din tak, ZDV ke muqabley mein)")
ax.set_title(f"3. RMST fark (daira = 95% bharosa)")

# 4. Event-free at TAU
ax = fig.add_subplot(gs[1, 0])
for a in range(4):
    lo_, hi_ = np.percentile(boot["surv"][a], [2.5, 97.5])
    ax.bar(a, pt_surv[a] * 100, color=ARM_COLORS[a], width=0.55)
    ax.errorbar(a, pt_surv[a] * 100, yerr=[[pt_surv[a] * 100 - lo_ * 100], [hi_ * 100 - pt_surv[a] * 100]],
                color="#222222", capsize=5, lw=1.6)
    ax.text(a, hi_ * 100 + 1, f"{pt_surv[a]*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([ARM_NAMES[a] for a in range(4)])
ax.set_ylim(50, 100); ax.set_ylabel(f"{TAU} din par event se bache hue (%)")
ax.set_title(f"4. {TAU} din par kitne mareez event se bache")

# 5. log-log (PH check)
ax = fig.add_subplot(gs[1, 1])
for a in range(4):
    sf = km[a].survival_function_.iloc[:, 0]
    sf = sf[(sf.index > 30) & (sf < 0.999) & (sf > 0)]
    ax.plot(np.log(sf.index), np.log(-np.log(sf.values)), color=ARM_COLORS[a], lw=2, label=ARM_NAMES[a])
ax.set_xlabel("log(din)"); ax.set_ylabel("log(-log(bache rehne ka imkaan))")
ax.legend(loc="lower right"); ax.set_title("5. Proportional hazards ka ishara (lakeerein qareeb parallel hon)")

# 6. Khulasa
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
lines = [f"HR (adjusted), ZDV ke muqabley mein:"]
for a in (1, 2, 3):
    lines.append(f"   {ARM_NAMES[a]}: {ha.loc[f'arm_{a}', 'HR']:.2f}  ({ha.loc[f'arm_{a}', 'lo']:.2f} - {ha.loc[f'arm_{a}', 'hi']:.2f})")
lines += ["", f"{TAU} din par event se bache hue:"]
lines += [f"   {ARM_NAMES[a]}: {pt_surv[a]*100:.1f}%" for a in range(4)]
lines += ["", f"RMST fark ({TAU} din):"]
lines += [f"   {ARM_NAMES[a]}: +{pt_rmst[a]-pt_rmst[0]:.0f} din" for a in (1, 2, 3)]
lines += ["", "Teeno combination arms ek dusre se", "saaf alag nahi (pairwise log-rank)."]
ax.text(0.02, 0.98, "6. Khulasa", fontsize=12, fontweight="bold", va="top", transform=ax.transAxes)
ax.text(0.02, 0.90, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.45, transform=ax.transAxes)

fig.suptitle("Treatment ka overall asar  -  ACTG175 (ZDV akele ke muqabley mein)", fontsize=16, fontweight="bold")
fig.savefig(FIG_DIR / "effects_1_summary.png")
print(f"   [graph save hua] {FIG_DIR / 'effects_1_summary.png'}")
print(f"   [table save hua] {TAB_DIR / 'effects_summary.csv'}")
print(f"   [table save hua] {TAB_DIR / 'effects_pairwise_logrank.csv'}")
print("\nEffects mukammal. Agla qadam: models.py (prediction) aur phir heterogeneity.")
if SHOW:
    plt.show()