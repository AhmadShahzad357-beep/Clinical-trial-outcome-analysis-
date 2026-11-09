"""
VALIDATION - ACTG175: kya mareez-ba-mareez asar ke andaze (CATE) ASLI hain ya sirf shor?
Chalane ka tareeqa (project folder se):
    python src/validation.py
Jaldi test (chhota version):  ... validation.py --fast        Window na khule:  ... validation.py --no-show
Pehle cate.py chalana zaroori hai (woh cate_oof_train.csv aur cate_test.csv banata hai).

Problem: asli CATE kabhi nazar nahi aata (har mareez ko sirf ek arm milta hai). Isliye "doubly-robust pseudo-outcome"
(psi) istemal hota hai: ek shor wala magar sahi (unbiased) andaza jo randomized trial mein har mareez ke asar ka
nishana hai. Andazon ko psi se parakhte hain:

  Jaanch 1 (Constant-effect muqabla): kya mareez-ba-mareez andaza "sab ko ek jaisa average asar" wale andaze se behtar hai?
  Jaanch 2 (Calibration slope):       psi ko andaze par regress karte hain. Slope 1 = bilkul sahi farq, 0 = andaze ka farq
                                      bekaar (shor). Slope ka daira 0 ko na chhuye to farq ka saboot.
  Jaanch 3 (Tertile calibration):     mareezon ko andaze ke hisaab se 3 groups (kam/darmiyana/zyada faida) mein baanten;
                                      asal mein bhi yehi tarteeb nikalti hai?
  Jaanch 4 (Global null):             treatment labels shuffle kar ke 'jhoota farq' ka paimana: bina asar ke bhi DR-Ridge
                                      kitna 'faida' dikha deta hai.
Train par out-of-fold andazon se pehle, phir TEST par sirf EK BAAR.
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cate as C                                      # load, SurvModel, dr_pseudo (cate.py ke functions dobara istemal)

warnings.filterwarnings("ignore")
ROOT, PROCESSED, FIG_DIR, TAB_DIR = C.ROOT, C.PROCESSED, C.FIG_DIR, C.TAB_DIR
FAST = "--fast" in sys.argv
SHOW = "--no-show" not in sys.argv
SEED = C.SEED
N_BOOT = 200 if FAST else 1000          # gain ke daire ke liye bootstrap
N_PERM = 8 if FAST else 40              # global null ke liye shuffle ki tadaad (har arm ke liye)
ARMS, ARM_NAMES, ARM_COLORS, METHODS, MCOL = C.ARMS, C.ARM_NAMES, C.ARM_COLORS, C.METHODS, C.MCOL


def head(t):
    print("=" * 62, f"\n{t}\n" + "=" * 62)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def fmt_p(p):
    return "< 0.0001" if p < 1e-4 else f"{p:.3f}"


# ------------------------------------------------------------------
# Naap
# ------------------------------------------------------------------
def ols_slope(psi, tau):
    """psi = a + b * tau_hat. b = calibration slope. Heteroskedasticity-robust SE (HC1)."""
    n = len(psi)
    if np.std(tau) < 1e-9:
        return np.nan, np.nan, np.nan
    X = np.column_stack([np.ones(n), tau])
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ psi; e = psi - X @ b
    Xe = X * e[:, None]
    V = XtXi @ (Xe.T @ Xe) @ XtXi * n / (n - 2)
    se = float(np.sqrt(V[1, 1])); z = b[1] / se
    return float(b[1]), se, float(2 * (1 - norm.cdf(abs(z))))


def gain_stats(psi, tau, rng, B):
    """Constant-effect model ke muqabley mein squared-error ka faida (relative %). Bootstrap daira."""
    def g(p, t):
        lc = np.mean((p - p.mean()) ** 2)
        return (lc - np.mean((p - t) ** 2)) / lc * 100
    n = len(psi); obs = g(psi, tau)
    bs = np.array([g(psi[i], tau[i]) for i in (rng.integers(0, n, n) for _ in range(B))])
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tertiles(psi, tau):
    q = pd.qcut(pd.Series(tau), 3, labels=False, duplicates="drop").values
    rows = []
    for k in sorted(set(q)):
        m = q == k
        rows.append((tau[m].mean(), psi[m].mean(), 1.96 * psi[m].std(ddof=1) / np.sqrt(m.sum())))
    return rows        # (andaza, asal(psi), 95% half-width)


def evaluate(psi, tau, rng, B):
    gain, glo, ghi = gain_stats(psi, tau, rng, B)
    b, se, p = ols_slope(psi, tau)
    tt = tertiles(psi, tau)
    diff = tt[-1][1] - tt[0][1]
    dse = np.sqrt((tt[-1][2] / 1.96) ** 2 + (tt[0][2] / 1.96) ** 2)
    return dict(gain=gain, gain_lo=glo, gain_hi=ghi, slope=b, slope_se=se, slope_p=p,
                top_minus_bottom=diff, tb_lo=diff - 1.96 * dse, tb_hi=diff + 1.96 * dse, tert=tt)


# ------------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    Xtr, ytr, ydf_tr, a_tr = C.load("train")
    Xte, yte, ydf_te, a_te = C.load("test")
    n, nte = len(Xtr), len(Xte)
    oof = pd.read_csv(PROCESSED / "cate_oof_train.csv", index_col="row")
    tst = pd.read_csv(PROCESSED / "cate_test.csv", index_col="row")
    strata = pd.Series(a_tr).astype(str) + "_" + ydf_tr["cens"].astype(str).values
    folds = list(StratifiedKFold(C.N_FOLDS, shuffle=True, random_state=SEED).split(Xtr, strata))
    rng = np.random.default_rng(SEED)

    head("STEP 1: Tayyari (psi banana)")
    print("psi = doubly-robust pseudo-outcome (asar ka shor wala magar sahi andaza). Train par out-of-fold,")
    print("test par train se seekhe hue Cox models se (test ke outcome se kuch seekha nahi jata).")
    mu_oof = {a: np.zeros(n) for a in range(4)}
    for tr, va in folds:
        for a in range(4):
            idx = tr[a_tr[tr] == a]
            mu_oof[a][va] = C.SurvModel("cox", len(idx), n).fit(Xtr.iloc[idx], ytr[idx]).rmst(Xtr.iloc[va])
    mu_te = {}
    for a in range(4):
        idx = np.where(a_tr == a)[0]
        mu_te[a] = C.SurvModel("cox", len(idx), n).fit(Xtr.iloc[idx], ytr[idx]).rmst(Xte)
    print(f"Taiyar [{time.time()-t0:.0f} sec]\n")

    # ---------------- Step 2: train (out-of-fold) ----------------
    head("STEP 2: Train par jaanch (out-of-fold andaze)")
    print("gain%   : constant-effect model ke muqabley mein psi ki ghalti kitni % kam hui (0 se oopar = behtar)")
    print("slope   : calibration slope (1 = bilkul sahi farq, 0 = farq shor)")
    print("top-bot : sab se zyada vs sab se kam faida wale tertile ka asal fark (din)")
    print("DHYAN   : train ke out-of-fold andaze par slope shor ki halat mein MANFI aa sakta hai (cross-validation ka asar).")
    print("          Isliye train ke slope ka faisla Step 3 ke shuffle-null se hoga, akele asymptotic p se nahi.\n")
    res = {}
    keys = []
    for arm in ARMS:
        pos, psi, _, A = C.dr_pseudo(arm, a_tr, ydf_tr, mu_oof)
        for m in METHODS:
            tau = oof[f"{ARM_NAMES[arm]}__{m}"].values[pos]
            r = evaluate(psi, tau, rng, N_BOOT)
            res[(arm, m)] = r; keys.append((arm, m))
    ph = holm([res[k]["slope_p"] if not np.isnan(res[k]["slope_p"]) else 1.0 for k in keys])
    for k, p_ in zip(keys, ph):
        res[k]["slope_p_holm"] = p_
    rows = []
    for arm in ARMS:
        print(f"{ARM_NAMES[arm]} vs ZDV:")
        print(f"   {'tareeqa':9s} {'gain% (95% daira)':>24s} {'slope (95% daira)':>24s} {'p (Holm)':>9s} {'top-bot din (95%)':>24s}")
        for m in METHODS:
            r = res[(arm, m)]
            print(f"   {m:9s} {r['gain']:7.2f} ({r['gain_lo']:6.2f},{r['gain_hi']:6.2f})   "
                  f"{r['slope']:6.2f} ({r['slope']-1.96*r['slope_se']:6.2f},{r['slope']+1.96*r['slope_se']:6.2f})   "
                  f"{fmt_p(r['slope_p_holm']):>9s}   {r['top_minus_bottom']:6.1f} ({r['tb_lo']:6.1f},{r['tb_hi']:6.1f})")
            rows.append({"set": "train_oof", "arm": ARM_NAMES[arm], "method": m, "gain_%": round(r["gain"], 2),
                         "gain_lo": round(r["gain_lo"], 2), "gain_hi": round(r["gain_hi"], 2),
                         "slope": round(r["slope"], 3), "slope_lo": round(r["slope"] - 1.96 * r["slope_se"], 3),
                         "slope_hi": round(r["slope"] + 1.96 * r["slope_se"], 3), "p_holm": round(r["slope_p_holm"], 4),
                         "top_minus_bottom_din": round(r["top_minus_bottom"], 1)})
        print()
    evid = [(ARM_NAMES[a], m) for (a, m) in keys
            if res[(a, m)]["slope_p_holm"] < 0.05 and res[(a, m)]["slope"] > 0 and res[(a, m)]["gain_lo"] > 0]
    print("Saboot (slope Holm p < 0.05, slope > 0, aur gain ka daira 0 se oopar):",
          evid if evid else "KOI NAHI. Kisi arm, kisi tareeqe mein mareez-ba-mareez farq ka saaf saboot nahi mila.")
    print(f"[{time.time()-t0:.0f} sec guzre]\n")

    # ---------------- Step 3: global null (DR-Ridge) ----------------
    head("STEP 3: Global null (treatment labels shuffle) - sirf DR-Ridge")
    print("Bina kisi asli asar ke bhi DR-Ridge kitna 'gain' dikhata hai? Har arm ke liye labels ko ZDV/arm ke beech")
    print(f"{N_PERM} baar shuffle kar ke poora pipeline dobara chalaya. Asli gain null se bara na ho to farq shor.\n")

    def dr_ridge_oof(arm, a_vec):
        mu = {0: np.zeros(n), arm: np.zeros(n)}
        for tr, va in folds:
            for g in (0, arm):
                idx = tr[a_vec[tr] == g]
                mu[g][va] = C.SurvModel("cox", len(idx), n).fit(Xtr.iloc[idx], ytr[idx]).rmst(Xtr.iloc[va])
        pos, psi, _, _ = C.dr_pseudo(arm, a_vec, ydf_tr, mu)
        tau = np.zeros(n)
        for tr, va in folds:
            use = np.isin(pos, tr)
            Xp = Xtr.iloc[pos[use]]; sc = StandardScaler().fit(Xp)
            r = RidgeCV(alphas=np.logspace(-1, 4, 26)).fit(sc.transform(Xp), psi[use])
            tau[va] = r.predict(sc.transform(Xtr.iloc[va]))
        return psi, tau[pos]

    def gain_only(psi, tau):
        lc = np.mean((psi - psi.mean()) ** 2)
        return (lc - np.mean((psi - tau) ** 2)) / lc * 100

    null = {}
    for arm in ARMS:
        psi_o, tau_o = dr_ridge_oof(arm, a_tr)
        obs_gain, obs_sd = gain_only(psi_o, tau_o), tau_o.std()
        obs_slope = ols_slope(psi_o, tau_o)[0]
        pair = np.where((a_tr == 0) | (a_tr == arm))[0]
        gs, sds, sl = [], [], []
        for _ in range(N_PERM):
            a_p = a_tr.copy(); a_p[pair] = rng.permutation(a_tr[pair])
            psi_p, tau_p = dr_ridge_oof(arm, a_p)
            gs.append(gain_only(psi_p, tau_p)); sds.append(tau_p.std()); sl.append(ols_slope(psi_p, tau_p)[0])
        gs, sds, sl = np.array(gs), np.array(sds), np.array(sl)
        p_gain = (1 + np.sum(gs >= obs_gain)) / (1 + N_PERM)
        p_slope = (1 + np.sum(sl >= obs_slope)) / (1 + N_PERM)
        null[arm] = dict(obs_gain=obs_gain, gs=gs, obs_sd=obs_sd, sds=sds, p=p_gain, obs_slope=obs_slope, sl=sl, p_slope=p_slope)
        print(f"{ARM_NAMES[arm]}:")
        print(f"   gain : asli {obs_gain:6.2f}% | null average {gs.mean():6.2f}%, 95% hissa {np.percentile(gs,95):6.2f}% tak | p = {p_gain:.3f}")
        print(f"   slope: asli {obs_slope:6.2f}  | null average {sl.mean():6.2f} (shor mein bhi slope aisa hi manfi aata hai) | p = {p_slope:.3f}")
        print(f"   SD   : asli {obs_sd:5.1f} din | null {sds.mean():5.1f} din")
    print(f"[{time.time()-t0:.0f} sec guzre]\n")

    # ---------------- Step 4: test (ek baar) ----------------
    head("STEP 4: TEST par ek baar jaanch")
    print("Ab test khul gaya. Tareeqe aur settings train se tay ho chuke hain, test par kuch tune nahi hua.\n")
    res_te = {}
    for arm in ARMS:
        pos, psi, _, _ = C.dr_pseudo(arm, a_te, ydf_te, mu_te)
        for m in METHODS:
            tau = tst[f"{ARM_NAMES[arm]}__{m}"].values[pos]
            res_te[(arm, m)] = evaluate(psi, tau, rng, N_BOOT)
    ph = holm([res_te[k]["slope_p"] if not np.isnan(res_te[k]["slope_p"]) else 1.0 for k in keys])
    for k, p_ in zip(keys, ph):
        res_te[k]["slope_p_holm"] = p_
    for arm in ARMS:
        print(f"{ARM_NAMES[arm]} vs ZDV (test):")
        print(f"   {'tareeqa':9s} {'gain% (95% daira)':>24s} {'slope (95% daira)':>24s} {'p (Holm)':>9s} {'top-bot din (95%)':>24s}")
        for m in METHODS:
            r = res_te[(arm, m)]
            print(f"   {m:9s} {r['gain']:7.2f} ({r['gain_lo']:6.2f},{r['gain_hi']:6.2f})   "
                  f"{r['slope']:6.2f} ({r['slope']-1.96*r['slope_se']:6.2f},{r['slope']+1.96*r['slope_se']:6.2f})   "
                  f"{fmt_p(r['slope_p_holm']):>9s}   {r['top_minus_bottom']:6.1f} ({r['tb_lo']:6.1f},{r['tb_hi']:6.1f})")
            rows.append({"set": "test", "arm": ARM_NAMES[arm], "method": m, "gain_%": round(r["gain"], 2),
                         "gain_lo": round(r["gain_lo"], 2), "gain_hi": round(r["gain_hi"], 2),
                         "slope": round(r["slope"], 3), "slope_lo": round(r["slope"] - 1.96 * r["slope_se"], 3),
                         "slope_hi": round(r["slope"] + 1.96 * r["slope_se"], 3), "p_holm": round(r["slope_p_holm"], 4),
                         "top_minus_bottom_din": round(r["top_minus_bottom"], 1)})
        print()
    evid_te = [(ARM_NAMES[a], m) for (a, m) in keys
               if res_te[(a, m)]["slope_p_holm"] < 0.05 and res_te[(a, m)]["slope"] > 0 and res_te[(a, m)]["gain_lo"] > 0]
    print("Test par saboot:", evid_te if evid_te else "KOI NAHI.")
    pd.DataFrame(rows).to_csv(TAB_DIR / "validation_results.csv", index=False)

    # ---------------- Step 5: graph ----------------
    head("STEP 5: Graph")
    plt.rcParams.update({"axes.grid": False})
    fig = plt.figure(figsize=(20, 15.5), constrained_layout=True)
    gs_ = fig.add_gridspec(3, 3)

    def forest(ax, R, stat, lo_f, hi_f, title, xlabel, refs, xlim=None):
        y = 0
        yt, yl = [], []
        for arm in ARMS[::-1]:
            for m in METHODS[::-1]:
                r = R[(arm, m)]
                lo, hi, v = lo_f(r), hi_f(r), stat(r)
                ax.plot([lo, hi], [y, y], color=MCOL[m], lw=2.2)
                ax.scatter(v, y, color=MCOL[m], s=45, zorder=3)
                y += 1
            yt.append(y - 3); yl.append(ARM_NAMES[arm]); y += 1
        for x, ls_ in refs:
            ax.axvline(x, color="#333333", ls=ls_, lw=1.2)
        ax.set_yticks(yt); ax.set_yticklabels(yl); ax.set_xlabel(xlabel); ax.set_title(title)
        if xlim:
            ax.set_xlim(xlim)

    leg = [Patch(color=MCOL[m], label=m) for m in METHODS]
    ax = fig.add_subplot(gs_[0, 0])
    forest(ax, res, lambda r: r["gain"], lambda r: r["gain_lo"], lambda r: r["gain_hi"],
           "1. Train (out-of-fold): constant-effect se behtar?", "gain % (0 se oopar = behtar, daira 0 ko chhuye to farq nahi)",
           [(0, "--")])
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5)
    ax = fig.add_subplot(gs_[0, 1])
    forest(ax, res, lambda r: r["slope"], lambda r: r["slope"] - 1.96 * r["slope_se"], lambda r: r["slope"] + 1.96 * r["slope_se"],
           "2. Train (out-of-fold): calibration slope (manfi = CV ka asar)", "slope (0 = farq shor, 1 = bilkul sahi)", [(0, "--"), (1, ":")], xlim=(-9, 7))

    best_m = max(METHODS, key=lambda m: np.mean([res[(a, m)]["gain"] for a in ARMS]))

    def calib(ax, R, title):
        for arm in ARMS:
            tt = R[(arm, best_m)]["tert"]
            xs, ys, hw = [t[0] for t in tt], [t[1] for t in tt], [t[2] for t in tt]
            ax.errorbar(xs, ys, yerr=hw, fmt="-o", color=ARM_COLORS[arm], capsize=4, lw=2, ms=7, label=ARM_NAMES[arm])
        allv = [t[i] for a in ARMS for t in R[(a, best_m)]["tert"] for i in (0, 1)]
        lims = [min(allv) - 40, max(allv) + 40]
        ax.plot(lims, lims, color="#333333", ls="--", lw=1.2, label="Bilkul sahi")
        ax.set_xlabel(f"{best_m} ka andaza (din, group ka average)"); ax.set_ylabel("Asal asar (psi ka average, din)")
        ax.legend(loc="upper left"); ax.set_title(title)

    ax = fig.add_subplot(gs_[0, 2])
    calib(ax, res, f"3. Train: 3 groups (kam/darmiyana/zyada faida), {best_m}")

    for k_, arm in enumerate(ARMS):
        ax = fig.add_subplot(gs_[1, k_])
        d = null[arm]
        ax.hist(d["gs"], bins=12, color="#9AA5B1", edgecolor="white", label="Null (shuffle)")
        ax.axvline(d["obs_gain"], color="#C8553D", lw=2.5, label=f"Asli ({d['obs_gain']:.2f}%)")
        ax.set_xlabel("DR-Ridge ka gain %"); ax.set_ylabel("Shuffle ki tadaad")
        ax.legend(loc="upper right"); ax.set_title(f"{4+k_}. Global null: {ARM_NAMES[arm]} (p = {d['p']:.3f})")

    ax = fig.add_subplot(gs_[2, 0])
    forest(ax, res_te, lambda r: r["gain"], lambda r: r["gain_lo"], lambda r: r["gain_hi"],
           "7. TEST: constant-effect se behtar?", "gain % (0 se oopar = behtar)", [(0, "--")])
    ax = fig.add_subplot(gs_[2, 1])
    forest(ax, res_te, lambda r: r["slope"], lambda r: r["slope"] - 1.96 * r["slope_se"], lambda r: r["slope"] + 1.96 * r["slope_se"],
           "8. TEST: calibration slope", "slope (0 = farq shor, 1 = bilkul sahi)", [(0, "--"), (1, ":")], xlim=(-12, 36))
    ax = fig.add_subplot(gs_[2, 2])
    calib(ax, res_te, f"9. TEST: 3 groups, {best_m}")

    fig.suptitle("Validation: kya mareez-ba-mareez asar ke andaze asli hain?  (ACTG175)", fontsize=17, fontweight="bold")
    fig.savefig(FIG_DIR / "validation_1_summary.png", dpi=150, bbox_inches="tight")
    print(f"   [graph save hua] {FIG_DIR / 'validation_1_summary.png'}")
    print(f"   [table save hua] {TAB_DIR / 'validation_results.csv'}")
    print(f"\nValidation mukammal ({time.time()-t0:.0f} sec).")
    if SHOW:
        plt.show()