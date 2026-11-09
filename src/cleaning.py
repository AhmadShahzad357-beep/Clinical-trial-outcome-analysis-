"""Cleaning aur audit (ACTG 175): missing values, leakage columns hatana, linearity aur outlier checks.

Chalane ka tareeqa (project folder se):  python src/cleaning.py        (figures save hoti hain)
Window na khule:                          python src/cleaning.py --no-show
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[1]
SHOW = "--no-show" not in sys.argv
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)


def finish(fig_name):
    """Figure save karo; --no-show na ho to window bhi kholo."""
    plt.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close()


def basic_check(data, title):
    print("=" * 60)
    print(title)
    print("=" * 60)
    print("Shape:", data.shape)
    print()
    data.info()
    miss = pd.DataFrame({
        "n_missing": data.isna().sum(),
        "pct_missing": (data.isna().mean() * 100).round(2),
    }).sort_values("n_missing", ascending=False)
    print("\nMissing table (har feature):")
    print(miss.to_string())
    print("\nTotal missing:", int(data.isna().sum().sum()))
    print()


# ---------- 1. Load ----------
df = pd.read_csv(ROOT / "data" / "raw" / "ACTG175_original.csv")

# ---------- 2. Hatane se PEHLE ----------
basic_check(df, "BEFORE: columns hatane se pehle")

# ---------- 3. Hatana ----------
# leakage (randomization ke baad ke): cd496, cd420, cd820, r, offtrt
# ID: pidnum
drop_cols = ["cd496", "cd420", "cd820", "r", "offtrt", "pidnum"]
df = df.drop(columns=drop_cols)

# ---------- 4. Hatane ke BAAD ----------
basic_check(df, "AFTER: columns hatane ke baad")
print("Baqi columns:", df.columns.tolist())
print("Hate hue columns abhi data mein hain?",
      {c: (c in df.columns) for c in drop_cols})


# ---------- 5. Linearity check ----------
cont = ["age", "wtkg", "preanti", "cd40", "cd80"]

base = ["age", "wtkg", "hemo", "homo", "drugs", "karnof", "oprior", "z30",
        "preanti", "race", "gender", "symptom", "treat", "cd40", "cd80"]
d = df[base + ["days", "cens"]].copy()
d = pd.concat([d, pd.get_dummies(df["strat"], prefix="strat", drop_first=True).astype(int)], axis=1)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), cont):
    cols = [x for x in d.columns if x != c]
    cph = CoxPHFitter().fit(d[cols], "days", "cens")
    res = cph.compute_residuals(d[cols], "martingale")["martingale"]
    x = d[c].loc[res.index]
    bins = pd.qcut(x, 10, duplicates="drop")
    m = pd.DataFrame({"x": x, "r": res}).groupby(bins, observed=True).mean()
    ax.scatter(x, res, s=4, alpha=0.2)
    ax.plot(m["x"], m["r"], "r-o")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(c); ax.set_xlabel(c); ax.set_ylabel("martingale residual")
axes.ravel()[-1].axis("off")
plt.tight_layout()
finish("cleaning_1_linearity.png")


# ---------- 6. Outliers check ----------
rows = []
for c in cont:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n = int(((df[c] < lo) | (df[c] > hi)).sum())
    rows.append({"feature": c, "min": df[c].min(), "max": df[c].max(),
                 "lower_fence": round(lo, 1), "upper_fence": round(hi, 1),
                 "n_outliers": n, "pct": round(n / len(df) * 100, 2)})
print(pd.DataFrame(rows).to_string(index=False))

fig, axes = plt.subplots(1, len(cont), figsize=(16, 4))
for ax, c in zip(axes, cont):
    ax.boxplot(df[c]); ax.set_title(c)
plt.tight_layout()
finish("cleaning_2_outliers.png")

# ---------- Save: cleaned data -> processed ----------
save_path = ROOT / "data" / "processed" / "actg175_clean.csv"
df.to_csv(save_path, index=False)

# Check: wapas parh kar dekhein
check = pd.read_csv(save_path)
print("Saved shape:", check.shape)
print("Saved missing:", int(check.isna().sum().sum()))
print("cd496 file mein hai?", "cd496" in check.columns)