"""
Build the static site in web/ (deployable to Vercel, Netlify, GitHub Pages ...).

    python scripts/build_web.py

What it does
  * copies app/static (HTML, CSS, JS, Chart.js) and results/figures into web/
  * dumps every GET endpoint of the FastAPI app to web/data/*.json
  * exports the two ridge Cox models (coefficients, scaling, baseline hazard) to web/data/predictor.json
    so the patient predictor runs in the browser, with no server.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import main as M  # noqa: E402
import predictor as P  # noqa: E402

WEB = ROOT / "web"
DATA = WEB / "data"


def dump(name, obj):
    (DATA / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def export_model(bundle_part, cont_cols, tau):
    model = bundle_part["model"]
    scaler = model[0].named_transformers_["sc"]
    cox = model[-1]
    cols_after = list(cont_cols) + [c for c in bundle_part["columns"] if c not in cont_cols]  # ColumnTransformer order
    H = cox.cum_baseline_hazard_
    keep = H.x <= tau
    times = [0.0] + [float(t) for t in H.x[keep]]
    cumhaz = [0.0] + [float(v) for v in H.y[keep]]
    return {
        "cont": list(cont_cols),
        "columns": cols_after,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": cox.coef_.tolist(),
        "times": times,
        "cumhaz": [round(v, 8) for v in cumhaz],
        "tau": tau,
        "ranges": bundle_part["ranges"],
        "train_risk": [round(float(r), 5) for r in bundle_part["train_risk"]],
    }


def main():
    if WEB.exists():
        shutil.rmtree(WEB)
    (WEB / "static").mkdir(parents=True)
    DATA.mkdir(parents=True)

    for f in ("app.js", "style.css"):
        shutil.copy(M.STATIC / f, WEB / "static" / f)
    shutil.copytree(M.STATIC / "vendor", WEB / "static" / "vendor")
    shutil.copy(M.STATIC / "index.html", WEB / "index.html")
    shutil.copytree(M.FIGURES, WEB / "figures")

    dump("overview", M.overview())
    dump("km", M.km())
    dump("effects", M.effects())
    dump("models", M.models())
    dump("heterogeneity", M.heterogeneity())
    dump("figures", M.figures())
    dump("config", M.config())
    dump("health", {"status": "ok", "mode": "static"})

    pr = P.Predictor()
    b = pr.b
    dump("predictor", {
        "actg": export_model(b["actg"], P.ACTG_CONT, P.ACTG_TAU),
        "gbsg": export_model(b["gbsg"], P.GBSG_CONT, P.GBSG_TAU),
    })
    (WEB / "vercel.json").write_text(json.dumps({"cleanUrls": True, "headers": [
        {"source": "/data/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=300"}]},
        {"source": "/figures/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=86400"}]}]}, indent=2))
    print("web/ built:", sum(1 for _ in WEB.rglob("*") if _.is_file()), "files")


if __name__ == "__main__":
    main()