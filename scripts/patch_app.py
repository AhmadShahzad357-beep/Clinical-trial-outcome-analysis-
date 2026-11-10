"""Patch app/static/app.js: static-site mode, browser-side predictor and a working Live demo button.
Run from the project folder:  python scripts/patch_app.py"""
from pathlib import Path

p = Path("app/static/app.js")
s = p.read_text(encoding="utf-8").replace("\r\n", "\n")
if "STATIC_MODE" in s:
    raise SystemExit("app.js is already patched.")

OLD_API = '  const api = (url, opts) => fetch(url, opts).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });\n'
NEW_API = r'''  const getJSON = (url, opts) => fetch(url, opts).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });

  /* ---------------- data access: FastAPI server OR static site ----------------
     With the FastAPI server, /api/* is used. On a static host (Vercel, GitHub Pages ...) there is no server, so the
     same results are read from /data/*.json (written by scripts/build_web.py) and the patient predictor runs in the
     browser from the exported Cox coefficients. The mode is detected once at start-up. */
  let STATIC_MODE = false;
  let predictorModels = null;
  const ACTG_ARMS = [[0, "ZDV"], [1, "ZDV+ddI"], [2, "ZDV+zal"], [3, "ddI"]];
  const GBSG_ARMS = [[0, "No hormonal therapy"], [1, "Hormonal therapy"]];
  const LIMITS = {
    actg175: { age: [12, 90], wtkg: [30, 170], cd40: [0, 1500], cd80: [0, 6000], preanti: [0, 3500],
               hemo: [0, 1], homo: [0, 1], drugs: [0, 1], z30: [0, 1], race: [0, 1], gender: [0, 1], symptom: [0, 1] },
    gbsg2: { age: [18, 90], tsize: [1, 150], pnodes: [0, 60], progrec: [0, 3000], estrec: [0, 1500], post: [0, 1], grade: [1, 3] },
  };

  function coxCurve(m, x, n = 36) {
    // x: object of raw features -> survival at n points from 0 to tau (same grid as the server)
    let lp = 0;
    m.columns.forEach((c, i) => {
      const ci = m.cont.indexOf(c);
      const v = ci >= 0 ? (x[c] - m.mean[ci]) / m.scale[ci] : x[c];
      lp += m.coef[i] * v;
    });
    const hr = Math.exp(lp);
    const H = (t) => {                       // step function: last baseline value at or before t
      let lo = 0, hi = m.times.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (m.times[mid] <= t) lo = mid; else hi = mid - 1; }
      return m.cumhaz[lo];
    };
    const grid = Array.from({ length: n }, (_, i) => (m.tau * i) / (n - 1));
    const curve = grid.map(t => (t > 0 ? Math.exp(-H(t) * hr) : 1));
    return { grid, curve, sTau: Math.exp(-H(m.tau) * hr), risk: lp };
  }
  const percentile = (sorted, r) => {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < r) lo = mid + 1; else hi = mid; }
    return (lo / sorted.length) * 100;
  };
  function localPredict(trial, p) {
    const lim = LIMITS[trial];
    for (const [k, [lo, hi]] of Object.entries(lim)) if (!(p[k] >= lo && p[k] <= hi)) throw new Error("422");
    const isActg = trial === "actg175", m = predictorModels[isActg ? "actg" : "gbsg"];
    const arms = isActg ? ACTG_ARMS : GBSG_ARMS, base = {};
    const warnings = [];
    m.cont.forEach(c => {
      base[c] = Number(p[c]);
      const [lo, hi] = m.ranges[c];
      if (base[c] < lo || base[c] > hi) warnings.push(`${c} = ${base[c]} is outside the range seen in training (${lo} to ${hi}); extrapolation.`);
    });
    if (isActg) {
      if (![70, 80, 90, 100].includes(Number(p.karnof))) throw new Error("422");
      ["hemo", "homo", "drugs", "z30", "race", "gender", "symptom"].forEach(c => { base[c] = Number(p[c]); });
      base.karnof_70_80 = Number(p.karnof) <= 80 ? 1 : 0; base.karnof_90 = Number(p.karnof) === 90 ? 1 : 0;
    } else {
      base.post = Number(p.post);
      ["tsize", "pnodes", "progrec", "estrec"].forEach(c => { base[c] = Math.log1p(base[c]); });   // same transform as training
      base.g2 = Number(p.grade) === 2 ? 1 : 0; base.g3 = Number(p.grade) === 3 ? 1 : 0;
    }
    const results = {}, curves = {}; let times = null;
    arms.forEach(([arm, name]) => {
      const row = { ...base };
      if (isActg) [1, 2, 3].forEach(a => { row["arm_" + a] = arm === a ? 1 : 0; }); else row.treat = arm;
      const r = coxCurve(m, row);
      times = r.grid; curves[name] = r.curve;
      results[name] = { event_free_prob: r.sTau, risk_score: r.risk, risk_percentile: percentile(m.train_risk, r.risk) };
    });
    const ref = results[arms[0][1]].event_free_prob, dk = isActg ? "difference_vs_ZDV_pts" : "difference_vs_no_therapy_pts";
    Object.values(results).forEach(r => { r[dk] = (r.event_free_prob - ref) * 100; });
    return { trial: isActg ? "ACTG 175" : "GBSG2", horizon_days: m.tau,
             outcome: isActg ? "event-free (no CD4 drop, AIDS or death)" : "recurrence-free survival",
             times, arms: results, curves, warnings };
  }
  async function detectMode() {
    try { const h = await getJSON("/api/health"); STATIC_MODE = h.status !== "ok"; } catch (e) { STATIC_MODE = true; }
  }
  const api = async (url, opts) => {
    if (!STATIC_MODE) return getJSON(url, opts);
    if (url.startsWith("/api/predict/")) {
      if (!predictorModels) predictorModels = await getJSON("/data/predictor.json");
      return localPredict(url.split("/").pop(), JSON.parse(opts.body));
    }
    return getJSON("/data/" + url.replace("/api/", "") + ".json");
  };
'''

NEW_CFG = r'''  async function initConfig() {
    let cfg = { live_demo_url: "", repo_url: "" };
    try { cfg = await api("/api/config"); } catch (e) { /* keep defaults */ }
    const demo = $("#demoBtn"), repo = $("#repoBtn");
    const hosted = !["localhost", "127.0.0.1", "[::1]", ""].includes(location.hostname);
    if (cfg.live_demo_url) demo.href = cfg.live_demo_url;
    else if (hosted) { demo.href = location.origin; demo.removeAttribute("target"); }   // already on the live site
    else demo.addEventListener("click", e => { e.preventDefault(); toast("Set live_demo_url in config.yaml (or the LIVE_DEMO_URL variable) to the deployed address."); });
    if (cfg.repo_url) repo.href = cfg.repo_url;
    else repo.addEventListener("click", e => { e.preventDefault(); toast("Set repo_url in config.yaml to the GitHub repository address."); });
  }

'''

assert OLD_API in s, "original api() line not found - is this the original app.js?"
s = s.replace(OLD_API, NEW_API)
a = s.index("  async function initConfig() {")
b = s.index("  /* ---------------- OVERVIEW")
s = s[:a] + NEW_CFG + s[b:]
old_init = "  async function init() {\n    initChrome(); initConfig();"
assert old_init in s, "init() not found"
s = s.replace(old_init, "  async function init() {\n    await detectMode();\n    initChrome(); initConfig();")
p.write_text(s, encoding="utf-8")
print("app.js patched")