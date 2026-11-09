/* Clinical Trial Outcome Analysis dashboard. Plain JavaScript, Chart.js (vendored) for line/bar charts, hand-built SVG for forest plots. */
(() => {
  "use strict";

  /* ---------------- helpers ---------------- */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const api = (url, opts) => fetch(url, opts).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
  const fmt = (v, d = 1) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } },
  };
  let toastTimer;
  function toast(msg) {
    const t = $("#toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 3400);
  }

  const ARM_COLOR = () => ({
    "ZDV": css("--c-zdv"), "ZDV+ddI": css("--c-ddi-zdv"), "ZDV+zal": css("--c-zal"), "ddI": css("--c-ddi"),
    "No hormonal therapy": css("--c-zdv"), "Hormonal therapy": css("--c-zal"),
  });
  const METHOD_COLOR = () => ({
    "S-Cox": "#94a3b8", "T-Cox": css("--c-zdv"), "T-RSF": css("--c-zal"), "DR-Ridge": "#c8553d", "DR-RF": css("--c-ddi"),
  });

  /* ---------------- Chart.js defaults ---------------- */
  const charts = {};
  function setChart(id, config) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart($("#" + id), config);
  }
  function chartBase() {
    const ink = css("--muted"), grid = css("--grid");
    return {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { color: css("--ink-2"), usePointStyle: true, boxWidth: 8, padding: 16, font: { size: 12 } } },
        tooltip: { backgroundColor: css("--ink"), titleColor: css("--bg"), bodyColor: css("--bg"), padding: 10, cornerRadius: 8, displayColors: true },
      },
      scales: {
        x: { ticks: { color: ink, font: { size: 11 } }, grid: { color: grid, drawTicks: false }, border: { color: css("--line-2") } },
        y: { ticks: { color: ink, font: { size: 11 } }, grid: { color: grid }, border: { display: false } },
      },
    };
  }

  /* ---------------- SVG forest plot ---------------- */
  function niceTicks(min, max, n = 5) {
    const span = max - min, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const start = Math.ceil(min / step) * step, out = [];
    for (let v = start; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function forest(el, rows, o = {}) {
    const W = 540, L = o.labelW || 190, Rm = 18, T = 8;
    const xs = rows.filter(r => !r.header).flatMap(r => [r.lo, r.hi, r.est]);
    let min = o.min ?? Math.min(...xs), max = o.max ?? Math.max(...xs);
    if (o.min == null || o.max == null) { const pad = (max - min) * .08; min -= pad; max += pad; }
    if (o.ref != null) { min = Math.min(min, o.ref - (max - min) * .04); max = Math.max(max, o.ref + (max - min) * .04); }
    const px = (v) => L + 10 + (v - min) / (max - min) * (W - L - Rm - 10);
    const ticks = niceTicks(min, max, 5), dec = o.dec ?? (max - min < 3 ? 1 : 0);
    let y = T, body = "", grid = "";
    rows.forEach(r => {
      if (r.header) { body += `<text x="0" y="${y + 18}" class="sub" style="font-weight:700;letter-spacing:.06em">${esc(r.header.toUpperCase())}</text>`; y += 28; return; }
      const h = o.rowH || 44, cy = y + h / 2, col = r.color || css("--accent");
      body += `<text x="0" y="${cy - 2}" class="lbl">${esc(r.label)}</text>`;
      if (r.sub) body += `<text x="0" y="${cy + 13}" class="sub">${esc(r.sub)}</text>`;
      if (r.lo != null) body += `<line x1="${px(r.lo)}" x2="${px(r.hi)}" y1="${cy}" y2="${cy}" stroke="${col}" stroke-width="2.6" stroke-linecap="round"/>`;
      body += `<circle cx="${px(r.est)}" cy="${cy}" r="${r.lo != null ? 5 : 4}" fill="${col}" stroke="${css("--surface")}" stroke-width="1.5"><title>${esc(r.tip || "")}</title></circle>`;
      y += h;
    });
    const H = y + 30;
    ticks.forEach(t => {
      grid += `<line class="gridline" x1="${px(t)}" x2="${px(t)}" y1="${T}" y2="${y}"/><text class="tick" x="${px(t)}" y="${y + 16}" text-anchor="middle">${fmt(t, dec)}</text>`;
    });
    if (o.ref != null) grid += `<line class="ref" x1="${px(o.ref)}" x2="${px(o.ref)}" y1="${T}" y2="${y}"/>`;
    if (o.xlabel) grid += `<text class="sub" x="${px((min + max) / 2)}" y="${y + 30}" text-anchor="middle">${esc(o.xlabel)}</text>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H + (o.xlabel ? 14 : 0)}" role="img" aria-label="${esc(o.aria || "Forest plot")}">${grid}${body}<line class="axis" x1="${L + 10}" x2="${W - Rm}" y1="${y}" y2="${y}"/></svg>`;
  }

  /* ---------------- tables ---------------- */
  function renderTable(el, cols, rows, opt = {}) {
    el.innerHTML = `<thead><tr>${cols.map(c => `<th class="${c.num ? "num" : ""}">${c.h}</th>`).join("")}</tr></thead><tbody>` +
      rows.map(r => `<tr class="${opt.pick && opt.pick(r) ? "pick" : ""}">${cols.map(c => `<td class="${c.num ? "num" : ""}">${c.f(r)}</td>`).join("")}</tr>`).join("") + "</tbody>";
  }

  /* ---------------- state ---------------- */
  const D = {};
  const state = { kmTrial: "actg175", modelTrial: "actg", valSet: "test", predTrial: "actg175" };
  const renderers = [];

  /* ---------------- theme, nav, config ---------------- */
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    $("#themeIcon").innerHTML = t === "dark"
      ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.9 4.9l1.4 1.4m11.4 11.4l1.4 1.4M2 12h2m16 0h2M4.9 19.1l1.4-1.4m11.4-11.4l1.4-1.4"/>'
      : '<path d="M21 13A9 9 0 1111 3a7 7 0 0010 10z"/>';
  }
  function initChrome() {
    const saved = store.get("theme");
    applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
    $("#themeBtn").onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      store.set("theme", next); applyTheme(next); renderers.forEach(f => f());
    };
    const side = $("#sidebar"), scrim = $("#scrim");
    const closeMenu = () => { side.classList.remove("open"); scrim.classList.remove("open"); };
    $("#menuBtn").onclick = () => { side.classList.add("open"); scrim.classList.add("open"); };
    scrim.onclick = closeMenu;
    $$("#nav a").forEach(a => a.addEventListener("click", closeMenu));
    const links = new Map($$("#nav a").map(a => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) { links.forEach(l => l.classList.remove("active")); links.get(e.target.id)?.classList.add("active"); } });
    }, { rootMargin: "-30% 0px -60% 0px" });
    $$("main section").forEach(s => io.observe(s));
  }
  async function initConfig() {
    let cfg = { live_demo_url: "", repo_url: "" };
    try { cfg = await api("/api/config"); } catch (e) { /* keep defaults */ }
    const demo = $("#demoBtn"), repo = $("#repoBtn");
    if (cfg.live_demo_url) demo.href = cfg.live_demo_url;
    else demo.addEventListener("click", e => { e.preventDefault(); toast("Live demo link will be added after the project is published to GitHub."); });
    if (cfg.repo_url) repo.href = cfg.repo_url;
    else repo.addEventListener("click", e => { e.preventDefault(); toast("Repository link will be added after the project is published to GitHub."); });
  }

  /* ---------------- OVERVIEW ---------------- */
  function renderKpis() {
    const [a, g] = D.overview.trials;
    const items = [
      { k: "Patients analysed", v: (a.patients + g.patients).toLocaleString("en-US"), s: `ACTG 175: ${a.patients.toLocaleString("en-US")}  |  GBSG2: ${g.patients}`, c: "" },
      { k: "Events observed", v: (a.events + g.events).toLocaleString("en-US"), s: `${a.event_rate}% and ${g.event_rate}% event rate`, c: "" },
      { k: "Strongest effect", v: "HR " + fmt(a.best_hr_adj, 2), s: `ZDV+ddI vs ZDV, +${fmt(a.best_rmst_days, 1)} days at 900 d`, c: "blue" },
      { k: "Cox C-index (test)", v: `${fmt(a.cox_test_c, 3)} / ${fmt(g.cox_test_c, 3)}`, s: "ACTG 175 / GBSG2, test set opened once", c: "blue" },
    ];
    $("#kpis").innerHTML = items.map(i => `<div class="kpi ${i.c}"><small>${i.k}</small><strong>${i.v}</strong><span>${i.s}</span></div>`).join("");
  }
  function renderKM() {
    const d = D.km[state.kmTrial], colors = ARM_COLOR();
    $("#kmSub").textContent = state.kmTrial === "actg175" ? "Probability of staying event-free, by treatment arm" : "Probability of staying recurrence-free, by therapy";
    const all = Object.values(d.groups).flatMap(g => g.survival);
    const ymin = Math.max(0, Math.floor((Math.min(...all) - .05) * 10) / 10);
    const base = chartBase();
    const horizonLine = {
      id: "horizon", afterDatasetsDraw(c) {
        const { ctx, chartArea: a, scales: { x } } = c;
        const idx = d.times.findIndex(t => t >= d.horizon); const px = x.getPixelForValue(idx);
        ctx.save(); ctx.strokeStyle = css("--muted"); ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(px, a.top); ctx.lineTo(px, a.bottom); ctx.stroke();
        ctx.fillStyle = css("--muted"); ctx.font = "11px sans-serif"; ctx.fillText("Horizon " + d.horizon + " d", px + 6, a.top + 12); ctx.restore();
      },
    };
    setChart("kmChart", {
      type: "line", plugins: [horizonLine],
      data: { labels: d.times, datasets: Object.entries(d.groups).map(([name, g]) => ({
        label: `${name} (n=${g.n})`, data: g.survival, borderColor: colors[name], backgroundColor: colors[name], stepped: true, pointRadius: 0, borderWidth: 2.4, pointHoverRadius: 4,
      })) },
      options: { ...base, interaction: { mode: "index", intersect: false },
        scales: { x: { ...base.scales.x, title: { display: true, text: "Days since randomization", color: css("--muted") }, ticks: { ...base.scales.x.ticks, maxTicksLimit: 9, callback(v) { return this.getLabelForValue(v); } } },
                  y: { ...base.scales.y, min: ymin, max: 1, ticks: { ...base.scales.y.ticks, callback: v => Math.round(v * 100) + "%" } } },
        plugins: { ...base.plugins, tooltip: { ...base.plugins.tooltip, callbacks: { title: i => "Day " + i[0].label, label: c => ` ${c.dataset.label.split(" (")[0]}: ${(c.parsed.y * 100).toFixed(1)}%` } } } },
    });
  }

  /* ---------------- EFFECTS ---------------- */
  function renderEffects() {
    const E = D.effects, colors = ARM_COLOR();
    forest($("#rmstForest"), E.arms.map(a => ({ label: a.arm, sub: `${fmt(a.rmst)} days (${fmt(a.rmst_lo)} to ${fmt(a.rmst_hi)})`, est: a.rmst, lo: a.rmst_lo, hi: a.rmst_hi, color: colors[a.arm], tip: `${a.arm}: +${a.rmst} days` })),
      { ref: 0, min: 0, max: 120, dec: 0, rowH: 64, xlabel: "Extra event-free days in the first 900 days", aria: "Restricted mean survival time gain versus ZDV" });

    setChart("evFreeChart", {
      type: "bar",
      data: { labels: ["ZDV", ...E.arms.map(a => a.arm)], datasets: [{ data: [E.reference_event_free_900, ...E.arms.map(a => a.event_free_900)],
        backgroundColor: ["ZDV", ...E.arms.map(a => a.arm)].map(n => colors[n]), borderRadius: 6, maxBarThickness: 64 }] },
      options: { ...chartBase(), plugins: { legend: { display: false }, tooltip: { ...chartBase().plugins.tooltip, callbacks: { label: c => ` ${c.parsed.y.toFixed(1)}% event-free` } } },
        scales: { x: chartBase().scales.x, y: { ...chartBase().scales.y, min: 50, max: 100, ticks: { ...chartBase().scales.y.ticks, callback: v => v + "%" } } } },
      plugins: [{ id: "lbl", afterDatasetsDraw(c) { const { ctx } = c; ctx.save(); ctx.fillStyle = css("--ink"); ctx.font = "600 12px sans-serif"; ctx.textAlign = "center";
        c.getDatasetMeta(0).data.forEach((b, i) => ctx.fillText(c.data.datasets[0].data[i].toFixed(1) + "%", b.x, b.y - 7)); ctx.restore(); } }],
    });

    renderTable($("#effTable"), [
      { h: "Arm", f: r => `<span class="dot" style="background:${colors[r.arm]}"></span>${r.arm}` },
      { h: "HR (crude)", num: 1, f: r => fmt(r.hr_unadj, 2) },
      { h: "HR (adjusted)", num: 1, f: r => fmt(r.hr_adj, 2) },
      { h: "Event-free 900 d", num: 1, f: r => fmt(r.event_free_900) + "%" },
      { h: "Risk difference", num: 1, f: r => `+${fmt(r.risk_diff)} pts (${fmt(r.risk_lo)} to ${fmt(r.risk_hi)})` },
      { h: "NNT", num: 1, f: r => fmt(r.nnt) },
      { h: "RMST gain", num: 1, f: r => `+${fmt(r.rmst)} d (${fmt(r.rmst_lo)} to ${fmt(r.rmst_hi)})` },
    ], E.arms);

    const subs = E.subgroups.map(s => ({ label: s.subgroup === "Sab mareez" ? "All patients" : s.subgroup, sub: `n=${s.n}, ${s.events} events  |  HR ${fmt(s.HR, 2)} (${fmt(s.lo, 2)} to ${fmt(s.hi, 2)})`,
      est: s.HR, lo: s.lo, hi: s.hi, color: s.hi >= 1 ? css("--warn") : css("--accent"), tip: `HR ${s.HR}` }));
    forest($("#subForest"), subs, { ref: 1, min: 0.2, max: 1.2, dec: 1, labelW: 215, xlabel: "Hazard ratio (orange: interval touches 1, small groups)", aria: "Subgroup hazard ratios" });

    renderTable($("#pairTable"), [
      { h: "Arm A", f: r => r.arm_A }, { h: "Arm B", f: r => r.arm_B },
      { h: "Adjusted p", num: 1, f: r => r.p_adjusted },
    ], E.pairwise);
    renderTable($("#censTable"), [
      { h: "Arm", f: r => r.arm }, { h: "n", num: 1, f: r => r.n }, { h: "Events", num: 1, f: r => r.event },
      { h: "Censored", num: 1, f: r => fmt(r["censored_%"]) + "%" }, { h: "Median follow-up", num: 1, f: r => fmt(r.median_followup_days, 0) + " d" },
    ], E.censoring);
  }

  /* ---------------- MODELS ---------------- */
  const MODEL_SHORT = (n) => n.startsWith("Cox, treatment") ? "Cox (no treatment)" : n.replace("Random Survival Forest", "RSF").replace("Gradient Boosting", "GBM").replace("Cox (ridge)", "Cox (ridge)");
  function renderModels() {
    const t = state.modelTrial, M = D.models[t], isA = t === "actg";
    const labels = M.cv.map(r => MODEL_SHORT(r.model));
    const cv = M.cv.map(r => r.c), se = M.cv.map(r => r.se);
    const test = labels.map((_, i) => (M.test[i] ? M.test[i].C_harrell : null));
    const base = chartBase();
    const errBars = { id: "err", afterDatasetsDraw(c) {
      const meta = c.getDatasetMeta(0), y = c.scales.y, { ctx } = c; ctx.save(); ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.4;
      meta.data.forEach((b, i) => { const x = b.x, a = y.getPixelForValue(cv[i] - se[i]), z = y.getPixelForValue(cv[i] + se[i]);
        ctx.beginPath(); ctx.moveTo(x, a); ctx.lineTo(x, z); ctx.moveTo(x - 5, a); ctx.lineTo(x + 5, a); ctx.moveTo(x - 5, z); ctx.lineTo(x + 5, z); ctx.stroke(); });
      ctx.restore(); } };
    setChart("cChart", {
      type: "bar", plugins: [errBars],
      data: { labels, datasets: [
        { label: "Cross-validation (train)", data: cv, backgroundColor: css("--blue"), borderRadius: 6, maxBarThickness: 46 },
        { label: "Test set (opened once)", data: test, backgroundColor: css("--accent"), borderRadius: 6, maxBarThickness: 46 } ] },
      options: { ...base, scales: { x: { ...base.scales.x, ticks: { ...base.scales.x.ticks, maxRotation: 0, font: { size: 11.5 } } }, y: { ...base.scales.y, min: 0.5, max: 0.8, title: { display: true, text: "C-index", color: css("--muted") } } },
        plugins: { ...base.plugins, tooltip: { ...base.plugins.tooltip, callbacks: { label: c => c.parsed.y == null ? "" : ` ${c.dataset.label}: ${c.parsed.y.toFixed(3)}` } } } },
    });
    const coxT = M.cv[0].c, noT = M.cv[3].c;
    $("#modelNote").textContent = isA
      ? `Cox led in both cross-validation and test. Adding the treatment arm raised the cross-validated C-index from ${fmt(noT, 3)} to ${fmt(coxT, 3)}. Black bars: one standard error.`
      : `All three models are within each other's uncertainty (test intervals overlap almost completely), so the simplest model is preferred. Black bars: one standard error.`;
    $("#testSub").textContent = isA ? "428 patients, 104 events. Intervals are wide because the test set is small." : "138 patients, 60 events. Intervals are wide because the test set is small.";
    const aucKey = Object.keys(M.test[0]).find(k => k.startsWith("AUC"));
    const ciKey = isA ? "C_harrell_95%" : "C_95%";
    renderTable($("#testTable"), [
      { h: "Model", f: r => r.model },
      { h: "C-index (95% CI)", num: 1, f: r => `${fmt(r.C_harrell, 3)} (${r[ciKey].replace(" - ", " to ")})` },
      { h: aucKey.replace("_", " "), num: 1, f: r => fmt(r[aucKey], 3) },
      { h: "IBS skill", num: 1, f: r => fmt(r.IBS_skill, 3) },
    ], M.test, { pick: r => r.model === "Cox (ridge)" });

    const ch = D.models.actg.chosen;
    const chosenSet = new Set([`Cox|alpha=${ch.Cox.alpha}`, `RSF|min_samples_leaf=${ch.RSF.min_samples_leaf}, max_features=${ch.RSF.max_features}`, `GBM|n_estimators=${ch.GBM.n_estimators}, max_depth=${ch.GBM.max_depth}`]);
    renderTable($("#gridTable"), [
      { h: "Model", f: r => r.model }, { h: "Settings", f: r => r.settings },
      { h: "C-index", num: 1, f: r => fmt(r.C_harrell, 4) }, { h: "SE", num: 1, f: r => fmt(r.SE, 4) },
      { h: "Uno C", num: 1, f: r => fmt(r.C_uno, 4) }, { h: "IBS skill", num: 1, f: r => fmt(r.IBS_skill, 4) },
    ], D.models.actg.grid, { pick: r => chosenSet.has(`${r.model}|${r.settings}`) });
  }

  /* ---------------- HETEROGENEITY ---------------- */
  function renderCate() {
    const arm = $("#cateArm").value, mc = METHOD_COLOR();
    const rows = D.hetero.actg.cate_summary.filter(r => r.arm === arm).map(r => ({
      label: r.method, sub: `mean ${fmt(r.mean)} d  |  ${fmt(r["share_positive_%"], 0)}% positive`, est: r.mean, lo: r.p10, hi: r.p90, color: mc[r.method], tip: `${r.method}: mean ${r.mean} days` }));
    forest($("#cateRange"), rows, { ref: 0, min: -60, max: 180, dec: 0, rowH: 56, labelW: 175, xlabel: "Estimated days gained vs ZDV (900-day RMST)", aria: "Patient-level effect ranges" });
  }
  function renderVal() {
    const mc = METHOD_COLOR(), s = state.valSet; let rows = [];
    if (s === "gbsg_test") {
      D.hetero.gbsg.validation.filter(r => r.set === "test").forEach(r => rows.push({ label: r.method, sub: `gain ${fmt(r["gain_%"], 2)}%  |  slope ${fmt(r.slope, 2)}`, est: r["gain_%"], lo: r.gain_lo, hi: r.gain_hi, color: mc[r.method] }));
    } else {
      ["ZDV+ddI", "ZDV+zal", "ddI"].forEach(arm => {
        rows.push({ header: arm + " vs ZDV" });
        D.hetero.actg.validation.filter(r => r.set === s && r.arm === arm).forEach(r => rows.push({ label: r.method, sub: `gain ${fmt(r["gain_%"], 2)}%  |  slope ${fmt(r.slope, 2)}  |  Holm p ${fmt(r.p_holm, 3)}`, est: r["gain_%"], lo: r.gain_lo, hi: r.gain_hi, color: mc[r.method] }));
      });
    }
    forest($("#valForest"), rows, { ref: 0, dec: 0, labelW: 205, xlabel: "Gain over constant-effect baseline (%), bootstrap 95% interval", aria: "Validation gain" });
  }

  /* ---------------- PREDICTOR ---------------- */
  const yesNo = [[0, "No"], [1, "Yes"]];
  const FORMS = {
    actg175: {
      url: "/api/predict/actg175", title: "ACTG 175 prediction",
      fields: [
        { g: "Demographics" },
        { k: "age", l: "Age", u: "years", t: "n", v: 35, min: 12, max: 90, step: 1 },
        { k: "wtkg", l: "Weight", u: "kg", t: "n", v: 75, min: 30, max: 170, step: 0.5 },
        { k: "gender", l: "Sex", t: "s", v: 1, o: [[0, "Female"], [1, "Male"]] },
        { k: "race", l: "Race", t: "s", v: 0, o: [[0, "White"], [1, "Non-white"]] },
        { g: "Laboratory and clinical" },
        { k: "cd40", l: "CD4 count", u: "cells/mm3", t: "n", v: 340, min: 0, max: 1500, step: 1 },
        { k: "cd80", l: "CD8 count", u: "cells/mm3", t: "n", v: 890, min: 0, max: 6000, step: 1 },
        { k: "karnof", l: "Karnofsky score", t: "s", v: 90, o: [[100, "100"], [90, "90"], [80, "80"], [70, "70"]] },
        { k: "symptom", l: "Symptomatic HIV", t: "s", v: 0, o: yesNo },
        { g: "History" },
        { k: "preanti", l: "Prior antiretroviral therapy", u: "days", t: "n", v: 140, min: 0, max: 3500, step: 1 },
        { k: "z30", l: "ZDV in prior 30 days", t: "s", v: 0, o: yesNo },
        { k: "homo", l: "Homosexual activity", t: "s", v: 0, o: yesNo },
        { k: "drugs", l: "IV drug use", t: "s", v: 0, o: yesNo },
        { k: "hemo", l: "Haemophilia", t: "s", v: 0, o: yesNo },
      ],
    },
    gbsg2: {
      url: "/api/predict/gbsg2", title: "GBSG2 prediction",
      fields: [
        { g: "Patient" },
        { k: "age", l: "Age", u: "years", t: "n", v: 55, min: 18, max: 90, step: 1 },
        { k: "post", l: "Menopausal status", t: "s", v: 1, o: [[0, "Premenopausal"], [1, "Postmenopausal"]] },
        { g: "Tumour" },
        { k: "tsize", l: "Tumour size", u: "mm", t: "n", v: 25, min: 1, max: 150, step: 1 },
        { k: "grade", l: "Tumour grade", t: "s", v: 2, o: [[1, "Grade 1"], [2, "Grade 2"], [3, "Grade 3"]] },
        { k: "pnodes", l: "Positive lymph nodes", t: "n", v: 3, min: 0, max: 60, step: 1 },
        { g: "Receptors" },
        { k: "progrec", l: "Progesterone receptor", u: "fmol", t: "n", v: 30, min: 0, max: 3000, step: 1 },
        { k: "estrec", l: "Oestrogen receptor", u: "fmol", t: "n", v: 35, min: 0, max: 1500, step: 1 },
      ],
    },
  };
  function buildForm() {
    const cfg = FORMS[state.predTrial];
    $("#predForm").innerHTML = cfg.fields.map(f => {
      if (f.g) return `<div class="group-title">${f.g}</div>`;
      const label = `<label for="f_${f.k}">${f.l}${f.u ? `<span>${f.u}</span>` : ""}</label>`;
      const input = f.t === "n"
        ? `<input id="f_${f.k}" name="${f.k}" type="number" inputmode="decimal" min="${f.min}" max="${f.max}" step="${f.step}" value="${f.v}" required>`
        : `<select id="f_${f.k}" name="${f.k}">${f.o.map(([v, t]) => `<option value="${v}" ${v === f.v ? "selected" : ""}>${t}</option>`).join("")}</select>`;
      return `<div class="field">${label}${input}</div>`;
    }).join("");
  }
  async function predict(ev) {
    ev.preventDefault();
    const cfg = FORMS[state.predTrial], body = {};
    cfg.fields.filter(f => f.k).forEach(f => { body[f.k] = Number($("#f_" + f.k).value); });
    const btn = $("#predBtn"); btn.disabled = true; btn.textContent = "Predicting";
    try {
      const res = await api(cfg.url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      showResult(res);
    } catch (e) {
      toast("Please check the values: some are outside the allowed range.");
    } finally { btn.disabled = false; btn.textContent = "Predict"; }
  }
  let lastResult = null;
  function showResult(res) {
    lastResult = res;
    $("#resultEmpty").hidden = true; $("#resultBody").hidden = false;
    const colors = ARM_COLOR(), key = Object.keys(res.arms)[0] === "ZDV" ? "difference_vs_ZDV_pts" : "difference_vs_no_therapy_pts";
    const refName = Object.keys(res.arms)[0];
    $("#resTitle").textContent = `Predicted ${res.trial === "ACTG 175" ? "event-free" : "recurrence-free"} probability at ${res.horizon_days} days`;
    $("#resSub").textContent = `Outcome: ${res.outcome}. Ridge Cox model, trained on the training split.`;
    $("#armCards").innerHTML = Object.entries(res.arms).map(([name, a]) => {
      const d = a[key], isRef = name === refName;
      const sub = isRef ? "Reference" : `<span class="${d >= 0 ? "pos" : "neg"}">${d >= 0 ? "+" : ""}${fmt(d)} pts vs ${refName === "ZDV" ? "ZDV" : "no therapy"}</span>`;
      return `<div class="arm-card" style="--c:${colors[name]}"><small>${name}</small><strong>${fmt(a.event_free_prob * 100)}%</strong>${isRef ? `<span>${sub}</span>` : sub}</div>`;
    }).join("");
    const base = chartBase();
    setChart("predChart", {
      type: "line",
      data: { labels: res.times.map(t => Math.round(t)), datasets: Object.entries(res.curves).map(([n, s]) => ({ label: n, data: s, borderColor: colors[n], backgroundColor: colors[n], stepped: true, pointRadius: 0, borderWidth: 2.4 })) },
      options: { ...base, interaction: { mode: "index", intersect: false },
        scales: { x: { ...base.scales.x, title: { display: true, text: "Days", color: css("--muted") }, ticks: { ...base.scales.x.ticks, maxTicksLimit: 8 } },
                  y: { ...base.scales.y, min: Math.max(0, Math.floor((Math.min(...Object.values(res.curves).flat()) - .05) * 10) / 10), max: 1, ticks: { ...base.scales.y.ticks, callback: v => Math.round(v * 100) + "%" } } },
        plugins: { ...base.plugins, tooltip: { ...base.plugins.tooltip, callbacks: { title: i => "Day " + i[0].label, label: c => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}%` } } } },
    });
    $("#warnBox").innerHTML = (res.warnings || []).map(w => `<div>${esc(w)}</div>`).join("");
  }

  /* ---------------- FIGURES ---------------- */
  const FIG_META = {
    eda_1_outcome: ["Outcome and follow-up", "EDA"], eda_2a_continuous: ["Continuous feature distributions", "EDA"], eda_2b_binary_categorical: ["Binary and categorical features", "EDA"],
    eda_3_useless_features: ["Constant and redundant features", "EDA"], eda_4_baseline_balance: ["Baseline balance between arms", "EDA"], eda_5_censoring_by_arm: ["Censoring by arm", "EDA"],
    eda_6a_correlation: ["Feature correlation", "EDA"], eda_6b_cramers_v: ["Association between categorical features", "EDA"], eda_7a_km_arms: ["Kaplan-Meier by arm", "EDA"],
    eda_7b_km_by_feature: ["Kaplan-Meier by feature", "EDA"], eda_8_event_rate_binary: ["Event rate by binary feature", "EDA"], eda_9a_km_treat_subgroups: ["Treatment effect in subgroups (curves)", "EDA"],
    eda_9b_forest_treatment: ["Treatment effect in subgroups (forest)", "EDA"], eda_10_patterns: ["Risk patterns across feature deciles", "EDA"], eda_11_summary: ["EDA one-page summary", "EDA"],
    cleaning_1_linearity: ["Linearity check (martingale residuals)", "EDA"], cleaning_2_outliers: ["Outlier check", "EDA"],
    effects_1_summary: ["Treatment effect summary", "Effects"], models_1_comparison: ["Model comparison", "Models"], cate_1_overview: ["Patient-level effect estimates", "Heterogeneity"],
    validation_1_summary: ["Validation of patient-level effects", "Heterogeneity"], gbsg2_1_data_effect: ["GBSG2: data and effect", "GBSG2"], gbsg2_2_models_cate_validation: ["GBSG2: models, effects, validation", "GBSG2"],
  };
  function renderFigures() {
    const groups = ["All", "EDA", "Effects", "Models", "Heterogeneity", "GBSG2"]; let cur = "All";
    const items = D.figures.map(f => { const k = f.replace(".png", ""), m = FIG_META[k] || [k.replace(/_/g, " "), "EDA"]; return { file: f, title: m[0], group: m[1] }; });
    const draw = () => {
      $("#figChips").innerHTML = groups.map(g => `<button class="chip ${g === cur ? "on" : ""}" data-g="${g}">${g}</button>`).join("");
      $("#figGrid").innerHTML = items.filter(i => cur === "All" || i.group === cur).map(i =>
        `<figure class="fig" data-src="/figures/${i.file}" data-cap="${esc(i.title)}"><div class="thumb"><img loading="lazy" src="/figures/${i.file}" alt="${esc(i.title)}"></div><figcaption>${esc(i.title)}<small>${i.group}</small></figcaption></figure>`).join("");
    };
    $("#figChips").onclick = e => { const b = e.target.closest(".chip"); if (b) { cur = b.dataset.g; draw(); } };
    $("#figGrid").onclick = e => { const f = e.target.closest(".fig"); if (!f) return; $("#lbImg").src = f.dataset.src; $("#lbCap").textContent = f.dataset.cap; $("#lightbox").hidden = false; };
    const close = () => { $("#lightbox").hidden = true; };
    $("#lightbox").onclick = e => { if (e.target.id !== "lbImg") close(); };
    document.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
    draw();
  }

  /* ---------------- wiring ---------------- */
  function seg(id, key, attr, cb) {
    $(id).addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      $$("button", $(id)).forEach(x => x.classList.toggle("on", x === b));
      state[key] = b.dataset[attr]; cb();
    });
  }
  async function init() {
    initChrome(); initConfig();
    try {
      const [overview, km, effects, models, hetero, figures] = await Promise.all(
        ["overview", "km", "effects", "models", "heterogeneity", "figures"].map(n => api("/api/" + n)));
      Object.assign(D, { overview, km, effects, models, hetero, figures });
    } catch (e) { toast("Could not load analysis results from the server."); return; }

    Chart.defaults.font.family = '-apple-system, "Segoe UI", Inter, Roboto, sans-serif';
    renderKpis(); renderKM(); renderEffects(); renderModels(); renderCate(); renderVal(); buildForm(); renderFigures();
    renderers.push(renderKM, renderEffects, renderModels, renderCate, renderVal, () => lastResult && showResult(lastResult));

    seg("#kmSeg", "kmTrial", "trial", renderKM);
    seg("#modelSeg", "modelTrial", "trial", renderModels);
    seg("#valSeg", "valSet", "set", renderVal);
    seg("#predSeg", "predTrial", "trial", () => { buildForm(); $("#resultEmpty").hidden = false; $("#resultBody").hidden = true; lastResult = null; });
    $("#cateArm").addEventListener("change", renderCate);
    $("#predForm").addEventListener("submit", predict);
    $("#resetBtn").addEventListener("click", buildForm);
  }
  init();
})();
