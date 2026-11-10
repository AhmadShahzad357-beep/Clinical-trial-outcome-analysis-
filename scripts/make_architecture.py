"""Generate docs/architecture.svg (animated: flowing connectors, travelling dots, staggered fade-in, pulsing output).
Run:  python scripts/make_architecture.py"""
from pathlib import Path

W, H = 1420, 610
NW, NH = 170, 80
COLS = [30, 260, 490, 720, 950]
MAIN, TOP, BOT = 245, 95, 395
DASH_X, DASH_W, DASH_Y, DASH_H = 1180, 210, 110, 350

nodes = [  # id, x, y, step, title, file, desc
    ("raw", COLS[0], MAIN, "1", "Raw data", "ACTG 175 · GBSG2", "Two public trials"),
    ("clean", COLS[1], MAIN, "2", "Clean and audit", "cleaning.py", "Leakage columns removed"),
    ("eda", COLS[1], TOP, "3", "EDA", "eda.py", "Balance, censoring, KM"),
    ("feat", COLS[2], MAIN, "4", "Feature engineering", "features.py", "Train / test split first"),
    ("eff", COLS[3], TOP, "5", "Treatment effect", "effects.py", "HR · RMST · NNT"),
    ("mod", COLS[3], MAIN, "6", "Prediction models", "models.py", "Cox · RSF · GBM"),
    ("cate", COLS[3], BOT, "7", "Heterogeneity", "cate.py", "5 CATE learners"),
    ("pred", COLS[4], MAIN, "8", "Patient predictor", "predictor.py", "Ridge Cox per trial"),
    ("val", COLS[4], BOT, "9", "Validation", "validation.py", "Calibration · global null"),
]
C = {n[0]: n for n in nodes}
def R(i): return (C[i][1] + NW, C[i][2] + NH / 2)
def L(i): return (C[i][1], C[i][2] + NH / 2)
def T(i): return (C[i][1] + NW / 2, C[i][2])
def B(i): return (C[i][1] + NW / 2, C[i][2] + NH)
dl = lambda y: (DASH_X, y)

def curve(a, b):
    (x1, y1), (x2, y2) = a, b
    if abs(x1 - x2) < 1:                         # vertical
        return f"M{x1:.0f} {y1:.0f} L{x2:.0f} {y2:.0f}"
    dx = (x2 - x1) * 0.5
    return f"M{x1:.0f} {y1:.0f} C{x1+dx:.0f} {y1:.0f} {x2-dx:.0f} {y2:.0f} {x2:.0f} {y2:.0f}"

edges = [
    ("raw", R("raw"), L("clean")), ("clean", T("clean"), B("eda")), ("c2f", R("clean"), L("feat")),
    ("f2e", R("feat"), L("eff")), ("f2m", R("feat"), L("mod")), ("f2c", R("feat"), L("cate")),
    ("m2p", R("mod"), L("pred")), ("c2v", R("cate"), L("val")),
    ("e2d", R("eff"), dl(170)), ("p2d", R("pred"), dl(285)), ("v2d", R("val"), dl(400)),
]

o = []
a = o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d" font-family="Segoe UI, Inter, Helvetica, Arial, sans-serif">')
a('<title id="t">Pipeline architecture</title><desc id="d">Raw data is cleaned, split into train and test, analysed for treatment effect, prediction and heterogeneity, validated, and served by a dashboard. The same protocol is replicated on GBSG2.</desc>')
a("""<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1526"/><stop offset="1" stop-color="#10243a"/></linearGradient>
<linearGradient id="acc" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#2dd4bf"/><stop offset="1" stop-color="#38bdf8"/></linearGradient>
<linearGradient id="dash" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0f766e"/><stop offset="1" stop-color="#155e75"/></linearGradient>
<filter id="sh" x="-20%" y="-20%" width="140%" height="150%"><feDropShadow dx="0" dy="6" stdDeviation="8" flood-color="#000" flood-opacity=".35"/></filter>
<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 1L9 5L0 9z" fill="#5eead4"/></marker>
<style>
.node{animation:in .7s ease-out backwards}
.edge{fill:none;stroke:#5eead4;stroke-width:2;stroke-opacity:.85;stroke-dasharray:7 7;animation:flow 1.1s linear infinite}
.base{fill:none;stroke:#1e3a52;stroke-width:5;stroke-linecap:round}
.pulse{animation:pulse 2.6s ease-in-out infinite;transform-box:fill-box;transform-origin:center}
.ring{animation:ring 2.6s ease-out infinite;transform-box:fill-box;transform-origin:center}
.band{stroke-dasharray:6 6;animation:flow 2.4s linear infinite}
@keyframes in{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes flow{to{stroke-dashoffset:-14}}
@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.025)}}
@keyframes ring{0%{opacity:.55;transform:scale(1)}100%{opacity:0;transform:scale(1.08)}}
@media (prefers-reduced-motion:reduce){.node{animation:none}.edge,.band,.pulse,.ring{animation:none}}
</style>
</defs>""")
a(f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>')
a(f'<rect x=".5" y=".5" width="{W-1}" height="{H-1}" rx="18" fill="none" stroke="#24415c"/>')
a('<text x="36" y="44" font-size="22" font-weight="700" fill="#f1f5f9">Pipeline architecture</text>')
a('<text x="36" y="66" font-size="13" fill="#94a3b8">One leakage-safe protocol · two trials · results served by a dashboard</text>')
a(f'<rect x="{W-246}" y="26" width="210" height="30" rx="15" fill="#0f2a3d" stroke="#2dd4bf" stroke-opacity=".5"/>')
a(f'<g transform="translate({W-232},33)" fill="none" stroke="#5eead4" stroke-width="1.8"><rect x="0" y="6" width="12" height="9" rx="2"/><path d="M3 6V4a3 3 0 016 0v2"/></g>')
a(f'<text x="{W-207}" y="46" font-size="12.5" font-weight="600" fill="#5eead4">Test set opened once</text>')

heads = ["DATA", "PREPARE", "SPLIT", "ANALYSE", "PREDICT · VALIDATE"]
for i, h in enumerate(heads):
    a(f'<text x="{COLS[i]+NW/2}" y="86" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="#64829e">{h}</text>')
a(f'<text x="{DASH_X+DASH_W/2}" y="86" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.6" fill="#64829e">DELIVER</text>')

# connectors: base track + animated dashes + travelling dot
for i, (eid, p1, p2) in enumerate(edges):
    d = curve(p1, p2)
    a(f'<path id="p_{eid}" class="base" d="{d}"/>')
    a(f'<path class="edge" d="{d}" marker-end="url(#ar)" style="animation-delay:{i*0.13:.2f}s"/>')
    a(f'<circle r="4.2" fill="#fff" filter="url(#glow)"><animateMotion dur="{2.6+(i%3)*0.4:.1f}s" begin="{0.9+i*0.35:.2f}s" repeatCount="indefinite" rotate="auto"><mpath href="#p_{eid}"/></animateMotion></circle>')

# nodes
for i, (nid, x, y, step, title, file, desc) in enumerate(nodes):
    a(f'<g class="node" style="animation-delay:{0.1+i*0.12:.2f}s" filter="url(#sh)">')
    a(f'<rect x="{x}" y="{y}" width="{NW}" height="{NH}" rx="12" fill="#12263b" stroke="#2c4d6b"/>')
    a(f'<rect x="{x}" y="{y}" width="5" height="{NH}" rx="2.5" fill="url(#acc)"/>')
    a(f'<circle cx="{x+27}" cy="{y+26}" r="12" fill="#0f766e"/><text x="{x+27}" y="{y+31}" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">{step}</text>')
    a(f'<text x="{x+47}" y="{y+30}" font-size="13" font-weight="700" fill="#f1f5f9">{title}</text>')
    a(f'<text x="{x+18}" y="{y+52}" font-size="12" fill="#5eead4" font-family="Consolas, Menlo, monospace">{file}</text>')
    a(f'<text x="{x+18}" y="{y+69}" font-size="11.5" fill="#94a3b8">{desc}</text>')
    a('</g>')

# dashboard
a('<g class="node" style="animation-delay:1.4s" filter="url(#sh)">')
a(f'<rect class="ring" x="{DASH_X}" y="{DASH_Y}" width="{DASH_W}" height="{DASH_H}" rx="16" fill="none" stroke="#2dd4bf" stroke-width="2"/>')
a(f'<g class="pulse"><rect x="{DASH_X}" y="{DASH_Y}" width="{DASH_W}" height="{DASH_H}" rx="16" fill="url(#dash)" stroke="#5eead4" stroke-opacity=".6"/></g>')
cx = DASH_X + DASH_W / 2
a(f'<text x="{cx}" y="{DASH_Y+44}" text-anchor="middle" font-size="18" font-weight="700" fill="#fff">Dashboard</text>')
a(f'<text x="{cx}" y="{DASH_Y+66}" text-anchor="middle" font-size="12" fill="#ccfbf1">6 pages · light and dark</text>')
a(f'<text x="{cx}" y="{DASH_Y+100}" text-anchor="middle" font-size="12" fill="#e2fbf6">Effects · Models</text>')
a(f'<text x="{cx}" y="{DASH_Y+120}" text-anchor="middle" font-size="12" fill="#e2fbf6">Heterogeneity · Predictor</text>')
a(f'<text x="{cx}" y="{DASH_Y+140}" text-anchor="middle" font-size="12" fill="#e2fbf6">Figure gallery</text>')
a(f'<rect x="{DASH_X+18}" y="{DASH_Y+210}" width="{DASH_W-36}" height="50" rx="10" fill="#0b1526" fill-opacity=".55"/>')
a(f'<text x="{cx}" y="{DASH_Y+232}" text-anchor="middle" font-size="13" font-weight="600" fill="#fff">FastAPI server</text>')
a(f'<text x="{cx}" y="{DASH_Y+250}" text-anchor="middle" font-size="11.5" fill="#99f6e4" font-family="Consolas, Menlo, monospace">app/main.py</text>')
a(f'<rect x="{DASH_X+18}" y="{DASH_Y+272}" width="{DASH_W-36}" height="50" rx="10" fill="#0b1526" fill-opacity=".55"/>')
a(f'<text x="{cx}" y="{DASH_Y+294}" text-anchor="middle" font-size="13" font-weight="600" fill="#fff">Static site · Vercel</text>')
a(f'<text x="{cx}" y="{DASH_Y+312}" text-anchor="middle" font-size="11.5" fill="#99f6e4" font-family="Consolas, Menlo, monospace">web/ · live demo</text>')
a('</g>')

# replication band
a('<g class="node" style="animation-delay:1.6s">')
a(f'<rect class="band" x="30" y="515" width="{W-60}" height="62" rx="14" fill="#0f2a3d" fill-opacity=".7" stroke="#38bdf8" stroke-opacity=".7"/>')
a('<circle cx="64" cy="546" r="12" fill="#0369a1"/><text x="64" y="551" text-anchor="middle" font-size="13" font-weight="700" fill="#fff">10</text>')
a('<text x="88" y="541" font-size="14" font-weight="700" fill="#f1f5f9">Replication on a second trial</text>')
a('<text x="88" y="561" font-size="12" fill="#94a3b8"><tspan fill="#7dd3fc" font-family="Consolas, Menlo, monospace">gbsg2.py</tspan>  runs the same protocol on GBSG2: propensity-adjusted effects, models, CATE and validation</text>')
a('</g>')
a('</svg>')
Path(__file__).resolve().parents[1].joinpath("docs", "architecture.svg").write_text("\n".join(o), encoding="utf-8")
print("docs/architecture.svg written")