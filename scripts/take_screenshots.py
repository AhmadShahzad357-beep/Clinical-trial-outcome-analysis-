from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://clinical-trial-outcome-analysis.vercel.app"
OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

SHOTS = [
    ("dashboard_overview.png", "#overview"),
    ("dashboard_treatment_effect.png", "#effects"),
    ("dashboard_models.png", "#models"),
    ("dashboard_heterogeneity.png", "#hetero"),
    ("dashboard_predictor.png", "#predict"),
]

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1500, "height": 820})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(2000)
    for name, hash_ in SHOTS:
        page.evaluate(f"document.querySelector('{hash_}').scrollIntoView()")
        if hash_ == "#predict":
            page.click("#predBtn")
            page.wait_for_timeout(1000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / name))
        print("saved", name)
    b.close()