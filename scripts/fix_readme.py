from pathlib import Path

p = Path("README.md")
s = p.read_text(encoding="utf-8").replace("\r\n", "\n")

# 1. subgroups screenshot -> heterogeneity screenshot
s = s.replace("docs/images/dashboard_subgroups.png", "docs/images/dashboard_heterogeneity.png")
s = s.replace("Subgroup forest plot, pairwise log-rank tests and censoring table",
              "Heterogeneity page: patient-level effect estimates and validation")

# 2. add the patient predictor screenshot after the models screenshot
marker = 'docs/images/dashboard_models.png'
if "dashboard_predictor.png" not in s:
    start = s.index(marker)
    end = s.index("</p>", start) + len("</p>")
    block = ('\n<p align="center">\n'
             '  <img src="docs/images/dashboard_predictor.png" '
             'alt="Patient predictor with predicted event-free curves" width="90%">\n'
             '</p>')
    s = s[:end] + block + s[end:]

p.write_text(s, encoding="utf-8")
print("README updated")