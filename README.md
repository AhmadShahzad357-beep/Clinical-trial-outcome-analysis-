# Clinical Trial Outcome Analysis

<p align="center">
  <a href="https://clinical-trial-outcome-analysis.vercel.app"><img alt="Live demo" src="https://img.shields.io/badge/%E2%96%B6%20Live%20demo-open%20the%20dashboard-0f766e?style=for-the-badge"></a>
</p>

## 🔭 Overview

Clinical trials usually report one number: the **average** effect of a treatment. This project goes further, on
**time-to-event (survival) data with censoring**, and answers four connected questions with one consistent protocol:

1. **Does the treatment work?** Kaplan-Meier curves, Cox hazard ratios, restricted mean survival time (RMST), NNT and
   covariate-adjusted estimates.
2. **Can we predict who will do well?** Cox regression versus Random Survival Forest versus Gradient Boosting,
   compared with cross-validation and a test set that is opened exactly once.
3. **Does the benefit differ from patient to patient?** Patient-level effect estimates (CATE) from five learners.
4. **Is that difference real or just noise?** A dedicated validation stage with a constant-effect baseline,
   calibration slope, global-null permutation test and out-of-sample test.

The same protocol is run on two trials, **ACTG 175** (HIV, four treatment arms) and **GBSG2** (breast cancer,
hormonal therapy), so conclusions are not tied to a single dataset.

The results are served through an interactive **FastAPI dashboard** that also contains a patient-level **survival prediction tool**.

---

## 🎯 Aim and research questions

**Aim:** build and document an end-to-end, leakage-safe analysis that separates what the data can truly support
(average benefit, outcome prediction) from what it cannot yet support (patient-specific benefit), and show that it
behaves consistently across two trials.

| # | Research question |
|---|---|
| RQ1 | How large is the treatment effect, in hazard ratio, days gained (RMST) and absolute risk? |
| RQ2 | Does machine learning predict outcomes better than a regularized Cox model? |
| RQ3 | Do estimated patient-level effects (CATE) capture real heterogeneity? |
| RQ4 | Do the conclusions replicate on a second trial with a different design? |

---

## ❓ Problem statement

| Challenge | Why it matters |
|---|---|
| **Average effect hides individuals** | A drug that helps on average may help some patients much more, or not at all. Clinicians want to know *who* benefits. |
| **Censoring** | Many patients leave follow-up before an event happens. Ordinary regression and plain percentages are biased on such data. |
| **Data leakage** | Using variables measured after randomization, or fitting preprocessing on the full dataset, quietly inflates results. |
| **False discoveries in "personalized medicine"** | Flexible ML models can find patient-level differences in pure noise, and many reports skip honest validation. |
| **Trials are not always cleanly randomized** | In GBSG2 the treated and untreated groups differ at baseline, so a naive comparison would mislead. |

---

## ✨ Key features

- **Leakage-safe by construction:** the train/test split comes first; scaling, encoding and model settings are
  learned on training data only; post-randomization variables are removed.
- **Censoring-aware throughout:** Kaplan-Meier, Cox, RMST, IPCW outcomes, integrated Brier score, Uno's C-index.
- **Five heterogeneity learners:** S-Cox, T-Cox, T-RSF and two doubly-robust learners (DR-Ridge, DR-RF), all
  out-of-fold (cross-fitted).
- **Honest validation:** constant-effect baseline, calibration slope with robust intervals, tertile calibration,
  a global-null permutation test, and a test set opened once.
- **Confounding-aware:** propensity-score (AIPW) analysis for the non-randomized GBSG2 comparison.
- **Simplicity preferred:** among settings within one standard error of the best, the simplest is chosen.
- **Readable output:** every script prints plain-language explanations and saves figures and tables.
- **Single-command stages:** each stage is one script with a `--fast` mode for quick checks.
- **Prediction system:** two ridge Cox models (one per trial) turn baseline measurements into the predicted probability of staying event-free at the trial horizon, under each treatment option.
- **FastAPI dashboard:** results, interactive charts, a patient predictor and a figure gallery in one professional web interface, backed by a documented JSON API.

---

## 🏗 Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Animated pipeline architecture" width="100%">
</p>

> The test data is opened once, after every choice is fixed on the training split.


---

## 🖥 Dashboard

<p align="center">
  <img src="docs/images/dashboard_overview.png" alt="Overview page with headline numbers, Kaplan-Meier curves and key findings" width="90%">
</p>
<p align="center">
  <img src="docs/images/dashboard_treatment_effect.png" alt="Treatment effect page: days gained, event-free percentage and effect size table" width="90%">
</p>
<p align="center">
  <img src="docs/images/dashboard_heterogeneity.png" alt="Heterogeneity page: patient-level effect estimates and validation" width="90%">
</p>
<p align="center">
  <img src="docs/images/dashboard_models.png" alt="Prediction models page: C-index for Cox, Random Survival Forest and Gradient Boosting" width="90%">
</p>
<p align="center">
  <img src="docs/images/dashboard_predictor.png" alt="Patient predictor with predicted event-free curves" width="90%">
</p>

---

## 💡 Benefits

- **For researchers and biostatisticians:** a transparent, reusable template for survival analysis with honest
  evaluation, ready to run on any two-group or multi-arm trial with a time-to-event outcome.
- **For clinical machine-learning work:** a concrete demonstration that a flexible model is not automatically better
  than a regularized Cox model, and a method for checking whether "personalized" effects are real.
- **For decision making:** results are reported in units clinicians understand (event-free percentage, days gained,
  NNT), with intervals, not only hazard ratios.
- **For reviewers and collaborators:** every choice is documented, every script is runnable, and every figure and
  table can be regenerated.
- **For avoiding false claims:** the validation stage protects against reporting patient-level differences that are
  only noise.

---

## ✅ Conclusion

On two real trials, the pipeline gives a consistent picture. **The treatments help on average**: all three ACTG 175
combination arms cut the hazard of the primary event roughly in half compared with ZDV alone, and hormonal therapy in
GBSG2 lowers it by about a third. **Prediction is a solved-enough problem for simple models**: a regularized Cox model
matched or beat random forests and boosting. **Patient-specific benefit is not yet supported**: after a constant-effect
baseline, calibration slope, global-null test and a one-time test evaluation, no patient-level difference could be
confirmed in either trial.

The practical message is that, with data of this size, reporting a solid average effect with honest uncertainty is
better than claiming personalization the data cannot support. The same code can be pointed at larger trials, where
heterogeneity can be tested with more power.