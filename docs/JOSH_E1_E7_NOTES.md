# Josh's E2-E5 work

## Earlier work used

- The repository already had the calibrated simulator in `sim2/`, Experiments A-F, and the AAAI draft. Those supply the operating point, scenarios, and Brier loss target used here.
- The September 15 `wmhs_submission_overleaf` package had a seven-rule, 100-seed, 50-round retraining study in `code/retraining_stress.py` and `evidence/retraining_stress/`. It used treatment-by-covariate interactions and an independent standard-care reference. We adapted that design for E3 with this repository's five rules, `kappa=2`, and a held alert rate. Its earlier numerical results are not used as results of the new run.
- The prior submission also established that the policy-last decomposition has no extra interaction residual. The draft equation now states the exact three-term identity.

## New work on this branch

| Task | Change |
|---|---|
| E2 | Added a main-text information-budget table. Oracle mechanism weights are labeled as simulation knowledge. |
| E3 | Added `sim2/exp_H_correction.py`, replaced `merged_expH.json`, and regenerated Figure 3 as two panels with one vertical scale and a standard-care zero. |
| E4 | Added `sim2/exp_AB_robustness.py` and an appendix table. Experiments A and B now run at `d=10,20` with linear and nonlinear outcome truth. |
| E5 | Added `sim2/exp_G2_sites.py` and `merged_expG2.json`. Site identities persist across six periods; rollout priority can follow site characteristics. Table 1 and the results text report the measured biases. |

## Main results

- E3, `kappa=2`, round 8: the interaction correction averts **8.146 percentage points** of events versus standard care; naive refitting averts **5.531**. The untreated and unalerted corrections avert **8.152** and **8.135**. The interaction correction does not rank above them in this run.
- E4: the no-action game still assigns nearly all of the S3 Brier change to the marginal outcome mechanism. At `eps=0`, union-graph policy-share bias remains about **-0.036 to -0.040** Brier units when sample size reaches 96,000.
- E5: with random rollout order, retrospective-term bias is **-0.00075** Brier units. With rollout linked to site characteristics and site-specific trends, it is **-0.00267**. The estimated deployment-caused covariate mean shift is **0.293** versus its true **0.250**; the estimated exogenous increment is **0.072** versus **0.114**. A separate location-shift reweighting of observed pre-rollout losses measures the Brier-scale X split; its biases are reported in the appendix and saved JSON.

## Reproduce

Run from the repository root with Python 3.10 or later:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sim2/exp_H_correction.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sim2/exp_AB_robustness.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python sim2/exp_G2_sites.py
.venv/bin/python sim2/make_figure3_E3.py
cp figures/merged_expH.json figures/merged_expAB_robustness.json figures/merged_expG2.json aaai/figures/
.venv/bin/python sim2/validate_josh_results.py
```

The root and `aaai/figures/` JSON copies are identical. Figure 3 is generated in both directories. Each E4 and E5 row uses an independent seed block; methods within a row share patients where pairing is intended. `sim2/validate_josh_results.py` checks the saved rows and their reported summaries.

These are synthetic results. The E3 action-based corrections assume recorded treatment and sufficient measured covariates. The E5 site comparison requires a credible comparison trend; the characteristic-linked scenarios show its bias when that condition fails. The Brier-scale X reweighting also assumes a Gaussian location shift and has finite-sample bias even with random rollout. The draft still has unrelated writing and proof TODOs outside E2-E5.
