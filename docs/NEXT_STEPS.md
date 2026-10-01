# Next steps (handoff)

State: E1 to E5 and E7 are done and were checked against the manuscript in a
second pass; the corrections are listed in `docs/JOSH_E1_E7_NOTES.md`. E6 is
ready to run: `sim2/run_e6.sh` goes from a credentialed MIMIC-IV download to
the paper numbers, and was tested end to end on the synthetic fixture. Read
`docs/JOSH_E1_E7_NOTES.md` first; `docs/E6_MIMIC.md` describes the E6 design.

## 1. Run E6 on credentialed MIMIC-IV (required; closes task E6)

**Where.** Your own machine, with PhysioNet credentialed access to MIMIC-IV.

**Data handling (hard rule).** MIMIC-IV rows must not leave your machine or
reach any hosted AI service (PhysioNet credentialed data use agreement and its
guidance on online services). Run the data steps yourself. An AI assistant
must not open anything under `data/` or `~/physionet`, or the cohort error
log; the scripts print and save aggregates only.

### 1.1 Smoke test on the open demo (optional, recommended)

```sh
sim2/run_e6.sh demo
```

Fetches the six demo tables (100 patients, ODbL, about 2 MB) into
`data/mimic-iv-demo` and runs the extraction with errors shown. Expected: it
finishes and prints a few dozen to a few hundred admissions; the acceptance
checks fail at this size and are not enforced here. If extraction fails, fix
`sim2/e6_mimic_cohort.py` (`TYPES`, `EXCLUDED_UNITS`) before the large
download.

### 1.2 Download

```sh
sim2/run_e6.sh download <physionet-username>
```

Six tables of MIMIC-IV v3.1 (about 3 GB, mostly `labevents`) into
`~/physionet/mimiciv/3.1`, password asked once, partial files resumed,
SHA-256 checked. Needs about 5 GB free. `MIMIC_VERSION=2.2` for v2.2,
`MIMIC_ROOT=...` for another location.

### 1.3 Cohort and checks

```sh
sim2/run_e6.sh cohort
```

Writes `data/e6_cohort.npz` and exits with status 3 if any acceptance check
fails:

- admissions: at least 5,000 (expect tens of thousands);
- event rate (ICU admission or in-hospital death within 12 h of a ward
  creatinine draw) within 0.5% to 15% (expect about 1% to 10%); outside it,
  check the ward definition in `EXCLUDED_UNITS`;
- every `anchor_year_group` from 2008 - 2010 to 2017 - 2019 present;
- missingness of each lab below 10%.

Also read the printed list of ward units: every unit in it should be a ward.

### 1.4 Main and sensitivity runs

```sh
sim2/run_e6.sh main          # 200 / 50 replicates; about 20 to 40 min with 4 workers
sim2/run_e6.sh sensitivity   # rrr 0.25, 0.75; alert rate 5%, 20%; periods by admission year; 2020 - 2022 (v3.x)
```

### 1.5 Report and manuscript

```sh
sim2/run_e6.sh report
```

Regenerates figures and tables (`aaai/tables/e6_mimic.tex` and
`e6_sensitivity.tex`, which the E6 appendix includes automatically), prints
the numbers F7 quotes and a draft of the F7 result sentences
(`sim2/e6_report.py`), and runs the validator, which fails until the
manuscript is updated:

- F7: replace `\todo{Results from \texttt{merged\_expE6.json}...}` with the
  result sentences. Check the wording against the sign of the policy term
  (section 2, prevalence); the draft chooses it from the sign.
- Appendix E6: replace `\todo{Numbers and Figure~\ref{fig:mimic}...}` with a
  sentence pointing to Tables `tab:mimic` and `tab:mimic-sens`.

**Done when:** the validator passes with the E6 check active (it rejects
fixture or demo-sized cohorts, open F7 or appendix TODOs, and F7 text without
the reported numbers), Figure 7 renders, and only the aggregate files printed
by the report step are committed; `data/` stays untracked.

## 2. Decisions for the team

- **Prevalence** (appendix Tables `tab:prevalence` and `tab:prevalence-est`).
  At a realistic event rate successful alerting lowers the Brier score and the
  monitor books the improvement to the outcome mechanism. The new
  low-prevalence run (6.6%, alert rate 20%) shows F1 and F4 hold with
  estimators, not only oracle values (see `JOSH_E1_E7_NOTES.md`). The
  introduction's "performance drops" premise (W1) and the results framing (W7)
  still need a decision: state the prevalence dependence, move the
  low-prevalence point into the main text, or also attribute a metric that
  falls at every prevalence (AUROC does).
- **E1 headline rule.** The task rule (refit when the exogenous share exceeds
  one half) refits needlessly under pure covariate shift. The outcome-share
  rule has zero regret in S1 to S4 at the calibrated point; at 6.6% prevalence
  it loses 0.55 points in S4, where the outcome share sits at 0.52, next to the
  cut-off. Refitting on untreated patients has zero worst-case regret at both
  operating points. Decide which rule the paper leads with.
- **Logging requirements** (Section 5): add alert exposure and either the
  clinician-response probabilities or enough to model them; with estimated
  propensities the weighted refit of E3 is within 0.01 points of the logged
  version.
- **Identification table, site stepped wedge, prospective contrast ("No").**
  In E5 the crossover difference in differences minus the DiD-anchored
  deployment-caused term recovers the prospective contrast within 1.4% in all
  four cells. The control change cancels, so this is a within-site,
  covariate-adjusted before/after comparison: it holds only without
  outcome-mechanism drift, which E5 does not simulate. Keep "No" or qualify it.
- **Threshold-discontinuity appendix** (`\todo` in Appendix A). The numbers are
  in `merged_expE7.json`, part E, including the policy-effect curve. The curve
  is hump-shaped and falls to zero at high scores, so condition (b) of
  Proposition 3 (monotone effect) does not hold in the simulation.

## 3. Reference checks (L2/L3)

Done. `liley2021model` and `johnson2023mimic` were correct. The holdout
paper's published title is "Holdout sets for safe predictive model updating"
(key now `haidarwehbe2025holdout`). The open TODOs in `references.bib` are
closed: Lenert 2019 and Sperrin 2019 volume and pages confirmed, arXiv
2601.00716 is Guan and Zhou, "Detecting Performance Degradation under Data
Shift in Pathology Vision-Language Model" (key `guan2026detecting`), the
Gonzalez et al. author list is complete (key `gonzalez2024regulating`, first
posted December 2024), and Keogh and van Geloven 2024 has volume and pages.

## 4. Assessed and not used

See `docs/SYNTHETIC_HOSPITAL_ASSESSMENT.md`: Synthetic Hospital (Park, Chen,
Dettmers, 2026); the trajectory generator of Zhou et al. (arXiv 2603.06720;
code CC BY-NC, no weights or data released); GHOSTS (code public, synthetic
corpus not released, ICU only). None removes the need for credentialed data.

## 5. Optional extensions

- **eICU-CRD** (credentialed, 208 hospitals): real multi-site data with
  hospital identifiers, to ground the site-level findings of E5. The E6
  pattern (local run, text-only reads, aggregate output) carries over.
- **AUROC attribution**: the third prevalence option above. `dgp.is_values`
  already takes `metric="auroc"`; `naive2_est` and `E3` in `merged.py` are
  written for the Brier score and would need weighted-AUROC versions.
