# Next steps (handoff)

State at hand-off: all of Josh's experiment tasks (E1 to E7) are done, apart
from the E6 run on credentialed MIMIC-IV. Read `docs/JOSH_E1_E7_NOTES.md`
first; `docs/E6_MIMIC.md` describes the E6 design. Work is on branch
`codex/josh-e2-e5` (PR tucana1/drift-attribution#1).

## 1. Run E6 on credentialed MIMIC-IV (required; closes task E6)

**Where.** Your own machine, with PhysioNet credentialed access to MIMIC-IV.

**Data handling (hard rule).** MIMIC-IV rows must not leave your machine or
reach any hosted AI service (PhysioNet credentialed data use agreement and
its guidance on online services). If an AI coding agent helps with this step,
it may run the scripts and read what they print (counts, rates, aggregate
estimates) and the aggregate JSON they write. It must not open anything under
`data/` or the MIMIC-IV CSVs, and must not print rows. The two E6 scripts only
print and save aggregates.

### 1.1 Environment

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-e6.txt
```

### 1.2 Smoke test on the open MIMIC-IV demo (optional, recommended)

The demo (100 patients, open access, no credentials) has the real table
layout, so it catches schema problems before the large download. Results are
meaningless at this size.

```sh
wget -r -N -c -np -nH --cut-dirs=3 -P ~/physionet/mimic-iv-demo \
     https://physionet.org/files/mimic-iv-demo/2.2/
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root ~/physionet/mimic-iv-demo \
     --out data/e6_demo_cohort.npz
```

Expected: the extraction finishes and prints a few dozen to a few hundred
admissions. If it fails, fix `sim2/e6_mimic_cohort.py` (the `TYPES` table or
the ward-unit exclusions) before the full run. The experiment script will
refuse to write paper outputs for a cohort this small, and may stop for lack
of events; the demo is only for the extraction.

### 1.3 Download the full data (only the six tables needed)

```sh
for f in hosp/patients hosp/admissions hosp/transfers hosp/labevents hosp/d_labitems icu/icustays; do
  wget -N -c --user <physionet-username> --ask-password \
       -P ~/physionet/mimiciv/$(dirname $f) https://physionet.org/files/mimiciv/3.1/$f.csv.gz
done
```

`labevents` is the large file (a few GB compressed). v2.2 also works.

### 1.4 Build the cohort and check it

```sh
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root ~/physionet/mimiciv --out data/e6_cohort.npz
```

Stop and investigate before step 1.5 if any of these fails:

- admissions: tens of thousands or more (the size guard in step 1.5 needs at least 5,000);
- event rate (ICU admission or in-hospital death within 12 h of a ward
  creatinine draw): roughly 1% to 10%; outside 0.5% to 15%, check the ward
  definition in `EXCLUDED_UNITS`;
- every `anchor_year_group` from 2008 - 2010 to 2017 - 2019 present
  (v3.x adds 2020 - 2022, excluded by default);
- missingness of the core labs below about 10%.

### 1.5 Run the experiment

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz
```

This writes `figures/merged_expE6.json`, `figures/fig7_mimic.pdf` and their
copies in `aaai/figures/`. Allow 15 to 30 minutes.

### 1.6 Sensitivity runs (recommended, appendix)

Same command with a tagged `--output figures/merged_expE6_<tag>.json` and
`--figure figures/fig7_mimic_<tag>.pdf`:

- treatment effect `--rrr 0.25` and `--rrr 0.75`;
- alert rate `--alert-rate 0.05` and `--alert-rate 0.20`;
- COVID-era drift: `--late "2020 - 2022"` (v3.x only).

### 1.7 Update the manuscript

- F7 (`\paragraph{F7: semi-synthetic MIMIC-IV`): replace the `\todo` with, from
  `merged_expE6.json`, the policy target and the proposed estimate with its
  interval in `deploy_only` and `drift_and_deploy`, the monitor's P(Y|X)
  term, the union-graph estimate (`deploy_only`), the observed AUROC change,
  and round-8 events averted for keep, naive, refit on A=0 and weighted refit
  under the fixed threshold. Check the sign of the policy term before writing
  "drop" or "improvement" (see section 2).
- Appendix "Semi-synthetic MIMIC-IV experiment (E6)": remove the `\todo`; add
  cohort size, event rate by period and the score's AUROC (all under
  `cohort` in the JSON).
- Add quoted-number checks for F7 to `check_e6` in
  `sim2/validate_josh_results.py`, as `check_e1` does for F4.
- Regenerate and check:

```sh
.venv/bin/python sim2/make_figures2.py      # copies Figure 7 into aaai/figures/
.venv/bin/python sim2/make_tables.py
.venv/bin/python sim2/validate_josh_results.py
```

**Done when:** the validator passes with the E6 check active (not "skipped";
it rejects demo-sized or fixture cohorts and an open F7 TODO), Figure 7
renders in the compiled draft, and only aggregate files were committed
(`figures/merged_expE6*.json`, `figures/fig7_mimic*.pdf`, their mirrors and
the `.tex`); `data/` stays untracked.

## 2. Decisions for the team that this branch surfaces

- **Prevalence** (appendix Table 5): at realistic ward prevalence, successful
  alerting lowers the Brier score, and the monitor books that improvement to
  the outcome mechanism. The introduction's "performance drops" premise (W1)
  and the results framing (W7) should say this. Options: state the
  prevalence dependence; add a realistic-prevalence operating point to
  Experiment A; or also attribute a metric that falls at every prevalence
  (AUROC does).
- **E1 headline rule:** the to-do's rule (refit when the exogenous share
  exceeds one half) refits needlessly under pure covariate shift; the
  outcome-mechanism rule has zero regret in S1 to S4. Decide which the paper
  leads with.
- **Logging requirements** (Section 5): add alert exposure and the
  clinician-response probabilities, which the weighted refit of E3 needs.

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

Synthetic Hospital (Park, Chen, Dettmers, 2026): see
`docs/SYNTHETIC_HOSPITAL_ASSESSMENT.md`.

## 5. Optional extensions

- **eICU-CRD** (credentialed, 208 hospitals): real multi-site data with
  hospital identifiers, to ground the site-level findings of E5.
- **COVID-era drift** in E6 (section 1.6).
- **GHOSTS** synthetic ICU time series as an open fallback testbed, after
  checking the corpus licence.
