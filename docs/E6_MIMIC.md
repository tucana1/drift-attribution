# E6: semi-synthetic MIMIC-IV experiment

Task E6 in the team to-do: real covariates and outcomes for a deterioration
task, a simulated alert policy and treatment effect on top, Experiments A and C
repeated. The code is complete and tested end to end on a synthetic stand-in.
The figure and numbers for the paper have to be produced by a team member with
credentialed MIMIC-IV access, on their own machine. `docs/NEXT_STEPS.md` has
the checklist and the manuscript edits.

## Data use

MIMIC-IV is credentialed data. The PhysioNet data use agreement does not allow
sharing it with third parties, and PhysioNet's guidance on online services
extends that to sending it to hosted AI tools. Run the steps below yourself,
keep the tables and the cohort file outside git (`~/physionet`, `data/`), and
commit only aggregate outputs: `figures/merged_expE6*.json`,
`figures/fig7_mimic*.pdf`, their copies in `aaai/figures/` and
`aaai/tables/e6_*.tex`. These hold summary statistics over thousands of
admissions and no patient-level rows.

Every script prints counts, rates and estimates only. One exception needed
handling: DuckDB quotes the offending CSV row in conversion and parsing errors.
`e6_mimic_cohort.py` therefore reads every column as text and casts with
`TRY_CAST` (a bad value becomes NULL), and if extraction still fails it prints
only the failing step and the exception type; the full message goes to
`data/e6_cohort.error.log`, which must stay local. `--show-errors` prints it,
for the open demo and the fixture only. The experiment script refuses to write
into `figures/` from the fixture or from a cohort under 5,000 admissions (such
as the demo).

## Run

`sim2/run_e6.sh` wraps the steps; each can be run on its own.

```sh
sim2/run_e6.sh demo                      # optional: open MIMIC-IV demo, extraction only (~2 MB)
sim2/run_e6.sh download <physionet-user> # six tables of v3.1 (~3 GB) into ~/physionet/mimiciv/3.1
sim2/run_e6.sh cohort                    # data/e6_cohort.npz and the acceptance checks
sim2/run_e6.sh main                      # figures/merged_expE6.json and Figure 7
sim2/run_e6.sh sensitivity               # tagged runs for the appendix
sim2/run_e6.sh report                    # figures, tables, F7 numbers, validator
```

`download` asks for the PhysioNet password once (it goes to a temporary netrc
file, not the command line), resumes partial files and checks SHA-256 sums
against the release's `SHA256SUMS.txt`. It needs about 4 GB free (the tables
are about 2.6 GB, mostly `labevents`); set `MIMIC_ROOT` to put them elsewhere
and `MIMIC_VERSION=2.2` for v2.2. PhysioNet sends the Basic-auth challenge only
to wget-style clients (other user agents get the browser's 403 page), so the
script's curl identifies itself as wget-compatible. HTTP 401 after the
password means a wrong login; 403 means the login was accepted but the account
has no access to that MIMIC-IV version yet (credentialing, the CITI training
and the project's data use agreement are all required).
`cohort` keeps DuckDB's intermediate tables in an on-disk file next to the
cohort with a memory limit (`MEMORY_LIMIT`, default 4GB), so labevents does
not need to fit in RAM; the file is deleted afterwards.

The underlying commands, if you prefer them:

```sh
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-e6.txt
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root ~/physionet/mimiciv/3.1 --out data/e6_cohort.npz
.venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz --workers 4
.venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz --tag rrr25 --rrr 0.25   # sensitivity runs
.venv/bin/python sim2/make_figures2.py && .venv/bin/python sim2/make_tables.py
.venv/bin/python sim2/e6_report.py && .venv/bin/python sim2/validate_josh_results.py
```

## Cohort

- Adult admissions (age at admission 18 or older).
- One prediction time per admission, chosen at random (deterministic hash)
  among creatinine draws taken on a ward, at least 4 h after admission and
  before discharge. Ward means a `transfers` care unit that is not an ICU,
  emergency, PACU, obstetric, newborn, psychiatric, discharge-lounge or
  unknown unit, and that never appears as an ICU unit in `icustays`.
- Features: age, sex, emergency admission, hours since admission, a prior ICU
  stay in the same admission, and the last value in the preceding 24 h of
  creatinine, urea nitrogen, sodium, potassium, chloride, bicarbonate,
  glucose, anion gap, white cells, haemoglobin and platelets. Missing values
  are imputed with the training median and flagged.
- Outcome: ICU admission (`icu/icustays.intime`) or in-hospital death within
  12 h of the prediction time. The two components are stored separately.
- Periods: `anchor_year_group` 2008-2013 (early) and 2014-2019 (late);
  `--early` and `--late` change this. The cohort also stores
  `year(admittime) - anchor_year`, and `--period-by admission_year` places each
  admission by its approximate year (its anchor group shifted by that offset),
  dropping admissions whose three-year window straddles the boundary.

The cohort script prints the size, patients, event rate overall and by anchor
group, the outcome split, missingness, the prediction times by ward unit
(counts under 10 suppressed), and then the acceptance checks of
`NEXT_STEPS.md` step 1.4 as PASS/FAIL lines; it exits with status 3 if one
fails.

## What is simulated

1. Deployment of a logistic score fitted to 40% of early-period patients.
2. The alert threshold: a 10% alert rate on those patients (`--alert-rate`).
3. Clinician response: an alert is acted upon with probability 0.85 (`--p-act`).
4. The effect of the alert-triggered action: it prevents an event that would
   otherwise have occurred with probability 0.5 and never causes one (`--rrr`).
5. A randomised unalerted arm, 30% of post-deployment patients (`--control-frac`).
6. The retrain-and-redeploy loop.

Covariates, outcomes under historical care and periods are real. Potential
outcomes: Y(standard care) is the MIMIC-IV outcome; Y(alert policy) =
Y (1 - A B) with A ~ Bernoulli(p_act) for alerted patients and
B ~ Bernoulli(rrr). No pre-deployment patient received the alert-triggered
action, so the union-graph estimator faces the eps = 0 positivity failure of
Experiment B.

## Design

- Experiment A analogue. `deploy_only` draws the pre and post samples from the
  same pool of held-out early-period patients, the post sample under the alert
  policy, so the exogenous change is exactly zero and the policy term is the
  whole change (the S3 analogue). `drift_only` and `drift_and_deploy` use the
  held-out early pool before and the late pool after. Targets are exact for the
  pools in expectation over the simulated components; the X / outcome split of
  real drift has no ground truth and is reported, not scored.
- Replicates resample patients with replacement, each with all of their
  admissions, and redraw the simulated components. The monitor and the
  proposed estimator share one density-ratio fit per replicate.
- Experiment C analogue: the late period is split by patient into training
  (60%) and evaluation pools; eight rounds; fixed or rate-held threshold; keep,
  naive refit, refit on A = 0, weighted refit. Events averted are expected
  values on the evaluation pool against its historical outcomes.
- `--workers` runs replicates in parallel; results do not depend on the worker
  count. The main run takes about 20 to 40 minutes with 4 workers.

## Sensitivity runs

`--tag <name>` writes `merged_expE6_<name>.json` and `fig7_mimic_<name>.pdf`
into both figure folders. `run_e6.sh sensitivity` runs `rrr25`, `rrr75`
(`--rrr 0.25/0.75`), `alert05`, `alert20` (`--alert-rate 0.05/0.20`), `admyear`
(`--period-by admission_year`) and, when the cohort has the 2020 - 2022 anchor
group (v3.x), `covid` (`--late "2020 - 2022"`), with 100 and 30 replicates.
`make_tables.py` collects them into `aaai/tables/e6_sensitivity.tex`.

## Outputs and manuscript

- `figures/merged_expE6.json`, `figures/fig7_mimic.pdf` (and mirrors).
- `aaai/tables/e6_mimic.tex` (cohort in the caption, attribution and update
  rules) and `aaai/tables/e6_sensitivity.tex`; the appendix includes both when
  present.
- `sim2/e6_report.py` prints the numbers F7 quotes, in the exact format the
  validator checks, and a draft of the F7 result sentences whose wording
  follows the sign of the policy term.

## What to expect

With a 5 to 10% event rate, alerted patients mostly have predicted risk below
0.5. Preventing their events then lowers the Brier score (the derivative of
E[(r - Y)^2] in P(Y = 1) is 1 - 2r), so the policy term is negative. The score
still degrades as a model: AUROC falls, and it over-predicts (observed over
expected events and the calibration slope both fall). `sim2/check_performance_drop.py`
(`figures/performance_drop.json`) shows this at every prevalence of the
synthetic grid and at both hospitals of the open stand-in; only the sign of the
Brier change depends on prevalence (positive at the 40% operating point). The
attribution question is unchanged (the monitor books the policy term to the
outcome mechanism); the text should report the drop in discrimination and
calibration and the sign of the Brier change separately.

## Open stand-in (until MIMIC-IV access)

`sim2/e6_sepsis_cohort.py` builds the same cohort format from the
PhysioNet/CinC Challenge 2019 training data (open, CC BY 4.0; Reyna et al.,
Crit Care Med 2020): 40,336 ICU stays from two hospital systems, hourly vitals
and labs. The two hospitals take the place of the two periods, in both
directions (score fitted at A with a shift to B, and the reverse). One
prediction time per stay, at least 4 h into it and before sepsis onset;
outcome sepsis onset within 12 h; features age, sex, ICU type, hours in the
ICU, hours in hospital before ICU admission, and the last value in the
preceding 24 h of six vital signs and eight labs. Chloride and bicarbonate are
left out: hospital B almost never records them, so their missing indicators
identify the hospital and the density ratio between hospitals degenerates
(hospital classifier AUROC 0.98 with them, 0.79 without). 426 stays labelled
septic from their first hour have no valid prediction time; the other 39,910
are all used, and every refit uses the whole training pool (`--n-train 0`).
The files (313 MB) come from PhysioNet's open S3 mirror and are checked
against the S3 checksums; 2,000 cohort rows were rebuilt independently from
the raw files without a mismatch.

```sh
.venv/bin/python sim2/e6_sepsis_cohort.py          # fetch and build data/e6_sepsis_cohort.npz
.venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_sepsis_cohort.npz --early A --late B --n-train 0 \
    --output figures/e6_open/merged_expE6_sepsis2019_AtoB.json --figure figures/e6_open/fig_e6_sepsis2019_AtoB.pdf
.venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_sepsis_cohort.npz --early B --late A --n-train 0 \
    --output figures/e6_open/merged_expE6_sepsis2019_BtoA.json --figure figures/e6_open/fig_e6_sepsis2019_BtoA.pdf
```

`exp_E6_mimic.py` refuses to write a non-MIMIC cohort into the E6 slot; the
stand-in lives in `figures/e6_open/` and is not in the manuscript.

Results (1 October; event rate 3.6% at A, 1.8% at B):

- Deployment only. The policy contrast is -0.0033 at A and -0.0016 at B
  (deployment lowers the Brier score at these event rates, while AUROC falls
  by 0.036 and 0.040 and the score over-predicts). The monitor books the whole
  observed change to P(Y | X) (-0.0032 at A, -0.0014 at B) and nothing to P(X).
  The randomised arm recovers the policy term (-0.0032 and -0.0015, bias
  intervals covering zero). The union-graph estimator with oracle weights
  overshoots by 66% and 46%.
- Shift and deployment. The randomised arm recovers the policy term in both
  directions (-0.0007 against -0.0007; -0.0057 against -0.0057) and the
  exogenous change. Moving from B to A, the alert's benefit hides a real
  deterioration (+0.0181): the observed change is +0.0126, and the monitor
  assigns -0.0022 to P(Y | X) where the proposed decomposition finds +0.0020.
- Retraining at the new hospital, round 8, fixed threshold. At A (3.6% events)
  naive refitting averts 0.72 points and refitting on untreated patients 0.88
  (difference +0.16 [0.14, 0.18]); holding the alert rate reduces the
  difference to +0.02. At B (1.8% events) the difference is +0.017 [0.014,
  0.020]. The weighted refit falls between the two in both directions.

## Testing without MIMIC-IV

`sim2/e6_fixture.py` writes a synthetic dataset with the MIMIC-IV column
layout and a `FIXTURE_NOT_MIMIC` marker: real care-unit names, ED visits
without an admission, repeat admissions years after the anchor year, a
2020 - 2022 group, text-only lab values, quoted comments with commas and line
breaks, and a urine creatinine item with the blood item's label. Cohorts built
from it are flagged, and `exp_E6_mimic.py` refuses to write them into
`figures/`:

```sh
.venv/bin/python sim2/e6_fixture.py --out /tmp/mimic_fixture --patients 8000
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root /tmp/mimic_fixture --out /tmp/fixture_cohort.npz --show-errors
.venv/bin/python sim2/exp_E6_mimic.py --cohort /tmp/fixture_cohort.npz --reps-a 20 --reps-c 6 \
    --output /tmp/fixture_E6.json --figure /tmp/fixture_E6.pdf
.venv/bin/python sim2/e6_report.py /tmp/fixture_E6.json
```

Nothing produced from the fixture is a result.
