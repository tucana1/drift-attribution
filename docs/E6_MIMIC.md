# E6: semi-synthetic MIMIC-IV experiment

Task E6 in the team to-do: real covariates and outcomes for a deterioration
task, a simulated alert policy and treatment effect on top, Experiments A and
C repeated. The code is complete and tested on a synthetic stand-in. The
figure and numbers for the paper have to be produced by a team member with
credentialed MIMIC-IV access, on their own machine.

## Data use

MIMIC-IV is credentialed data. The PhysioNet data use agreement does not allow
sharing it with third parties, and PhysioNet's guidance on online services
extends that to sending it to hosted AI tools. Run the two scripts below
locally, keep the cohort file in `data/` (ignored by git), and commit only the
aggregate outputs: `figures/merged_expE6.json` and `figures/fig7_mimic.pdf`.
These contain summary statistics over thousands of admissions and no
patient-level rows.

## Run

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-e6.txt
# 1. cohort (a few minutes; labevents is the large table)
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root /path/to/mimiciv/3.1 --out data/e6_cohort.npz
# 2. experiments A and C, figure, JSON (about 10 to 20 minutes on 4 cores)
.venv/bin/python sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz
.venv/bin/python sim2/validate_josh_results.py
```

`--mimic-root` is the directory that contains `hosp/` and `icu/` (MIMIC-IV
v2.2 or v3.x, `.csv.gz` or `.csv`). The first script prints the cohort size,
event rate by `anchor_year_group` and missingness; check these before step 2.

## Cohort

- Adult admissions (age at admission 18 or older).
- One prediction time per admission, chosen at random (deterministic hash)
  among creatinine draws taken on a ward, at least 4 h after admission and
  before discharge. Ward means a `transfers` care unit that is not an ICU,
  emergency, PACU, obstetric, newborn, psychiatric, discharge-lounge or
  unknown unit.
- Features: age, sex, emergency admission, hours since admission, a prior ICU
  stay in the same admission, and the last value in the preceding 24 h of
  creatinine, urea nitrogen, sodium, potassium, chloride, bicarbonate,
  glucose, anion gap, white cells, haemoglobin and platelets. Missing values
  are imputed with the training median and flagged.
- Outcome: ICU admission (`icu/icustays.intime`) or in-hospital death within
  12 h of the prediction time.
- Periods: `anchor_year_group` 2008-2013 (early) and 2014-2019 (late). The
  script arguments `--early` and `--late` change this.

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

## What to expect

With a 5 to 10% event rate, alerted patients mostly have predicted risk below
0.5. Preventing their events then lowers the Brier score (the derivative of
E[(r - Y)^2] in P(Y = 1) is 1 - 2r), so the policy term is negative:
deployment makes the score look better on the Brier scale while observed
AUROC falls. The synthetic operating point (event rate near 40%) gives the
opposite sign. The attribution question is unchanged (the monitor books the
policy term to the outcome mechanism), but the text should not describe the
semi-synthetic result as a performance drop unless the numbers show one.

## Testing without MIMIC-IV

`sim2/e6_fixture.py` writes a small synthetic dataset with the MIMIC-IV column
layout and a `FIXTURE_NOT_MIMIC` marker. Cohorts built from it are flagged,
and `exp_E6_mimic.py` refuses to write them into `figures/`:

```sh
.venv/bin/python sim2/e6_fixture.py --out /tmp/mimic_fixture
.venv/bin/python sim2/e6_mimic_cohort.py --mimic-root /tmp/mimic_fixture --out /tmp/fixture_cohort.npz
.venv/bin/python sim2/exp_E6_mimic.py --cohort /tmp/fixture_cohort.npz --reps-a 20 --reps-c 6 \
    --output /tmp/fixture_E6.json --figure /tmp/fixture_E6.pdf
```

Nothing produced from the fixture is a result.
