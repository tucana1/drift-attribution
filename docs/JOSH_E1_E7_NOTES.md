# Josh's experiment tasks: E1 to E7

Status of the experiment tasks assigned to Josh in the September 16 team
to-do, what each script does, what changed from the first pass on this branch,
and what is still open. Every number in the manuscript that comes from these
tasks is read from a JSON in `figures/` written by a script in `sim2/`; tables
are generated into `aaai/tables/` by `sim2/make_tables.py` and
`sim2/make_tables_e7.py`, and `sim2/validate_josh_results.py` checks the
evidence, the mirrored copies, the generated tables and the quoted numbers.

## Status

| Task | Status | Evidence | Script |
|---|---|---|---|
| E1 decision rules | done; low-prevalence run added | `merged_expE1.json`, `merged_expE1_prev.json`, Table `tab:decision`, Figure 4 (`fig6_decision.pdf`) | `exp_E1_decision.py`, `make_figure_E1.py` |
| E2 information budget | done | Table `tab:information` (every estimator and update rule, oracle rows marked) | manuscript |
| E3 conditioning correction | done, extended; estimated-propensity arm added | `merged_expH.json`, Figure 3, Table `tab:retrain-app` | `exp_H_correction.py`, `make_figure3_E3.py` |
| E4 robustness | done, intervals added | `merged_expAB_robustness.json`, Table `tab:robustness`, Table `tab:prevalence` | `exp_AB_robustness.py`, `exp_prevalence.py` |
| E5 persistent-site wedge | done, estimator fixed | `merged_expG2.json`, Table `tab:sites`, Table 1 | `exp_G2_sites.py` |
| E6 MIMIC-IV semi-synthetic | code complete, runner ready; needs the credentialed run | `docs/E6_MIMIC.md` | `run_e6.sh`, `e6_mimic_cohort.py`, `exp_E6_mimic.py`, `e6_report.py`, `e6_fixture.py` |
| E7 intervals and harm | done; RD and net-benefit comparisons corrected; low-prevalence A added | `merged_expE7.json`, `merged_expE7_prev.json`, Figures 1, 2, 5, 6, Tables `tab:coverage`, `tab:positivity-ci`, `tab:harm`, `tab:prevalence-est` | `exp_E7_intervals.py`, `make_figures2.py` |

## What changed from the first pass, and why

**E3.** The first pass compared five rules at a rate-held threshold only, and
all corrections were correctly specified, so the "hierarchy sentence" said
only that the interaction correction does not beat the others. The rerun
(`exp_H_correction.py`) covers the fixed and the rate-held threshold,
homogeneous and heterogeneous effects, and a misspecified score class, with
seven rules. Findings: naive refitting under the fixed threshold enters the
period-2 cycle of Liley et al. (2021, Theorem 1), with observed AUROC in
antiphase to patient benefit; the rate-held threshold removes the loss only
under a homogeneous effect and hides a calibration loss; conditioning on the
action with main effects only fails under heterogeneity (the correction the
task was about); refitting below the threshold fails under misspecification;
unweighted refits on untreated patients drift under misspecification because
the policy selects who stays untreated; weighting by the inverse probability
of remaining untreated (needs alert exposure and response probabilities in
the log) keeps the benefit in all six settings. The fixed-threshold
oscillation and the antiphase result are back in Figure 3, as decision 5 and
the new wording in Section 1.2 of the to-do require.

**E5.** The first pass explained the bias of the Brier-scale X split under
random rollout as "finite-sample and location-model error". That was wrong:
the reweighting step is unbiased for a single site even at 400 patients. The
bias comes from imputing the controls' average covariate trend into each
site's loss curve when trends are site specific. The script now also reports
a DiD-anchored split (exogenous part from the not-yet-deployed sites' loss
change), which is unbiased under random order even with site-specific trends,
and the design is a full 2 x 2 (random or characteristic-linked order, common
or site-specific trends; 200 seeds per cell). New result: characteristic-linked
rollout biases the retrospective term by about -10% even with common covariate
trends, because parallel covariate trends do not give parallel Brier trends.

**E4.** Unchanged design; the appendix table is now generated with 95%
bootstrap intervals. Added `exp_prevalence.py`: at the calibrated operating
point the untreated event rate is about 40%; with the alert rate held at 20%
and the event rate lowered to 14% or 7%, the same deployment lowers the Brier
score. The monitor still books all of it to P(Y|X), so F1 survives, but the
paper's "performance drops" framing holds only at high prevalence.

**E2.** The table now lists every estimator and update rule reported
(attribution estimators, the E3 update rules, the E1 decision rules), with the
oracle column marking simulation-only rows.

**E1 (new).** Keep-or-refit decision after the monitoring window of S1-S4,
outcome = expected events in the next window with alerts acted upon, 200
replicates per scenario in independent seed blocks. The task's rule (refit
when the proposed decomposition assigns more than half of the change to
exogenous mechanisms) avoids the 11.5-point losses of always refitting in S3
and S4 but refits in S1, where naive refitting on standard-care data costs 4.6
points; refitting only when the outcome-mechanism term dominates has zero
regret in S1-S4. The monitor-driven rule refits in S3 and loses everything.
With a rate-held threshold the decision matters little (under 1.3 points).
Refitting on untreated patients dominates every keep-or-naive-refit rule.

**E7 (new).** `exp_E7_intervals.py` reruns Experiments A, B, D, E and F with an
independent seed block per row (decision 3) and stores every replicate, so all
main-text numbers carry 95% bootstrap intervals over replicates; it adds a
patient-level bootstrap for the attribution estimators and reports coverage
of each estimator's own estimand, and a harm term for unnecessary treatment.
Figures 1, 2, 5 and 6 now read this file. Results: the proposed estimator's
analyst intervals cover its estimand close to the nominal 95% (90% to 99%
across terms and scenarios); the monitor's intervals cover the marginal game
it estimates but never the policy-contrast outcome term in S3 or S4; at
eps = 0 the union-graph interval excludes its target at every n up to
400,000 while narrowing. A harm of 0.2 events per unnecessary treatment
removes about a sixth of the deployment benefit and changes no ranking.

**E6 (new).** Pipeline complete and tested end to end on a synthetic stand-in
with the MIMIC-IV table layout. The results need a local run on credentialed
data; see `docs/E6_MIMIC.md`. The manuscript has the section, the simulated
components list, the figure slot and a TODO for the numbers.

## Second pass (30 September)

Every script and quoted number was checked against the manuscript. Reruns on
macOS (arm64) reproduce the committed JSON to within 1e-12 and the figure
PDFs byte for byte, so the diffs in the rerun JSON files are float noise
except where noted.

Corrections to the manuscript (each traced to code or JSON):

- **F6, threshold discontinuity.** The jump at the threshold was compared with
  the effect of treatment against no treatment among the alerted, E[Y(pi1) -
  Y(0)] (-0.405), which counts the standard-care treatment of suspicious
  patients. The policy term needs E[Y(pi1) - Y(pi0)]: -0.291 among the
  alerted. The jump (-0.332) is the local policy effect at tau and overstates
  that average by 14%; it is not an 18% underestimate. The policy-effect curve
  is hump-shaped and falls to zero at high scores. `exp_E7_intervals.py` part
  E now reports both quantities and the curve.
- **F6, net benefit.** The randomised unalerted arm receives standard care, so
  it identifies net benefit against Y(pi0) (+0.080 at theta = 2.5, any
  adherence), not the counterfactual net benefit against Y(0) (+0.162). Part D
  reports `nb_standard_care`.
- **Two references.** Removed "the interaction grows with the size of the
  exogenous covariate shift": no script supported it, and it is not
  monotone (I(X, pi) peaks near delta = 0.75 and vanishes by delta = 1.5).
- **F3.** Main-effects conditioning (5.72 fixed, 5.47 rate-held) is compared
  with the rate-held naive refit (5.42), not naive refitting in general (1.82
  with the fixed threshold). Refitting on untreated or unalerted patients is
  described as an observational version of the holdout update of Liley et al.
  and Haidar-Wehbe et al., whose holdout is randomised.
- **F4.** The 1.3-point bound under the rate-held threshold holds for the rules
  in Table `tab:decision`; the holdout refit, not in the table, differs from
  naive refitting by 1.37 in S4.
- **Figure 4.** The "monitor" line plotted the union-graph analyst estimator;
  it now plots the two-player monitor, which part F now computes. F5's
  statements hold for both.
- **Tables.** The harm table caption states the interval half-width (0.05);
  the coverage table fits the column.

Code:

- **E1.** The share rules compared X + Y (or Y) with dR / 2, which inverts the
  rule when dR < 0. They now use shares relative to dR. Every reported
  replicate has dR > 0, so all E1 decisions and numbers are unchanged. The
  oracle-driven rules use expected Brier scores (no outcome noise), so the
  oracle policy term in S1 is exactly zero instead of +0.0015.
- **E3.** New arm `untreated_ipw_est`: the weighted refit with P(A = 1 | X)
  estimated by a logistic regression of A on X, the alert and their
  interaction. Within 0.01 points of the logged-probability version in all six
  settings, so the weighted refit does not need the response probabilities.
- **E7.** `--merge` reruns single parts and keeps the rest of the JSON; parts D,
  E and F were rerun this way (A and B are unchanged).
- **E4.** `exp_AB_robustness.py` writes through `write_json`, so a rerun
  updates both figure folders.
- **Validator.** Checks the quoted numbers of Two references, F1, F2 and F6,
  the E1 1.3-point bound, the E3 estimated weights and the low-prevalence
  appendix; the E3 keep-arm check covers every round (it stopped after the
  first row); the E5 sign check no longer hard-codes the minus sign.

**Low prevalence (team decision 1, option 2).** Experiment A (E7 machinery,
100 replicates with patient-level bootstrap) and E1 (200 replicates per
scenario) at b0 = -3.5, alert rate 20%, untreated event rate 6.6%
(`merged_expE7_prev.json`, `merged_expE1_prev.json`, Table
`tab:prevalence-est`). Under S3 the change is -0.0065; the monitor books all
of it to P(Y | X) and its interval covers the policy-contrast outcome term in
2% of replicates, while the randomised arm covers its target in 96%. Under S4
the policy term (-0.0178) offsets most of the outcome-mechanism shift
(+0.0242), and the monitor books +0.0098 to P(Y | X): the alert's success
masks a real deterioration. In E1 the monitor-driven rule refits in every S3
replicate and loses the benefit, the outcome-share rule loses 0.55 points in
S4 (its oracle share is 0.52, at the cut-off), and refitting on untreated
patients matches or beats keeping and naive refitting everywhere. The score
still degrades as a model at this prevalence: AUROC falls and the score
over-predicts after deployment (`sim2/check_performance_drop.py`); only the
Brier change is negative.

**E6.** Changes before the first credentialed run:

- Extraction reads every column as text and casts with `TRY_CAST`, with the
  CSV dialect fixed; DuckDB works in an on-disk file with a memory limit;
  errors are reported by step and type only, with the full message in a local
  log (DuckDB quotes the offending row in conversion errors). The script
  checks the NEXT_STEPS acceptance criteria and stores the outcome
  components, the care unit and `year(admittime) - anchor_year`.
- `deploy_only` draws pre and post from one early-period pool, so the
  exogenous target is exactly zero (with two patient-disjoint halves it was
  their sampling difference, about 0.001 Brier on the full cohort, against a
  policy term of a few thousandths). Replicates resample patients, as the
  docstring said, instead of admissions. Monitor and proposed estimator share
  one density-ratio fit; `--workers` parallelises with identical results.
- `--tag` for sensitivity runs, `--period-by admission_year`, and
  `sim2/run_e6.sh` (demo smoke test, download with checksum, cohort, main and
  sensitivity runs, report). `sim2/e6_report.py` prints the F7 numbers in the
  format the validator checks; `make_tables.py` writes `e6_mimic.tex` and
  `e6_sensitivity.tex`, which the appendix includes when present.
- The fixture now covers real care-unit names, ED visits without an
  admission, repeat admissions, a 2020 - 2022 group, text-only lab values,
  quoted comments with commas and line breaks, and a urine creatinine item.

## Seed blocks

See the Seeds section of `README.md` (root seed per experiment, and why the
first-pass randomised-arm seeds start at 100).

## Reproduce

```sh
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python sim2/exp_H_correction.py      # E3, about 3 min
.venv/bin/python sim2/exp_E1_decision.py       # E1, about 2 min
.venv/bin/python sim2/exp_G2_sites.py          # E5, under 1 min
.venv/bin/python sim2/exp_AB_robustness.py     # E4
.venv/bin/python sim2/exp_prevalence.py        # prevalence check (oracle)
.venv/bin/python sim2/exp_E7_intervals.py      # E7, about 20 min on 4 cores
# one part only, keeping the others: --parts D,E,F --merge
# low prevalence (appendix Table tab:prevalence-est), about 2 and 10 min
.venv/bin/python sim2/exp_E1_decision.py --b0 -3.5 --alert-rate 0.2 --root-seed 2026092911 --output merged_expE1_prev.json
.venv/bin/python sim2/exp_E7_intervals.py --parts A --b0 -3.5 --alert-rate 0.2 --root-seed 2026092917 --output merged_expE7_prev.json
.venv/bin/python sim2/make_figures2.py         # all figures
.venv/bin/python sim2/make_tables.py           # all generated tables
.venv/bin/python sim2/validate_josh_results.py
```

E6 has its own runner, `sim2/run_e6.sh` (`docs/E6_MIMIC.md`).

## Open items

The handoff for the next worker, with commands, checks and acceptance
criteria, is `docs/NEXT_STEPS.md`.

- E6 numbers and Figure 7: run `sim2/run_e6.sh` on credentialed data.
- Team decisions: prevalence framing (W1, W7), the E1 headline rule, logging
  requirements, the prospective column of the identification table, and the
  threshold-discontinuity appendix (`docs/NEXT_STEPS.md`, section 2).
- Synthetic Hospital, the Zhou et al. trajectory generator and GHOSTS were
  assessed as data sources and not used; see
  `docs/SYNTHETIC_HOSPITAL_ASSESSMENT.md`.
- Three references were added from the to-do's verified list (Liley 2021,
  Haidar-Wehbe 2025, Johnson 2023) and checked, along with the entries
  that had TODO notes in `references.bib` (see `docs/NEXT_STEPS.md`).
- The Section 5 S2 description said "change in w_y"; the code shifts the
  outcome intercept, and the text now says so.
