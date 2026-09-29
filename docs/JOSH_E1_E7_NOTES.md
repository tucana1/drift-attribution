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
| E1 decision rules | done | `merged_expE1.json`, Table `tab:decision`, Figure 4 (`fig6_decision.pdf`) | `exp_E1_decision.py`, `make_figure_E1.py` |
| E2 information budget | done | Table `tab:information` (every estimator and update rule, oracle rows marked) | manuscript |
| E3 conditioning correction | done, extended | `merged_expH.json`, Figure 3, Table `tab:retrain-app` | `exp_H_correction.py`, `make_figure3_E3.py` |
| E4 robustness | done (first pass), intervals added | `merged_expAB_robustness.json`, Table `tab:robustness`, Table `tab:prevalence` | `exp_AB_robustness.py`, `exp_prevalence.py` |
| E5 persistent-site wedge | done, estimator fixed | `merged_expG2.json`, Table `tab:sites`, Table 1 | `exp_G2_sites.py` |
| E6 MIMIC-IV semi-synthetic | code complete; needs the credentialed run | `docs/E6_MIMIC.md` | `e6_mimic_cohort.py`, `exp_E6_mimic.py`, `e6_fixture.py` |
| E7 intervals and harm | done | `merged_expE7.json`, Figures 1, 2, 5, 6, Tables `tab:coverage`, `tab:positivity-ci`, `tab:harm` | `exp_E7_intervals.py`, `make_figures2.py` |

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

## Seed blocks

Every E-series script draws from `np.random.SeedSequence([root, experiment,
row, replicate, ...])` (`sim2/common.py`): E1 root 2026092901, E3 2026092903,
E5 2026092905, E6 2026092906, E7 2026092907, prevalence 2026092908. Rows of a
sweep never share draws; rules or estimators compared within a row do.

## Reproduce

```sh
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python sim2/exp_H_correction.py      # E3, about 3 min
.venv/bin/python sim2/exp_E1_decision.py       # E1, about 2 min
.venv/bin/python sim2/exp_G2_sites.py          # E5, under 1 min
.venv/bin/python sim2/exp_AB_robustness.py     # E4
.venv/bin/python sim2/exp_prevalence.py        # prevalence check
.venv/bin/python sim2/exp_E7_intervals.py      # E7, about 20 min on 4 cores
.venv/bin/python sim2/make_figures2.py         # all figures
.venv/bin/python sim2/make_tables.py           # all generated tables
.venv/bin/python sim2/validate_josh_results.py
```

## Open items

- E6 numbers and Figure 7: run `docs/E6_MIMIC.md` on credentialed data.
- The low-prevalence sign flip (Table `tab:prevalence`) affects the framing of
  the introduction (W1) and results (W7); it is flagged there, not rewritten.
- Three references were added from the to-do's verified list (Liley 2021,
  Haidar-Wehbe 2025, Johnson 2023); their titles should be checked against
  the DOI pages (L2/L3), because this environment could not reach them.
- The Section 5 S2 description said "change in w_y"; the code shifts the
  outcome intercept, and the text now says so.
