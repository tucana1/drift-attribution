# Drift attribution under deployment feedback

Code, results and draft for the workshop paper. The draft is
`aaai/drift_attribution_aaai.tex`; every number in it is read from a JSON file in
`figures/` written by a script in this repository.

## Layout

| Path | Contents |
|---|---|
| `sim2/` | Simulator (`dgp.py`, `merged.py`), experiments, figure and table scripts, validator |
| `sim/` | First-pass simulator and experiments (kept for reference; not used by the draft) |
| `figures/` | Results (JSON) and figures (PDF); `aaai/figures/` holds identical copies for LaTeX |
| `aaai/` | Draft, bibliography, generated tables (`aaai/tables/`) |
| `docs/` | Notes on the experiments, the MIMIC-IV run and data sources |

## Setup

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt      # requirements-e6.txt for the MIMIC-IV run
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

Every command below runs from any directory and writes to both `figures/` and
`aaai/figures/`.

## Which script produces which file

Times are on 4 cores.

| File in `figures/` | Command | Time |
|---|---|---|
| `merged_expE7.json` (Experiments A, B, D, E, F with intervals) | `python3 sim2/exp_E7_intervals.py` | 12 min |
| `merged_expE7_prev.json` (Experiment A at 6.6% event rate) | `python3 sim2/exp_E7_intervals.py --parts A --b0 -3.5 --alert-rate 0.2 --root-seed 2026092917 --workers 6 --output merged_expE7_prev.json` | 9 min |
| `merged_expH.json` (Experiment C, update rules) | `python3 sim2/exp_H_correction.py` | 2.5 min |
| `merged_expE1.json` (keep-or-refit rules) | `python3 sim2/exp_E1_decision.py` | 1 min |
| `merged_expE1_prev.json` (same, 6.6% event rate) | `python3 sim2/exp_E1_decision.py --b0 -3.5 --alert-rate 0.2 --root-seed 2026092911 --output merged_expE1_prev.json` | 1 min |
| `merged_expG2.json` (site rollout) | `python3 sim2/exp_G2_sites.py` | 17 s |
| `merged_expAB_robustness.json` (more covariates, misspecification) | `python3 sim2/exp_AB_robustness.py` | 30 s |
| `merged_expP.json` (prevalence) | `python3 sim2/exp_prevalence.py` | 7 s |
| `performance_drop.json` | `python3 sim2/check_performance_drop.py` | 3 s |
| `merged_expA.json` ... `merged_expM.json` (first-pass A to E and M) | `python3 sim2/merged.py A` (or B, C, D, E, M) | under 15 s each |
| `merged_expF.json` (first-pass triage) | `python3 sim2/exp_F_triage.py` | 9 s |
| `exp1_results.json` ... `exp4_rd.json` (first pass, `sim/`) | `python3 sim/exp1_scenarios.py`, `sim/exp2_retrain.py`, `sim/exp3_sweep_exp4_rd.py`; figures `sim/make_figures.py` | under 10 s each |
| `merged_expE6.json` (MIMIC-IV) | `sim2/run_e6.sh`, on a machine with credentialed access (`docs/E6_MIMIC.md`) | |
| `e6_open/*.json` (open stand-in) | `sim2/exp_E6_mimic.py` on the `sim2/e6_sepsis_cohort.py` cohort (`docs/E6_MIMIC.md`) | |

`python3 sim2/validate_josh_results.py` fails if a data file in `figures/` has
no entry in this list (`PRODUCERS` in that script).

Checked on 5 October 2026 by rerunning every simulation command above (all but
the two E6 rows) in a clean copy
(about 30 minutes in total): every file reproduces, with numeric differences of
at most 2e-10 (floating-point rounding in the solvers), and the figures and
tables rebuilt from the rerun are byte-identical to the committed ones. The
`open_stand_in` part of `performance_drop.json` is filled only when
`data/e6_sepsis_cohort.npz` (built from PhysioNet data, not in git) is present.

## Figures and tables

```sh
python3 sim2/make_figures2.py   # all main-text figures (1 to 6; copies Figure 7 if the MIMIC-IV run exists)
python3 sim2/make_tables.py     # every generated table in aaai/tables/
python3 sim2/validate_josh_results.py
```

Both read only the saved JSON files, so they take seconds and do not rerun any
experiment. Figures are saved without a creation date, so regenerating them
gives identical files.

## Seeds

Experiments behind the draft draw every random number from
`np.random.SeedSequence([root, experiment, row, replicate, ...])`
(`sim2/common.py`, `seed_block` and `rng_for`). Each row of a sweep (a
scenario, an epsilon value, a triage level, a threshold rule) has its own
block, so no comparison across rows reuses patients. Estimators or rules
compared within a row share that row's draws, which makes those comparisons
paired.

| Experiment | Script | Root seed |
|---|---|---|
| Keep-or-refit rules | `exp_E1_decision.py` | 2026092901 |
| Keep-or-refit rules, 6.6% event rate | `exp_E1_decision.py --b0 -3.5` | 2026092911 |
| Experiment C, update rules | `exp_H_correction.py` | 2026092903 |
| Site rollout | `exp_G2_sites.py` | 2026092905 |
| MIMIC-IV | `exp_E6_mimic.py` | 2026092906 |
| A, B, D, E, F with intervals | `exp_E7_intervals.py` | 2026092907 |
| Experiment A, 6.6% event rate | `exp_E7_intervals.py --parts A --b0 -3.5` | 2026092917 |
| Prevalence | `exp_prevalence.py` | 2026092908 |
| Performance-drop check | `check_performance_drop.py` | 2026092909 |
| Robustness (d = 10, 20) | `exp_AB_robustness.py` | 2026092204 |

The first-pass scripts (`sim2/merged.py`, `sim2/exp_F_triage.py`, `sim/`) use
plain integer seeds with a fixed offset per experiment. In `merged.py` the
truth tables use seed 0, the union-graph and monitor estimators use seed
`sd = 0, 1, ...`, and the randomised-arm samples start at `100 + sd` so they
never reuse those streams (the comment above `SCEN` in `merged.py` gives the
details). In that scheme rows of a sweep share seeds, so the draft takes its
numbers from the reruns above, not from these files.
