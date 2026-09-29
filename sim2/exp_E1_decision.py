"""E1: does attribution improve the retraining decision? Outcome-level comparison across S1-S4.

Timeline. e0 is the pre-deployment period in which the score f0 was built. e1
is the monitoring window, generated exactly as scenario S1-S4 of Experiment A
(merged.SCEN): in S1 and S2 the score runs silently (policy pi0), in S3 and S4
its alerts are acted upon. At the end of e1 the team sees the Brier-score change
and decides whether to refit the score. In e2 the chosen score's alerts are
acted upon; e2 keeps e1's exogenous state (covariate mean, outcome intercept).

Decision rules (paired within a replicate: same e0, e1 and e2 draws).
  never             keep f0.
  always            naive refit of the logistic score on the deployed-arm e1
                    (X, Y), the standard response to a monitored drop.
  exo_proposed      refit only if the proposed decomposition (randomised arm,
                    Experiment A) assigns more than half of the change to the
                    exogenous terms, X + Y > dR / 2 (the task E1 rule).
  outcome_proposed  refit only if the outcome-mechanism term alone carries more
                    than half of the change, Y > dR / 2.
  exo_monitor       exogenous rule driven by the two-player monitor without
                    action logs. It has no policy player, so it refits whenever
                    the change is positive; it coincides with `always`.
  outcome_monitor   outcome rule driven by the monitor, Ygx > dR / 2.
  exo_oracle, outcome_oracle
                    the two rules driven by the oracle policy-contrast
                    decomposition (separates estimation error from rule error).
  refit_untreated   reference: always refit on e1 patients with A = 0 (needs the
                    action log; targets the untreated risk).
  refit_holdout     reference: always refit on the randomised unalerted arm (the
                    holdout update of Liley et al. 2021).
  best_of_two       per replicate, the better of keeping and naive refitting
                    (upper bound for any keep-or-refit rule).

Outcome. Expected events per patient in e2 on a fresh evaluation sample, without
outcome noise, reported as events averted against standard care in e2
(percentage points), plus unnecessary treatments for the harm sensitivity.
Alert rules in e2: the fixed threshold tau_r, or the original alert rate held
on e2's covariate distribution. Each (threshold rule, scenario) row has its own
seed block.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import columns, rng_for, summarize, write_json
from dgp import mu_of, p_treat, p_y1, r_hat, shapley, suspicion, truth_values
from merged import E3, SCEN, make_cfg, naive2_est, sample_po, truth_E3_decomp

EXPERIMENT = 1
RULES = ("never", "always", "exo_proposed", "outcome_proposed", "exo_monitor", "outcome_monitor",
         "exo_oracle", "outcome_oracle", "refit_untreated", "refit_holdout", "best_of_two")
THRESHOLD_RULES = ("fixed", "rate")
HARM_GRID = (0.0, 0.05, 0.1, 0.2)


def refit(x, y):
    model = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
    return lambda xx: model.predict_proba(xx)[:, 1]


def outcome_term(decomp):
    return decomp["Y"] if "Y" in decomp else decomp["Ygx"]


def exogenous_rule(decomp):
    """Refit when the exogenous terms carry more than half of the change."""
    return bool(decomp["X"] + outcome_term(decomp) > 0.5 * decomp["dR"])


def outcome_rule(decomp):
    """Refit when the outcome-mechanism term alone carries more than half of the change."""
    return bool(outcome_term(decomp) > 0.5 * decomp["dR"])


def evaluate(cfg, score, x_eval, x_cal, rule, held_rate):
    """Expected events etc. in e2 when `score` drives the alerts."""
    s = score(x_eval)
    threshold = cfg.tau_r if rule == "fixed" else float(np.quantile(score(x_cal), 1 - held_rate))
    susp = suspicion(cfg, x_eval) > cfg.tau_s
    p = np.where(susp, cfg.q_hi, np.where(s > threshold, cfg.p_alert, cfg.eps))
    q0 = p_y1(cfg, x_eval, np.zeros(len(x_eval)), cfg.dy != 0)
    q1 = p_y1(cfg, x_eval, np.ones(len(x_eval)), cfg.dy != 0)
    return {"events": float(np.mean(q0 + p * (q1 - q0))), "alert_rate": float(np.mean(s > threshold)),
            "unnecessary": float(np.mean(p * (1 - q0))), "treated": float(np.mean(p))}


def standard_care(cfg, x_eval):
    p = p_treat(cfg, x_eval, False)
    q0 = p_y1(cfg, x_eval, np.zeros(len(x_eval)), cfg.dy != 0)
    q1 = p_y1(cfg, x_eval, np.ones(len(x_eval)), cfg.dy != 0)
    return float(np.mean(q0 + p * (q1 - q0))), float(np.mean(p * (1 - q0)))


def run(args):
    rows, truths = [], {}
    start = time.time()
    for t_index, t_rule in enumerate(THRESHOLD_RULES):
        for s_index, (scenario, kw) in enumerate(SCEN.items()):
            cfg = make_cfg(**kw)
            tv = truth_values(cfg, ["X", "pi", "Y"], n=args.n_truth, metric="brier",
                              seed=int(rng_for(args.root_seed, EXPERIMENT, 90, s_index).integers(2 ** 31)))
            oracle = truth_E3_decomp(tv)
            truths[scenario] = {"policy_contrast": oracle, "shapley": shapley(tv, ["X", "pi", "Y"])}
            f0 = lambda xx: r_hat(cfg, xx)
            # original alert rate: f0 on the pre-deployment covariate distribution
            base_rng = rng_for(args.root_seed, EXPERIMENT, 91, 0)
            held_rate = float(np.mean(f0(base_rng.normal(size=(200_000, cfg.d))) > cfg.tau_r))
            e2_mean = mu_of(cfg, None, {"X", "Y"})  # e2 keeps e1's exogenous covariate state
            for rep in range(args.reps):
                rng = rng_for(args.root_seed, EXPERIMENT, t_index, s_index, rep)
                s0 = sample_po(cfg, set(), args.n, rng)
                s1 = sample_po(cfg, {"X", "pi", "Y"}, args.n, rng, control_frac=args.control_frac)
                x_eval = rng.normal(size=(args.n_eval, cfg.d)) + e2_mean
                x_cal = rng.normal(size=(args.n_eval, cfg.d)) + e2_mean
                proposed, monitor = E3(cfg, s0, s1, "rollout"), naive2_est(cfg, s0, s1)
                deployed = s1["arm"] == 1
                refit_naive = refit(s1["x"][deployed], s1["y"][deployed])
                control = s1["arm"] == 0
                if control.sum() == 0:
                    control = np.ones(len(s1["y"]), bool)   # silent window: all patients are under pi0
                refit_holdout = refit(s1["x"][control], s1["y"][control])
                untreated = s1["a"] == 0
                refit_untr = refit(s1["x"][untreated], s1["y"][untreated])
                std_events, std_unnecessary = standard_care(cfg, x_eval)
                keep = evaluate(cfg, f0, x_eval, x_cal, t_rule, held_rate)
                retrain = evaluate(cfg, refit_naive, x_eval, x_cal, t_rule, held_rate)
                decisions = {"never": False, "always": True,
                             "exo_proposed": exogenous_rule(proposed), "outcome_proposed": outcome_rule(proposed),
                             "exo_monitor": exogenous_rule(monitor), "outcome_monitor": outcome_rule(monitor),
                             "exo_oracle": exogenous_rule(oracle), "outcome_oracle": outcome_rule(oracle)}
                outcome = {rule: (retrain if flag else keep) for rule, flag in decisions.items()}
                outcome["refit_untreated"] = evaluate(cfg, refit_untr, x_eval, x_cal, t_rule, held_rate)
                outcome["refit_holdout"] = evaluate(cfg, refit_holdout, x_eval, x_cal, t_rule, held_rate)
                outcome["best_of_two"] = retrain if retrain["events"] < keep["events"] else keep
                decisions["refit_untreated"] = decisions["refit_holdout"] = True
                decisions["best_of_two"] = bool(retrain["events"] < keep["events"])
                for rule in RULES:
                    o = outcome[rule]
                    rows.append({"threshold_rule": t_rule, "scenario": scenario, "rep": rep, "rule": rule,
                                 "retrained": decisions[rule], "events": o["events"],
                                 "standard_care_events": std_events,
                                 "events_averted_pp": 100 * (std_events - o["events"]),
                                 "unnecessary": o["unnecessary"], "standard_care_unnecessary": std_unnecessary,
                                 "treated": o["treated"], "alert_rate": o["alert_rate"],
                                 "dR_observed": float(proposed["dR"]),
                                 "proposed_X": float(proposed["X"]), "proposed_Y": float(proposed["Y"]),
                                 "proposed_pi": float(proposed["pi"]),
                                 "monitor_X": float(monitor["X"]), "monitor_Ygx": float(monitor["Ygx"])})
            print(f"finished {t_rule} {scenario} ({time.time() - start:.0f}s)", flush=True)
    return rows, truths


def summarise(rows, args):
    table, harm = [], []
    for t_rule in THRESHOLD_RULES:
        for scenario in SCEN:
            cell = [r for r in rows if r["threshold_rule"] == t_rule and r["scenario"] == scenario]
            by = {rule: sorted((r for r in cell if r["rule"] == rule), key=lambda r: r["rep"]) for rule in RULES}
            best = np.array([r["events_averted_pp"] for r in by["best_of_two"]])
            for rule in RULES:
                v = np.array([r["events_averted_pp"] for r in by[rule]])
                table.append({"threshold_rule": t_rule, "scenario": scenario, "rule": rule,
                              "events_averted_pp": summarize(v, seed=1),
                              "regret_vs_best_of_two_pp": summarize(best - v, seed=2),
                              "difference_vs_never_pp": summarize(v - np.array([r["events_averted_pp"] for r in by["never"]]), seed=3),
                              "difference_vs_always_pp": summarize(v - np.array([r["events_averted_pp"] for r in by["always"]]), seed=4),
                              "retrain_fraction": float(np.mean([r["retrained"] for r in by[rule]])),
                              "alert_rate": float(np.mean([r["alert_rate"] for r in by[rule]]))})
                for h in HARM_GRID:
                    net = [100 * ((r["standard_care_events"] - r["events"])
                                  - h * (r["unnecessary"] - r["standard_care_unnecessary"])) for r in by[rule]]
                    harm.append({"threshold_rule": t_rule, "scenario": scenario, "rule": rule,
                                 "harm_per_unnecessary_treatment": h, "net_events_averted_pp": summarize(net, seed=5)})
    # worst-case regret across scenarios per rule (scenario means; independent seed blocks per scenario)
    worst = []
    for t_rule in THRESHOLD_RULES:
        for rule in RULES:
            regrets = [e["regret_vs_best_of_two_pp"]["mean"] for e in table
                       if e["threshold_rule"] == t_rule and e["rule"] == rule]
            worst.append({"threshold_rule": t_rule, "rule": rule, "max_mean_regret_pp": float(max(regrets)),
                          "mean_regret_pp": float(np.mean(regrets))})
    return table, harm, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--n-eval", type=int, default=40000)
    parser.add_argument("--n-truth", type=int, default=400000)
    parser.add_argument("--control-frac", type=float, default=0.3)
    parser.add_argument("--root-seed", type=int, default=2026092901)
    parser.add_argument("--output", default="merged_expE1.json")
    args = parser.parse_args()
    rows, truths = run(args)
    table, harm, worst = summarise(rows, args)
    result = {
        "design": ("Keep-or-refit decision after the e1 monitoring window of scenarios S1-S4; outcome = expected "
                   "events in e2 (alerts acted upon, e1 exogenous state kept) against standard care. Independent "
                   "seed block per (threshold rule, scenario); rules paired within a replicate. Intervals are 95% "
                   "percentile bootstraps over replicates."),
        "parameters": {"reps": args.reps, "n_per_environment": args.n, "n_eval": args.n_eval,
                       "n_truth": args.n_truth, "control_frac": args.control_frac, "root_seed": args.root_seed,
                       "experiment_block": EXPERIMENT, "rules": list(RULES),
                       "threshold_rules": list(THRESHOLD_RULES), "harm_grid": list(HARM_GRID)},
        "truth": truths, "table": table, "harm_sensitivity": harm, "worst_case": worst,
        "runs": columns(rows, list(rows[0].keys())),
    }
    write_json(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
