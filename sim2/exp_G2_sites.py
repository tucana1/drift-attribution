"""E5: persistent-site stepped wedge with characteristic-linked rollout.

The reported estimator is a cohort-at-crossover difference in differences
against not-yet-deployed sites. It uses observed Brier losses and covariate
means. Simulator counterfactuals are used only to measure its bias. The last
cohort serves as a control and has no concurrent control at its own crossover.

Two estimators of the Brier-scale X split are reported.
  plug-in      (first version) reweights the switching site's own pre-rollout
               losses to the controls' average covariate shift (exogenous part)
               and to its own observed shift (total). It imputes the average
               trend into each site's loss curve, so it is biased whenever
               covariate trends are site specific, even with random rollout.
  DiD-anchored takes the exogenous part from the controls' observed loss
               change (the control arm of the difference in differences) and
               the induced part as the reweighted loss change at the site's
               own total shift minus that control change. With random rollout
               both parts are unbiased under site-specific trends; with
               characteristic-linked rollout they inherit the DiD bias.
Settings form a 2 x 2 design: rollout order (random or linked to the site
characteristic) by covariate trend (common or site specific).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import summarize, write_json
from dgp import p_treat, r_hat, sigmoid, suspicion
from merged import make_cfg


ROOT = Path(__file__).resolve().parents[1]
COHORTS = (2, 3, 4, 5)
PERIODS = 6


def response(cfg, x, site_characteristic):
    lin = x @ cfg.w_y + cfg.b0 + 0.4 * site_characteristic
    return sigmoid(lin), sigmoid(lin - cfg.theta)


def expected_loss(cfg, x, z, deployed):
    q0, q1 = response(cfg, x, z)
    p = p_treat(cfg, x, deployed)
    risk = r_hat(cfg, x)
    event_probability = q0 + p * (q1 - q0)
    return float(np.mean(risk ** 2 + (1 - 2 * risk) * event_probability))


def shifted_observed_loss(x, loss, shift):
    """Reweight a site's observed pre-rollout losses under a Gaussian mean shift.

    The mean and covariance are estimated from that site's observed covariates.
    The location-shift model is an explicit estimator assumption, not an oracle
    outcome function.
    """
    mean_x = x.mean(axis=0)
    covariance = np.cov(x, rowvar=False)
    displacement = np.full(x.shape[1], shift)
    direction = np.linalg.solve(covariance, displacement)
    log_weight = (x - mean_x) @ direction
    log_weight -= log_weight.max()
    weight = np.exp(log_weight)
    return float(np.average(loss, weights=weight))


def run_seed(seed, row_index, sites, n, rho, trend, triage, root_seed):
    cfg = make_cfg(deploy=True)
    rng = np.random.default_rng(np.random.SeedSequence([root_seed, row_index, seed]))
    z = rng.normal(size=sites)
    priority = rho * z + np.sqrt(1 - rho ** 2) * rng.normal(size=sites)
    cohort = np.empty(sites, dtype=int)
    for t, block in zip(COHORTS, np.array_split(np.argsort(-priority), len(COHORTS))):
        cohort[block] = t
    observed_loss = np.empty((sites, PERIODS))
    observed_x = np.empty((sites, PERIODS))
    patient_x = {}
    patient_loss = {}
    true_site = {}
    for j in range(sites):
        for t in range(PERIODS):
            deployed = t >= cohort[j]
            normal = rng.normal(size=(n, cfg.d))
            baseline_mean = 0.5 * z[j]
            temporal_shift = (0.10 + trend * z[j]) * t
            x_without_deployment = normal + baseline_mean + temporal_shift
            x = x_without_deployment + (triage if deployed else 0.0)
            probability = p_treat(cfg, x, deployed)
            action = rng.random(n) < probability
            q0, q1 = response(cfg, x, z[j])
            y = rng.random(n) < np.where(action, q1, q0)
            losses = (r_hat(cfg, x) - y) ** 2
            observed_loss[j, t] = np.mean(losses)
            observed_x[j, t] = np.mean(x[:, :2])
            if t == cohort[j] - 1 and cohort[j] in COHORTS[:-1]:
                patient_x[j] = x
                patient_loss[j] = losses
            if t == cohort[j] and t in COHORTS[:-1]:
                x_pretrend = normal + baseline_mean + temporal_shift - (0.10 + trend * z[j])
                standard_at_t = expected_loss(cfg, x_without_deployment, z[j], False)
                shifted_standard = expected_loss(cfg, x, z[j], False)
                deployed_risk = expected_loss(cfg, x, z[j], True)
                true_site[j] = {
                    "retro": deployed_risk - standard_at_t,
                    "x_endogenous_loss": shifted_standard - standard_at_t,
                    "policy_on_current_x": deployed_risk - shifted_standard,
                    "x_exogenous_loss": standard_at_t - expected_loss(cfg, x_pretrend, z[j], False),
                    "x_endogenous_mean": triage,
                    "x_exogenous_mean": 0.10 + trend * z[j],
                }
    measurements = {key: [] for key in (
        "retro_est", "retro_truth", "x_endogenous_mean_est", "x_endogenous_mean_truth",
        "x_exogenous_mean_est", "x_exogenous_mean_truth",
        "x_endogenous_loss_est", "x_endogenous_loss_truth",
        "x_exogenous_loss_est", "x_exogenous_loss_truth",
        "x_endogenous_loss_did_est", "x_exogenous_loss_did_est",
        "policy_on_current_x_truth",
    )}
    for t in COHORTS[:-1]:
        switching = np.flatnonzero(cohort == t)
        controls = np.flatnonzero(cohort > t)
        control_loss_change = float(np.mean(observed_loss[controls, t] - observed_loss[controls, t - 1]))
        control_x_change = float(np.mean(observed_x[controls, t] - observed_x[controls, t - 1]))
        for j in switching:
            observed_shift = observed_x[j, t] - observed_x[j, t - 1]
            exogenous_loss = shifted_observed_loss(patient_x[j], patient_loss[j], control_x_change)
            total_shift_loss = shifted_observed_loss(patient_x[j], patient_loss[j], observed_shift)
            measurements["retro_est"].append(observed_loss[j, t] - observed_loss[j, t - 1] - control_loss_change)
            measurements["retro_truth"].append(true_site[j]["retro"])
            measurements["x_endogenous_mean_est"].append(observed_shift - control_x_change)
            measurements["x_endogenous_mean_truth"].append(true_site[j]["x_endogenous_mean"])
            measurements["x_exogenous_mean_est"].append(control_x_change)
            measurements["x_exogenous_mean_truth"].append(true_site[j]["x_exogenous_mean"])
            measurements["x_exogenous_loss_est"].append(exogenous_loss - observed_loss[j, t - 1])
            measurements["x_endogenous_loss_est"].append(total_shift_loss - exogenous_loss)
            # DiD-anchored split: exogenous part from the controls' loss change.
            measurements["x_exogenous_loss_did_est"].append(control_loss_change)
            measurements["x_endogenous_loss_did_est"].append(
                total_shift_loss - observed_loss[j, t - 1] - control_loss_change)
            for key in ("x_endogenous_loss", "x_exogenous_loss", "policy_on_current_x"):
                measurements[key + "_truth"].append(true_site[j][key])
    out = {key: float(np.mean(value)) for key, value in measurements.items()}
    out["seed"] = seed
    out["rollout_z_correlation"] = float(np.corrcoef(z, -cohort)[0, 1])
    return out


ESTIMATES = (("retro", "retro"), ("x_endogenous_mean", "x_endogenous_mean"),
             ("x_exogenous_mean", "x_exogenous_mean"), ("x_endogenous_loss", "x_endogenous_loss"),
             ("x_exogenous_loss", "x_exogenous_loss"), ("x_endogenous_loss_did", "x_endogenous_loss"),
             ("x_exogenous_loss_did", "x_exogenous_loss"))


def summarize_runs(runs):
    summary = {}
    for est, target in ESTIMATES:
        estimates = np.array([r[est + "_est"] for r in runs])
        truths = np.array([r[target + "_truth"] for r in runs])
        bias = estimates - truths
        interval = summarize(bias, seed=len(summary))
        summary[est] = {
            "target": target,
            "estimate_mean": float(estimates.mean()),
            "truth_mean": float(truths.mean()),
            "bias_mean": float(bias.mean()),
            "bias_mcse": float(bias.std(ddof=1) / np.sqrt(len(runs))),
            "bias_sd": float(bias.std(ddof=1)),
            "bias_ci95": interval["ci95"],
            "relative_bias": float(bias.mean() / truths.mean()),
        }
    for target in ("x_endogenous_loss", "x_exogenous_loss", "policy_on_current_x"):
        summary[target + "_truth_mean"] = float(np.mean([r[target + "_truth"] for r in runs]))
    summary["rollout_z_correlation_mean"] = float(np.mean([r["rollout_z_correlation"] for r in runs]))
    return summary


def run(seeds=200, sites=48, n_per_site_period=400, triage=0.25, root_seed=2026092905):
    settings = (
        ("random_order_common_trend", 0.0, 0.0),
        ("random_order_site_trend", 0.0, 0.04),
        ("correlated_order_common_trend", 0.85, 0.0),
        ("correlated_order_site_trend", 0.85, 0.04),
    )
    rows = []
    for index, (name, rho, trend) in enumerate(settings):
        runs = [run_seed(seed, index, sites, n_per_site_period, rho, trend, triage, root_seed)
                for seed in range(seeds)]
        rows.append({"setting": name, "rollout_priority_rho": rho,
                     "site_specific_covariate_trend": trend,
                     "summary": summarize_runs(runs), "runs": runs})
        print(f"Finished {name}", flush=True)
    return {
        "design": "Same sites in six periods; four site cohorts roll out at periods 2-5. Crossover effects at periods 2-4 are compared with not-yet-deployed controls. 2 x 2 design: rollout priority random or linked to a fixed site characteristic that also affects covariates and outcomes, by common or site-specific covariate trends. Bias intervals are 95% percentile bootstraps over independent seeds.",
        "parameters": {"seeds_per_row": seeds, "sites": sites,
                       "patients_per_site_period": n_per_site_period,
                       "triage_shift_per_covariate": triage,
                       "common_covariate_trend_per_period": 0.10,
                       "root_seed": root_seed, "periods": PERIODS,
                       "cohorts": list(COHORTS)},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--sites", type=int, default=48)
    parser.add_argument("--n-per-site-period", type=int, default=400)
    parser.add_argument("--triage", type=float, default=0.25)
    parser.add_argument("--root-seed", type=int, default=2026092905)
    parser.add_argument("--output", default="merged_expG2.json")
    args = parser.parse_args()
    if args.seeds < 2 or args.sites < 8 or args.sites % 4 or args.n_per_site_period < 2:
        parser.error("Need >=2 seeds, sites divisible by 4 and >=8, and >=2 patients")
    result = run(args.seeds, args.sites, args.n_per_site_period, args.triage, args.root_seed)
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
