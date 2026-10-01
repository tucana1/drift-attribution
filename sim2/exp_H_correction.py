"""E3 (Experiment H): update rules after deployment, one standard-care reference.

Question. After the alert has been deployed, the team refits the score on
post-deployment data and redeploys it. Which update rules keep the benefit?

Settings (rows). Each row has its own seed block.
  threshold  fixed: alert when the score exceeds tau_r, the deployed rule.
             rate:  alert at the original alert rate, re-estimated for each
                    score on an independent calibration sample.
  truth      linear, kappa = 0: homogeneous treatment effect on the logit.
             linear, kappa = 2: effect theta + kappa * perp, where perp is the
                    covariate direction orthogonal to the score.
             nonlinear, kappa = 0: the untreated logit gains the E4 term
                    0.65 {X1 X2 + 0.3 (X2^2 - 1)}; the score class stays
                    logistic-linear, so every refit is misspecified.

Arms (paired within a seed: same evaluation, calibration and training draws).
  keep                      original score f0
  naive                     refit on all post-deployment (X, Y)
  unalerted                 refit on patients below the current alert threshold
  untreated                 refit on patients with A = 0
  untreated_ipw             refit on A = 0, weighted by 1 / P(A = 0 | X) from the
                            logged alert rule and response probabilities
  untreated_ipw_est         the same with P(A = 1 | X) estimated by a logistic
                            regression of A on X, the alert and their
                            interaction (no response probabilities needed;
                            estimates capped at 0.95)
  conditional_main          refit on (X, A), predict at A = 0
  conditional_interactions  refit on (X, A, X * A), predict at A = 0

The original score f0 is the untreated risk in the linear rows. In the
nonlinear row it is the population-best logistic approximation to the untreated
risk, so a correct and stable update reproduces it.

Metrics per round on the shared evaluation population, without outcome noise:
expected events and events averted against standard care, alert rate,
treatments, unnecessary treatments (treated with untreated outcome 0),
observed AUROC and Brier score in the deployed population (what a dashboard
reports), and calibration of the score against the untreated risk.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import columns, expected_auroc, expected_brier, rng_for, summarize, write_json
from dgp import r_hat, suspicion
from merged import make_cfg

ARMS = ("keep", "naive", "unalerted", "untreated", "untreated_ipw",
        "conditional_main", "conditional_interactions", "untreated_ipw_est")
SETTINGS = (
    # name, threshold rule, kappa, nonlinear coefficient
    ("fixed_k0", "fixed", 0.0, 0.0),
    ("fixed_k2", "fixed", 2.0, 0.0),
    ("fixed_nl", "fixed", 0.0, 0.65),
    ("rate_k0", "rate", 0.0, 0.0),
    ("rate_k2", "rate", 2.0, 0.0),
    ("rate_nl", "rate", 0.0, 0.65),
)
EXPERIMENT = 3  # seed-block id for E3
HARM_GRID = (0.0, 0.05, 0.1, 0.2)


def untreated_logit(cfg, x, nonlinear):
    """Logit of P(Y = 1 | X, A = 0); mirrors dgp.p_y1 for d = 2."""
    lin = x @ cfg.w_y + cfg.b0
    if nonlinear:
        lin = lin + nonlinear * (x[:, 0] * x[:, 1] + 0.3 * (x[:, 1] ** 2 - 1))
    return lin


def response(cfg, x, kappa, nonlinear):
    """Untreated and treated event probabilities."""
    perp = (x[:, 1] - x[:, 0]) / np.sqrt(2)
    lin = untreated_logit(cfg, x, nonlinear)
    return expit(lin), expit(lin - cfg.theta - kappa * perp)


def treat_probability(cfg, x, score, threshold, deployed=True):
    """Clinician policy: suspicious patients get q_hi; alerted others p_alert; rest eps."""
    alert = (score > threshold) if deployed else np.zeros(len(x), bool)
    return np.where(suspicion(cfg, x) > cfg.tau_s, cfg.q_hi,
                    np.where(alert, cfg.p_alert, cfg.eps))


def original_score(cfg, nonlinear, root):
    if not nonlinear:
        return lambda xx: r_hat(cfg, xx)
    # Population-best logistic approximation to the untreated risk.
    rng = rng_for(root, EXPERIMENT, 99, 0)
    x = rng.normal(size=(400_000, 2))
    y = (rng.random(len(x)) < expit(untreated_logit(cfg, x, nonlinear))).astype(int)
    model = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
    return lambda xx: model.predict_proba(xx)[:, 1]


def refit(arm, x, action, outcome, current_score, threshold, action_probability):
    model = LogisticRegression(C=1e6, max_iter=5000)
    if arm == "conditional_interactions":
        model.fit(np.column_stack((x, action, x * action[:, None])), outcome)
        return lambda xx: model.predict_proba(
            np.column_stack((xx, np.zeros(len(xx)), np.zeros_like(xx))))[:, 1]
    if arm == "conditional_main":
        model.fit(np.column_stack((x, action)), outcome)
        return lambda xx: model.predict_proba(np.column_stack((xx, np.zeros(len(xx)))))[:, 1]
    if arm == "untreated_ipw":
        keep = action == 0
        model.fit(x[keep], outcome[keep], sample_weight=1.0 / (1.0 - action_probability[keep]))
        return lambda xx: model.predict_proba(xx)[:, 1]
    if arm == "untreated_ipw_est":
        alert = (current_score > threshold).astype(float)
        design = np.column_stack((x, alert, x * alert[:, None]))
        propensity = LogisticRegression(C=1e6, max_iter=5000).fit(design, action)
        p_hat = np.minimum(propensity.predict_proba(design)[:, 1], 0.95)
        keep = action == 0
        model.fit(x[keep], outcome[keep], sample_weight=1.0 / (1.0 - p_hat[keep]))
        return lambda xx: model.predict_proba(xx)[:, 1]
    keep = {"naive": np.ones(len(action), bool), "untreated": action == 0,
            "unalerted": current_score <= threshold}[arm]
    model.fit(x[keep], outcome[keep])
    return lambda xx: model.predict_proba(xx)[:, 1]


def run_seed(setting_index, setting, seed, args, f0):
    name, rule, kappa, nonlinear = setting
    cfg = make_cfg(deploy=True)
    rng = rng_for(args.root_seed, EXPERIMENT, setting_index, seed, 0)
    ev = rng.normal(size=(args.n_eval, cfg.d))
    cal = rng.normal(size=(args.n_cal, cfg.d))
    q0, q1 = response(cfg, ev, kappa, nonlinear)
    p_std = treat_probability(cfg, ev, None, None, deployed=False)
    std_events = float(np.mean(q0 + p_std * (q1 - q0)))
    std_unnecessary = float(np.mean(p_std * (1 - q0)))
    held_rate = float(np.mean(f0(cal) > cfg.tau_r))
    rows = []
    for arm in ARMS:
        score = f0
        for rnd in range(1, args.rounds + 1):
            threshold = cfg.tau_r if rule == "fixed" else float(np.quantile(score(cal), 1 - held_rate))
            s = score(ev)
            p = treat_probability(cfg, ev, s, threshold)
            q_obs = q0 + p * (q1 - q0)
            events = float(np.mean(q_obs))
            rows.append({
                "setting": name, "seed": seed, "arm": arm, "round": rnd,
                "threshold": threshold, "alert_rate": float(np.mean(s > threshold)),
                "events": events, "standard_care_events": std_events,
                "events_averted_pp": 100 * (std_events - events),
                "treated": float(np.mean(p)),
                "unnecessary_treated": float(np.mean(p * (1 - q0))),
                "standard_care_unnecessary": std_unnecessary,
                "observed_auroc": expected_auroc(s, q_obs),
                "observed_brier": expected_brier(s, q_obs),
                "untreated_auroc": expected_auroc(s, q0),
                "calibration_in_large": float(np.mean(s) - np.mean(q0)),
                "untreated_risk_mse": float(np.mean((s - q0) ** 2)),
            })
            if arm == "keep" or rnd == args.rounds:
                continue
            # Common random numbers across arms: same covariates and uniforms.
            train = rng_for(args.root_seed, EXPERIMENT, setting_index, seed, rnd)
            x = train.normal(size=(args.n_train, cfg.d))
            u_action, u_outcome = train.random(args.n_train), train.random(args.n_train)
            current = score(x)
            pa = treat_probability(cfg, x, current, threshold)
            action = (u_action < pa).astype(float)
            t0, t1 = response(cfg, x, kappa, nonlinear)
            outcome = (u_outcome < np.where(action == 1, t1, t0)).astype(int)
            score = refit(arm, x, action, outcome, current, threshold, pa)
    return rows


def summarise(rows, args):
    summary, contrasts = [], []
    for index, (name, *_rest) in enumerate(SETTINGS):
        for arm in ARMS:
            for rnd in range(1, args.rounds + 1):
                cell = [r for r in rows if r["setting"] == name and r["arm"] == arm and r["round"] == rnd]
                entry = {"setting": name, "arm": arm, "round": rnd}
                for key in ("events_averted_pp", "alert_rate", "observed_auroc", "observed_brier",
                            "calibration_in_large", "untreated_risk_mse"):
                    entry[key] = summarize([r[key] for r in cell], seed=rnd)
                summary.append(entry)
        final = args.rounds
        per_seed = {}
        for r in rows:
            if r["setting"] == name and r["round"] == final:
                per_seed.setdefault(r["arm"], {})[r["seed"]] = r
        seeds = sorted(per_seed["keep"])
        for arm in ARMS[1:]:
            for ref in ("keep", "naive"):
                if arm == ref:
                    continue
                diff = [per_seed[arm][s]["events_averted_pp"] - per_seed[ref][s]["events_averted_pp"] for s in seeds]
                contrasts.append({"setting": name, "arm": arm, "reference": ref, "round": final,
                                  "paired_difference_pp": summarize(diff, seed=index)})
        # Harm sensitivity (E7): net events averted, penalising extra unnecessary treatments.
        for arm in ARMS:
            for h in HARM_GRID:
                vals = [100 * ((per_seed[arm][s]["standard_care_events"] - per_seed[arm][s]["events"])
                               - h * (per_seed[arm][s]["unnecessary_treated"] - per_seed[arm][s]["standard_care_unnecessary"]))
                        for s in seeds]
                contrasts.append({"setting": name, "arm": arm, "reference": "standard_care_harm",
                                  "harm_per_unnecessary_treatment": h, "round": final,
                                  "net_events_averted_pp": summarize(vals, seed=index)})
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--n-cal", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--root-seed", type=int, default=2026092903)
    parser.add_argument("--output", default="merged_expH.json")
    args = parser.parse_args()
    if min(args.seeds, args.n_train, args.n_eval, args.n_cal, args.rounds) < 2:
        parser.error("All sizes must be at least two")
    start, rows = time.time(), []
    for index, setting in enumerate(SETTINGS):
        cfg = make_cfg(deploy=True)
        f0 = original_score(cfg, setting[3], args.root_seed)
        for seed in range(args.seeds):
            rows.extend(run_seed(index, setting, seed, args, f0))
        print(f"finished {setting[0]} ({time.time() - start:.0f}s)", flush=True)
    summary, contrasts = summarise(rows, args)
    keys = list(rows[0].keys())
    result = {
        "design": ("Update rules after deployment. Expected events on a shared evaluation population "
                   "against a standard-care reference; fixed or rate-held alert threshold; homogeneous "
                   "(kappa=0) or heterogeneous (kappa=2) treatment effect; well-specified or misspecified "
                   "score class. Intervals are 95% percentile bootstraps over independent seeds."),
        "parameters": {"seeds": args.seeds, "n_train": args.n_train, "n_eval": args.n_eval,
                       "n_cal": args.n_cal, "rounds": args.rounds, "root_seed": args.root_seed,
                       "experiment_block": EXPERIMENT, "arms": list(ARMS),
                       "settings": [dict(zip(("name", "threshold", "kappa", "nonlinear"), s)) for s in SETTINGS],
                       "harm_grid": list(HARM_GRID)},
        "summary": summary, "contrasts": contrasts, "runs": columns(rows, keys),
    }
    write_json(args.output, result)
    print(f"wrote {args.output}: {len(rows)} rows in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
