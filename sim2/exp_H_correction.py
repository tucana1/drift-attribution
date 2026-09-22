"""E3: rate-held retraining with heterogeneous treatment response.

This adapts the treatment-interaction design in the September 15 WMHS
submission's code/retraining_stress.py to this repository's calibrated DGP.
All five rules use the same independent calibration and evaluation populations
within a seed. Expected event rates share a standard-care reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dgp import p_treat, r_hat, suspicion
from merged import make_cfg


ARMS = ("keep", "naive", "unalerted", "untreated", "conditional_interactions")
ROOT = Path(__file__).resolve().parents[1]


def response(cfg, x, kappa):
    perp = (x[:, 1] - x[:, 0]) / np.sqrt(2)
    lin = x @ cfg.w_y + cfg.b0
    return expit(lin), expit(lin - cfg.theta - kappa * perp)


def fitted_score(x, a, y, arm):
    model = LogisticRegression(max_iter=400)
    if arm == "conditional_interactions":
        features = np.column_stack((x, a, x * a[:, None]))
        model.fit(features, y)
        return lambda xx: model.predict_proba(
            np.column_stack((xx, np.zeros(len(xx)), np.zeros_like(xx)))
        )[:, 1]
    if arm == "unalerted":
        raise ValueError("unalerted subset is selected by the current score")
    selected = a == 0 if arm == "untreated" else np.ones(len(a), dtype=bool)
    model.fit(x[selected], y[selected])
    return lambda xx: model.predict_proba(xx)[:, 1]


def run_seed(kappa, seed, n_train, n_eval, n_cal, rounds, root_seed):
    cfg = make_cfg(deploy=True)
    key = [root_seed, int(kappa * 100), seed]
    ev = np.random.default_rng(np.random.SeedSequence(key + [1])).normal(size=(n_eval, cfg.d))
    cal = np.random.default_rng(np.random.SeedSequence(key + [2])).normal(size=(n_cal, cfg.d))
    q0, q1 = response(cfg, ev, kappa)
    standard_care = float(np.mean(q0 + p_treat(cfg, ev, False) * (q1 - q0)))
    held_rate = float(np.mean(r_hat(cfg, cal) > cfg.tau_r))
    rows = []
    for arm in ARMS:
        score = lambda xx: r_hat(cfg, xx)
        for round_number in range(1, rounds + 1):
            # Fresh independent calibration data are fixed within this seed.
            threshold = float(np.quantile(score(cal), 1 - held_rate))
            risk = score(ev)
            flagged = risk > threshold
            action_probability = np.where(
                suspicion(cfg, ev) > cfg.tau_s,
                cfg.q_hi,
                np.where(flagged, cfg.p_alert, cfg.eps),
            )
            events = float(np.mean(q0 + action_probability * (q1 - q0)))
            rows.append({
                "kappa": kappa, "seed": seed, "round": round_number, "arm": arm,
                "standard_care_events": standard_care, "events": events,
                "events_averted_pp": 100 * (standard_care - events),
                "alert_rate": float(np.mean(flagged)), "threshold": threshold,
                "untreated_risk_mse": float(np.mean((risk - q0) ** 2)),
            })
            if arm == "keep" or round_number == rounds:
                continue
            # Pair patient draws and potential outcomes across update rules.
            rng = np.random.default_rng(np.random.SeedSequence(key + [3, round_number]))
            x = rng.normal(size=(n_train, cfg.d))
            current_risk = score(x)
            current_action_probability = np.where(
                suspicion(cfg, x) > cfg.tau_s,
                cfg.q_hi,
                np.where(current_risk > threshold, cfg.p_alert, cfg.eps),
            )
            action = (rng.random(n_train) < current_action_probability).astype(float)
            train_q0, train_q1 = response(cfg, x, kappa)
            outcome = (rng.random(n_train) < np.where(action == 1, train_q1, train_q0)).astype(int)
            if arm == "unalerted":
                selected = current_risk <= threshold
                model = LogisticRegression(max_iter=400).fit(x[selected], outcome[selected])
                score = lambda xx, model=model: model.predict_proba(xx)[:, 1]
            else:
                score = fitted_score(x, action, outcome, arm)
    return rows


def run(seeds=20, n_train=8000, n_eval=20000, n_cal=20000,
        rounds=8, root_seed=2026092203):
    rows = []
    for kappa in (0.0, 2.0):
        for seed in range(seeds):
            rows.extend(run_seed(kappa, seed, n_train, n_eval, n_cal, rounds, root_seed))
    summary = []
    for kappa in (0.0, 2.0):
        for arm in ARMS:
            for round_number in range(1, rounds + 1):
                cells = [row for row in rows if row["kappa"] == kappa and row["arm"] == arm
                         and row["round"] == round_number]
                values = np.array([row["events_averted_pp"] for row in cells])
                summary.append({
                    "kappa": kappa, "arm": arm, "round": round_number,
                    "mean_events_averted_pp": float(values.mean()),
                    "sd_across_seeds_pp": float(values.std(ddof=1)),
                    "mcse_pp": float(values.std(ddof=1) / np.sqrt(seeds)),
                })
    return {
        "design": "Expected event reduction from standard care on a shared evaluation population; all rules hold the original alert rate on independent calibration data.",
        "parameters": {"seeds": seeds, "n_train": n_train, "n_eval": n_eval,
                       "n_cal": n_cal, "rounds": rounds, "root_seed": root_seed,
                       "kappa": [0.0, 2.0], "arms": list(ARMS)},
        "summary": summary, "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=8000)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--n-cal", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--root-seed", type=int, default=2026092203)
    parser.add_argument("--output", type=Path, default=ROOT / "figures/merged_expH.json")
    args = parser.parse_args()
    if min(args.seeds, args.n_train, args.n_eval, args.n_cal, args.rounds) < 2:
        parser.error("All sizes must be at least two")
    result = run(args.seeds, args.n_train, args.n_eval, args.n_cal, args.rounds, args.root_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output} ({len(result['runs'])} rows)")


if __name__ == "__main__":
    main()
