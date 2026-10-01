"""E4: repeat Experiments A and B with more covariates and outcome misspecification.

The deployed score remains logistic-linear. In misspecified cells, the true
untreated risk also depends on X_1*X_3 and a quadratic term. Oracle rows use
the simulator's mechanisms; they are not available from routine logs.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import write_json
from dgp import p_treat, p_y1, r_hat, sample_env, shapley, is_values, mu_of
from merged import SCEN, E3, make_cfg, naive2_est, sample_po, truth_E3_decomp


ROOT = Path(__file__).resolve().parents[1]
PLAYERS = ("X", "pi", "Y")
EPS = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)


def seed_for(root, d, nonlinear, experiment, row, repetition):
    # Independent seed blocks across every setting/row; only within-row
    # estimators share patients. This is needed for across-row bias claims.
    return np.random.SeedSequence([root, d, int(100 * nonlinear), experiment, row, repetition])


def expected_brier(cfg, x, coalition):
    q0 = p_y1(cfg, x, np.zeros(len(x)), "Y" in coalition)
    q1 = p_y1(cfg, x, np.ones(len(x)), "Y" in coalition)
    p = p_treat(cfg, x, "pi" in coalition and cfg.deploy)
    q = q0 + p * (q1 - q0)
    score = r_hat(cfg, x)
    return float(np.mean(score ** 2 + (1 - 2 * score) * q))


def truth_values_expected(cfg, n, seed):
    rng = np.random.default_rng(seed)
    # Common normal draws across coalitions reduce numerical error in contrasts.
    base_x = rng.normal(size=(n, cfg.d))
    values = {}
    for k in range(4):
        for S in itertools.combinations(PLAYERS, k):
            coalition = frozenset(S)
            x = base_x + mu_of(cfg, None, coalition)
            values[coalition] = expected_brier(cfg, x, coalition)
    return values


def summarize(values, target=None):
    a = np.asarray(values, dtype=float)
    result = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
              "mcse": float(a.std(ddof=1) / np.sqrt(len(a)))}
    if target is not None:
        result["bias"] = float(a.mean() - target)
    return result


def experiment_a(d, nonlinear, seeds, n, n_truth, root):
    rows = []
    for index, (name, scenario) in enumerate(SCEN.items()):
        setting = make_cfg(d=d, nonlinear_outcome=nonlinear, **scenario)
        truth = truth_values_expected(setting, n_truth,
                                      seed_for(root, d, nonlinear, 1, index, 999))
        truth_shapley = shapley(truth, PLAYERS)
        truth_policy = truth_E3_decomp(truth)
        runs = []
        for repetition in range(seeds):
            seed = seed_for(root, d, nonlinear, 1, index, repetition)
            rng = np.random.default_rng(seed)
            s0 = sample_po(setting, set(), n, rng)
            s1 = sample_po(setting, set(PLAYERS), n, rng, control_frac=0.3)
            oracle_seed = seed_for(root, d, nonlinear, 11, index, repetition)
            raw = shapley(is_values(setting, PLAYERS, n=n, seed=oracle_seed)[0], PLAYERS)
            clipped = shapley(is_values(setting, PLAYERS, n=n, seed=oracle_seed,
                                        clip=100)[0], PLAYERS)
            runs.append({"naive2_est": naive2_est(setting, s0, s1),
                         "E3_rollout": E3(setting, s0, s1, "rollout"),
                         "union3_raw": raw, "union3_clip": clipped})
        rows.append({
            "scenario": name, "truth_shapley": truth_shapley,
            "truth_policy_contrast": truth_policy,
            "summary": {
                "naive2_Ygx": summarize([r["naive2_est"]["Ygx"] for r in runs],
                                          truth_policy["dR"] if name == "S3" else None),
                "E3_pi": summarize([r["E3_rollout"]["pi"] for r in runs], truth_policy["pi"]),
                "union3_pi": summarize([r["union3_raw"]["pi"] for r in runs], truth_shapley["pi"]),
            }, "runs": runs,
        })
    return rows


def experiment_b(d, nonlinear, seeds, n, n_truth, root):
    rows = []
    for index, eps in enumerate(EPS):
        cfg = make_cfg(d=d, nonlinear_outcome=nonlinear, eps=eps, deploy=True)
        truth = truth_values_expected(cfg, n_truth,
                                      seed_for(root, d, nonlinear, 2, index, 999))
        truth_pi = truth_E3_decomp(truth)["pi"]
        truth_shapley_pi = shapley(truth, PLAYERS)["pi"]
        runs = []
        for repetition in range(seeds):
            seed = seed_for(root, d, nonlinear, 2, index, repetition)
            rng = np.random.default_rng(seed)
            s0 = sample_po(cfg, set(), n, rng)
            s1 = sample_po(cfg, set(PLAYERS), n, rng, control_frac=0.3)
            oracle_rng = np.random.default_rng(seed_for(root, d, nonlinear, 22, index, repetition))
            x, a, y = sample_env(cfg, set(), n, oracle_rng)
            # Same sample for raw and clipped oracle mechanism weights.
            from dgp import _weights_union3, brier
            def oracle(clip):
                values = {}
                for size in range(4):
                    for S in itertools.combinations(PLAYERS, size):
                        w, _ = _weights_union3(cfg, x, a, y, set(S), clip)
                        values[frozenset(S)] = brier(cfg, x, y, w)
                return shapley(values, PLAYERS)["pi"]
            p0 = p_treat(cfg, x, False)
            p1 = p_treat(cfg, x, True)
            wpi = np.where(a == 1, p1, 1 - p1) / np.where(a == 1, p0, 1 - p0)
            ess = float(wpi.sum() ** 2 / np.sum(wpi ** 2))
            runs.append({"union3_raw_pi": oracle(None), "union3_clip_pi": oracle(100),
                         "E3_rollout_pi": E3(cfg, s0, s1, "rollout")["pi"], "ess": ess})
        rows.append({
            "eps": eps, "truth_policy_contrast_pi": truth_pi,
            "truth_shapley_pi": truth_shapley_pi,
            "summary": {
                "union3_raw_pi": summarize([r["union3_raw_pi"] for r in runs], truth_shapley_pi),
                "union3_clip_pi": summarize([r["union3_clip_pi"] for r in runs], truth_shapley_pi),
                "E3_rollout_pi": summarize([r["E3_rollout_pi"] for r in runs], truth_pi),
                "ess": summarize([r["ess"] for r in runs], n),
            }, "runs": runs,
        })
    return rows


def n_scaling(d, nonlinear, seeds, n_truth, root, sizes):
    cfg = make_cfg(d=d, nonlinear_outcome=nonlinear, eps=0, deploy=True)
    truth = truth_values_expected(cfg, n_truth, seed_for(root, d, nonlinear, 3, 0, 999))
    target = shapley(truth, PLAYERS)["pi"]
    rows = []
    for index, n in enumerate(sizes):
        values = []
        for repetition in range(seeds):
            seed = seed_for(root, d, nonlinear, 3, index, repetition)
            values.append(shapley(is_values(cfg, PLAYERS, n=n, seed=seed)[0], PLAYERS)["pi"])
        rows.append({"n": n, "truth_shapley_pi": target,
                     "union3_raw_pi": summarize(values, target), "runs": values})
    return rows


def run(seeds=12, n=12000, n_truth=120000, root=2026092204, sizes=(6000, 24000, 96000)):
    result = {"design": "A and B repeated at d=10,20; linear and nonlinear outcome truth; logistic-linear deployed score. Oracle mechanism weights are simulation-only.",
              "parameters": {"seeds_per_row": seeds, "n_per_environment": n,
                             "n_truth": n_truth, "root_seed": root, "n_scaling": list(sizes),
                             "randomized_control_fraction": 0.3}, "settings": []}
    for d in (10, 20):
        for nonlinear in (0.0, 0.65):
            result["settings"].append({
                "d": d, "nonlinear_outcome": nonlinear,
                "A": experiment_a(d, nonlinear, seeds, n, n_truth, root),
                "B": experiment_b(d, nonlinear, seeds, n, n_truth, root),
                "n_scaling_eps_zero": n_scaling(d, nonlinear, seeds, n_truth, root, sizes),
            })
            print(f"Finished d={d}, nonlinear={nonlinear}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--n", type=int, default=12000)
    parser.add_argument("--n-truth", type=int, default=120000)
    parser.add_argument("--root-seed", type=int, default=2026092204)
    parser.add_argument("--output", default="merged_expAB_robustness.json", help="file name in figures/ and aaai/figures/")
    args = parser.parse_args()
    if min(args.seeds, args.n, args.n_truth) < 2:
        parser.error("Sample sizes and seeds must be at least two")
    result = run(args.seeds, args.n, args.n_truth, args.root_seed)
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
