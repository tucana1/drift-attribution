"""E7: intervals for every main-text estimate, analyst-facing bootstrap, harm sensitivity.

Reruns the main-text experiments at the calibrated operating point
(merged.make_cfg) with one independent seed block per row and stores every
replicate, so each reported mean carries a 95% percentile bootstrap interval
over replicates (Monte Carlo uncertainty of the simulation mean).

  A   attribution under S1-S4 (Figure 1, F1, the two-references paragraph)
  B   positivity continuum and n scaling at eps = 0 (Figure 2, F2)
  D   AUROC, events averted, standard and counterfactual net benefit (Figure 5, F6)
  E   threshold discontinuity (F6, appendix)
  F   deployment-caused covariate shift (Figure 4, F5)

Threshold discontinuity. The jump at tau identifies the local effect of the
alert policy, E[Y(pi1) - Y(pi0) | r = tau]. Its global counterpart is the
policy effect among the alerted, E[Y(pi1) - Y(pi0) | r > tau] (expected
values given X), not the effect of treatment against no treatment: alerted
patients who are also suspicious are treated with probability q_hi under both
policies. The treatment effect among the alerted is kept for reference.

Net benefit. The counterfactual net benefit uses the untreated outcome Y(0),
which no arm observes. The randomised unalerted arm identifies net benefit
against outcomes under standard care, Y(pi0), reported as nb_standard_care
(expected value given X).

Analyst-facing intervals. For A (S1-S4) and for B, each replicate also gets a
patient-level bootstrap (resampling pre- and post-deployment patients, the
post-deployment sample stratified by randomised arm) for the estimators an
analyst can compute. Coverage is measured against each estimator's own
estimand: the policy-contrast decomposition for the proposed estimator, the
marginal two-player game for the monitor, the Shapley game with oracle
coalition values for the union-graph estimator.

Oracle references use expected Brier scores (no outcome noise) with common
covariate draws across coalitions, averaged over independent blocks whose
spread gives their Monte Carlo error.

Harm. Net events averted = events averted - h x (extra treatments given to
patients whose untreated outcome is 0), for h in HARM_GRID.
"""
from __future__ import annotations

import os

# One BLAS thread per process: the replicates run in a process pool.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, rng_for, seed_block, summarize, write_json
from dgp import (_weights_union3, brier, is_values, mu_of, naive2_values, p_treat, p_y1, r_hat,
                 sample_env, shapley, suspicion)
from exp_AB_robustness import truth_values_expected
from exp_F_triage import union3_analyst
from merged import E3, SCEN, make_cfg, make_cfg_at, naive2_est, sample_po, truth_E3_decomp

EXPERIMENT = 7
P3 = ("X", "pi", "Y")
HARM_GRID = (0.0, 0.05, 0.1, 0.2)
EPS_GRID = (0.0, 1e-4, 1e-3, 1e-2, 5e-2)
N_SCALING = (20_000, 50_000, 100_000, 200_000, 400_000)


# ----------------------------------------------------------------------------- references
def oracle_tables(cfg, n, blocks, root, key):
    """Expected-Brier coalition tables over independent blocks."""
    tables = [truth_values_expected(cfg, n, seed_block(root, EXPERIMENT, 900, key, b)) for b in range(blocks)]
    return tables


def reference_summary(tables):
    """Shapley and policy-contrast references with Monte Carlo SE across blocks."""
    rows = []
    for tv in tables:
        sh = shapley(tv, list(P3))
        pc = truth_E3_decomp(tv)
        mc = {"to_empty": tv[frozenset({"pi"})] - tv[frozenset()],
              "to_X": tv[frozenset({"X", "pi"})] - tv[frozenset({"X"})],
              "to_Y": tv[frozenset({"pi", "Y"})] - tv[frozenset({"Y"})],
              "to_XY": tv[frozenset(P3)] - tv[frozenset({"X", "Y"})]}
        rows.append({"shapley_X": sh["X"], "shapley_pi": sh["pi"], "shapley_Y": sh["Y"],
                     "pc_X": pc["X"], "pc_Y": pc["Y"], "pc_pi": pc["pi"], "dR": pc["dR"],
                     "I_Xpi": mc["to_X"] - mc["to_empty"], "I_piY": mc["to_Y"] - mc["to_empty"], **mc})
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = {"value": float(v.mean()), "mc_se": float(v.std(ddof=1) / np.sqrt(len(v)))}
    return out


def naive2_reference(cfg, n, rng):
    """Population values of the marginal two-player game the monitor estimates."""
    base = rng.normal(size=(n, cfg.d))
    values = {}
    for S in ((), ("X",), ("Ygx",), ("X", "Ygx")):
        x = base + (mu_of(cfg, None, {"X", "pi"}) if "X" in S else 0.0)
        shifted = "Ygx" in S
        p = p_treat(cfg, x, shifted and cfg.deploy)
        q0 = p_y1(cfg, x, np.zeros(n), shifted)
        q1 = p_y1(cfg, x, np.ones(n), shifted)
        q = q0 + p * (q1 - q0)
        r = r_hat(cfg, x)
        values[frozenset(S)] = float(np.mean(r ** 2 + (1 - 2 * r) * q))
    return shapley(values, ["X", "Ygx"])


# ----------------------------------------------------------------------------- estimators with bootstrap
def union3_from_sample(cfg, x, a, y, clip=None):
    """Union-graph importance-sampling Shapley values on a given e0 sample, plus reusable weights."""
    loss = (r_hat(cfg, x) - y) ** 2
    weights = {}
    for k in range(4):
        for S in itertools.combinations(P3, k):
            w, _ = _weights_union3(cfg, x, a, y, set(S), clip=clip)
            weights[frozenset(S)] = w
    return loss, weights


def union3_values(loss, weights, idx=None):
    if idx is None:
        return {S: float(np.sum(w * loss) / np.sum(w)) for S, w in weights.items()}
    return {S: float(np.sum(w[idx] * loss[idx]) / np.sum(w[idx])) for S, w in weights.items()}


def take(sample, idx):
    return {k: v[idx] for k, v in sample.items()}


def stratified_indices(rng, arm):
    parts = [rng.choice(np.flatnonzero(arm == g), size=int(np.sum(arm == g)), replace=True)
             for g in np.unique(arm)]
    return np.concatenate(parts)


def percentile(draws, level=0.95):
    a = np.asarray(draws, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return [float("nan"), float("nan")]
    return [float(v) for v in np.quantile(a, [(1 - level) / 2, 1 - (1 - level) / 2])]


def analyst_bootstrap(cfg, s0, s1, x_is, a_is, y_is, boots, rng):
    """Patient-level bootstrap intervals for one replicate."""
    draws = {k: [] for k in ("E3_X", "E3_Y", "E3_pi", "n2_X", "n2_Ygx", "u_raw_pi", "u_clip_pi")}
    loss, w_raw = union3_from_sample(cfg, x_is, a_is, y_is, clip=None)
    _, w_clip = union3_from_sample(cfg, x_is, a_is, y_is, clip=100.0)
    n0, n_is = len(s0["y"]), len(y_is)
    for _ in range(boots):
        b0 = take(s0, rng.integers(0, n0, n0))
        b1 = take(s1, stratified_indices(rng, s1["arm"]))
        e3 = E3(cfg, b0, b1, "rollout")
        n2 = naive2_est(cfg, b0, b1)
        idx = rng.integers(0, n_is, n_is)
        with np.errstate(divide="ignore", invalid="ignore"):
            ur = shapley(union3_values(loss, w_raw, idx), list(P3))["pi"]
            uc = shapley(union3_values(loss, w_clip, idx), list(P3))["pi"]
        for k, v in (("E3_X", e3["X"]), ("E3_Y", e3["Y"]), ("E3_pi", e3["pi"]),
                     ("n2_X", n2["X"]), ("n2_Ygx", n2["Ygx"]), ("u_raw_pi", ur), ("u_clip_pi", uc)):
            draws[k].append(v)
    return {k: percentile(v) for k, v in draws.items()}


# ----------------------------------------------------------------------------- A
def scenario_cfg(scenario, args):
    """Scenario at the default operating point, or at --b0 / --alert-rate (realistic prevalence)."""
    return make_cfg_at(args.b0, args.alert_rate, **SCEN[scenario])


def a_replicate(job):
    s_index, scenario, rep, args = job
    cfg = scenario_cfg(scenario, args)
    rng = rng_for(args.root_seed, EXPERIMENT, 1, s_index, rep)
    s0 = sample_po(cfg, set(), args.n, rng)
    s1 = sample_po(cfg, set(P3), args.n, rng, control_frac=0.3)
    x_is, a_is, y_is = sample_env(cfg, set(), args.n, rng)       # the union-graph estimator's e0 sample
    loss, w_raw = union3_from_sample(cfg, x_is, a_is, y_is)
    _, w_clip = union3_from_sample(cfg, x_is, a_is, y_is, clip=100.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_raw = shapley(union3_values(loss, w_raw), list(P3))
        u_clip = shapley(union3_values(loss, w_clip), list(P3))
    p1 = p_treat(cfg, x_is, cfg.deploy)
    p0 = p_treat(cfg, x_is, False)
    wpi = np.where(a_is == 1, p1, 1 - p1) / np.where(a_is == 1, p0, 1 - p0)
    ess = float(wpi.sum() ** 2 / np.sum(wpi ** 2))
    n2_oracle = shapley(naive2_values(cfg, n=args.n, metric="brier",
                                      seed=seed_block(args.root_seed, EXPERIMENT, 11, s_index, rep)), ["X", "Ygx"])
    e3r, e3o, n2 = E3(cfg, s0, s1, "rollout"), E3(cfg, s0, s1, "oracle"), naive2_est(cfg, s0, s1)
    out = {"scenario": scenario, "rep": rep, "ess": ess,
           "dR_observed": float(np.mean((r_hat(cfg, s1["x"][s1["arm"] == 1]) - s1["y"][s1["arm"] == 1]) ** 2)
                                - np.mean((r_hat(cfg, s0["x"]) - s0["y"]) ** 2))}
    for name, dec in (("union3_raw", u_raw), ("union3_clip", u_clip), ("E3_rollout", e3r), ("E3_oracle", e3o)):
        for k in ("X", "pi", "Y"):
            out[f"{name}_{k}"] = float(dec[k])
    out["naive2_est_X"], out["naive2_est_Ygx"] = float(n2["X"]), float(n2["Ygx"])
    out["naive2_oracle_X"], out["naive2_oracle_Ygx"] = float(n2_oracle["X"]), float(n2_oracle["Ygx"])
    if rep < args.boot_reps_a:
        brng = rng_for(args.root_seed, EXPERIMENT, 12, s_index, rep)
        out["bootstrap"] = analyst_bootstrap(cfg, s0, s1, x_is, a_is, y_is, args.boots, brng)
    return out


# ----------------------------------------------------------------------------- B
def b_replicate(job):
    e_index, eps, rep, args = job
    cfg = make_cfg(eps=eps, deploy=True)
    rng = rng_for(args.root_seed, EXPERIMENT, 2, e_index, rep)
    s0 = sample_po(cfg, set(), args.n, rng)
    s1 = sample_po(cfg, set(P3), args.n, rng, control_frac=0.3)
    x_is, a_is, y_is = sample_env(cfg, set(), args.n, rng)
    loss, w_raw = union3_from_sample(cfg, x_is, a_is, y_is)
    _, w_clip = union3_from_sample(cfg, x_is, a_is, y_is, clip=100.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_raw = shapley(union3_values(loss, w_raw), list(P3))["pi"]
        u_clip = shapley(union3_values(loss, w_clip), list(P3))["pi"]
    p1, p0 = p_treat(cfg, x_is, True), p_treat(cfg, x_is, False)
    with np.errstate(divide="ignore", invalid="ignore"):
        wpi = np.where(a_is == 1, p1, 1 - p1) / np.where(a_is == 1, p0, 1 - p0)
        ess = float(wpi.sum() ** 2 / np.sum(wpi ** 2)) if np.isfinite(wpi).all() else float("nan")
    out = {"eps": eps, "rep": rep, "union3_raw_pi": float(u_raw), "union3_clip_pi": float(u_clip),
           "ess": ess, "E3_rollout_pi": float(E3(cfg, s0, s1, "rollout")["pi"])}
    if rep < args.boot_reps_b:
        brng = rng_for(args.root_seed, EXPERIMENT, 22, e_index, rep)
        bt = analyst_bootstrap(cfg, s0, s1, x_is, a_is, y_is, args.boots, brng)
        out["bootstrap"] = {k: bt[k] for k in ("E3_pi", "u_raw_pi", "u_clip_pi")}
    return out


def n_scaling_replicate(job):
    n_index, n, rep, args = job
    cfg = make_cfg(eps=0.0, deploy=True)
    rng = rng_for(args.root_seed, EXPERIMENT, 3, n_index, rep)
    x, a, y = sample_env(cfg, set(), n, rng)
    loss, w_raw = union3_from_sample(cfg, x, a, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = shapley(union3_values(loss, w_raw), list(P3))["pi"]
    alert_only = (r_hat(cfg, x) > cfg.tau_r) & (suspicion(cfg, x) <= cfg.tau_s)
    draws = []
    brng = rng_for(args.root_seed, EXPERIMENT, 33, n_index, rep)
    for _ in range(args.boots_n):
        idx = brng.integers(0, n, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            draws.append(shapley(union3_values(loss, w_raw, idx), list(P3))["pi"])
    return {"n": n, "rep": rep, "union3_raw_pi": float(est), "bootstrap_ci95": percentile(draws),
            "alert_only_patients": int(alert_only.sum()), "alert_only_treated": int((alert_only & (a == 1)).sum())}


# ----------------------------------------------------------------------------- D, E, F
def d_replicate(job):
    c_index, theta, pa, rep, args = job
    cfg = make_cfg(theta=theta, p_alert=pa, deploy=True)
    rng = rng_for(args.root_seed, EXPERIMENT, 4, c_index, rep)
    s0 = sample_po(cfg, set(), args.n_d, rng)
    s1 = sample_po(cfg, {"pi"}, args.n_d, rng)
    t = cfg.tau_r
    flag = s1["r"] > t
    nb_std = np.mean(flag & (s1["y"] == 1)) - np.mean(flag & (s1["y"] == 0)) * t / (1 - t)
    nb_cf = np.mean(flag & (s1["y0"] == 1)) - np.mean(flag & (s1["y0"] == 0)) * t / (1 - t)
    # expected events and unnecessary treatments on the e1 covariates (no outcome noise)
    x = s1["x"]
    q0, q1 = p_y1(cfg, x, np.zeros(len(x)), False), p_y1(cfg, x, np.ones(len(x)), False)
    p0, p1 = p_treat(cfg, x, False), p_treat(cfg, x, True)
    q_std = q0 + p0 * (q1 - q0)                 # event probability under standard care, Y(pi0)
    nb_standard_care = np.mean(flag * q_std) - np.mean(flag * (1 - q_std)) * t / (1 - t)
    return {"theta": theta, "p_alert": pa, "rep": rep,
            "auroc_e0": float(roc_auc_score(s0["y"], s0["r"])), "auroc_e1": float(roc_auc_score(s1["y"], s1["r"])),
            "auroc_cf": float(roc_auc_score(s1["y0"], s1["r"])), "averted_observed": float(s0["y"].mean() - s1["y"].mean()),
            "averted_expected": float(np.mean(p1 * (q0 - q1)) - np.mean(p0 * (q0 - q1))),
            "extra_unnecessary": float(np.mean((p1 - p0) * (1 - q0))),
            "nb_std": float(nb_std), "nb_cf": float(nb_cf), "nb_standard_care": float(nb_standard_care)}


def rd_grid(cfg):
    """Score bins above the alert threshold for the policy-effect curve."""
    return np.linspace(cfg.tau_r, 0.95, 13)


def e_replicate(job):
    rep, args = job
    cfg = make_cfg(deploy=True)
    rng = rng_for(args.root_seed, EXPERIMENT, 5, 0, rep)
    s1 = sample_po(cfg, {"pi"}, args.n_e, rng)
    x, r = s1["x"], s1["r"]
    alerted = r > cfg.tau_r
    q0, q1 = p_y1(cfg, x, np.zeros(len(x)), False), p_y1(cfg, x, np.ones(len(x)), False)
    policy = (s1["p1"] - s1["p0"]) * (q1 - q0)          # E[Y(pi1) - Y(pi0) | x]
    h = 0.04
    m = np.abs(r - cfg.tau_r) < h
    rr = r[m] - cfg.tau_r
    side = (rr > 0).astype(float)
    coef, *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), rr, side, rr * side], s1["y"][m], rcond=None)
    grid = rd_grid(cfg)
    curve = [float(np.mean(policy[alerted & (r >= a) & (r < b)])) if np.sum(alerted & (r >= a) & (r < b)) > 30
             else float("nan") for a, b in zip(grid[:-1], grid[1:])]
    return {"rep": rep, "policy_effect_alerted": float(np.mean(policy[alerted])),
            "treatment_effect_alerted": float(np.mean((s1["y"] - s1["y0"])[alerted])),
            "rd_local": float(coef[2]), "policy_effect_curve": curve}


def prospective_expected(cfg, n, rng):
    x = rng.normal(size=(n, cfg.d)) + mu_of(cfg, None, {"X", "pi"})
    shifted = cfg.dy != 0
    q0, q1 = p_y1(cfg, x, np.zeros(n), shifted), p_y1(cfg, x, np.ones(n), shifted)
    r = r_hat(cfg, x)
    out = {}
    for lab, deployed in (("pi1", True), ("pi0", False)):
        q = q0 + p_treat(cfg, x, deployed and cfg.deploy) * (q1 - q0)
        out[lab] = float(np.mean(r ** 2 + (1 - 2 * r) * q))
    return out["pi1"] - out["pi0"]


def f_replicate(job):
    row_index, setting, triage, kw, rep, args = job
    cfg = make_cfg(triage=triage, deploy=True, **kw)
    rng = rng_for(args.root_seed, EXPERIMENT, 6, row_index, rep)
    s0 = sample_po(cfg, set(), args.n, rng)
    s1 = sample_po(cfg, set(P3), args.n, rng, control_frac=0.3)
    seed_or = seed_block(args.root_seed, EXPERIMENT, 61, row_index, rep)
    uo = shapley(is_values(cfg, list(P3), n=args.n, metric="brier", seed=seed_or)[0], list(P3))
    ua = union3_analyst(cfg, n=args.n, seed=seed_block(args.root_seed, EXPERIMENT, 62, row_index, rep))
    e3 = E3(cfg, s0, s1, "rollout")
    n2 = naive2_est(cfg, s0, s1)
    return {"setting": setting, "triage": triage, "rep": rep, "uo_X": uo["X"], "uo_pi": uo["pi"],
            "ua_X": ua["X"], "ua_pi": ua["pi"], "e3_X": e3["X"], "e3_Y": e3["Y"], "e3_pi": e3["pi"],
            "n2_X": float(n2["X"]), "n2_Ygx": float(n2["Ygx"])}


# ----------------------------------------------------------------------------- driver
def pmap(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def coverage(rows, key, target):
    """Share of replicates whose bootstrap interval covers `target`, and median width."""
    cis = [r["bootstrap"][key] for r in rows if "bootstrap" in r]
    tg = np.broadcast_to(np.asarray(target, float), (len(cis),))
    hits = [lo <= t <= hi for (lo, hi), t in zip(cis, tg) if np.isfinite(lo)]
    widths = [hi - lo for lo, hi in cis if np.isfinite(lo)]
    return {"replicates": len(hits), "coverage": float(np.mean(hits)) if hits else float("nan"),
            "median_width": float(np.median(widths)) if widths else float("nan")}


def run_a(args, out):
    refs, n2refs = {}, {}
    for s_index, scenario in enumerate(SCEN):
        cfg = scenario_cfg(scenario, args)
        refs[scenario] = reference_summary(oracle_tables(cfg, args.n_truth, args.truth_blocks, args.root_seed, s_index))
        blocks = [naive2_reference(cfg, args.n_truth, rng_for(args.root_seed, EXPERIMENT, 910, s_index, b))
                  for b in range(args.truth_blocks)]
        n2refs[scenario] = {k: {"value": float(np.mean([b[k] for b in blocks])),
                                "mc_se": float(np.std([b[k] for b in blocks], ddof=1) / np.sqrt(len(blocks)))}
                            for k in ("X", "Ygx")}
    jobs = [(s_index, scenario, rep, args) for s_index, scenario in enumerate(SCEN) for rep in range(args.reps_a)]
    rows = pmap(a_replicate, jobs, args.workers)
    summary, cov = {}, {}
    for scenario in SCEN:
        rr = [r for r in rows if r["scenario"] == scenario]
        ref, n2ref = refs[scenario], n2refs[scenario]
        summary[scenario] = {k: summarize([r[k] for r in rr], seed=i) for i, k in enumerate(rr[0])
                             if k not in ("scenario", "rep", "bootstrap")}
        cov[scenario] = {
            "E3_rollout_X_vs_policy_contrast": coverage(rr, "E3_X", ref["pc_X"]["value"]),
            "E3_rollout_Y_vs_policy_contrast": coverage(rr, "E3_Y", ref["pc_Y"]["value"]),
            "E3_rollout_pi_vs_policy_contrast": coverage(rr, "E3_pi", ref["pc_pi"]["value"]),
            "naive2_X_vs_marginal_game": coverage(rr, "n2_X", n2ref["X"]["value"]),
            "naive2_Ygx_vs_marginal_game": coverage(rr, "n2_Ygx", n2ref["Ygx"]["value"]),
            "naive2_Ygx_vs_policy_contrast_Y": coverage(rr, "n2_Ygx", ref["pc_Y"]["value"]),
            "union3_raw_pi_vs_shapley": coverage(rr, "u_raw_pi", ref["shapley_pi"]["value"]),
            "union3_clip_pi_vs_shapley": coverage(rr, "u_clip_pi", ref["shapley_pi"]["value"]),
        }
    cfg = scenario_cfg("S3", args)
    x = rng_for(args.root_seed, EXPERIMENT, 930, 0).normal(size=(args.n_truth, cfg.d))
    out["A"] = {"operating_point": {"b0": cfg.b0, "tau_r": cfg.tau_r, "alert_rate": float(np.mean(r_hat(cfg, x) > cfg.tau_r)),
                                    "event_rate_untreated": float(np.mean(p_y1(cfg, x, np.zeros(len(x)), False)))},
                "references": refs, "naive2_references": n2refs, "summary": summary,
                "bootstrap_coverage": cov, "runs": rows}


def run_b(args, out):
    refs = {}
    for e_index, eps in enumerate(EPS_GRID):
        cfg = make_cfg(eps=eps, deploy=True)
        refs[str(eps)] = reference_summary(oracle_tables(cfg, args.n_truth, args.truth_blocks, args.root_seed, 100 + e_index))
    jobs = [(e_index, eps, rep, args) for e_index, eps in enumerate(EPS_GRID) for rep in range(args.reps_b)]
    rows = pmap(b_replicate, jobs, args.workers)
    summary = []
    for eps in EPS_GRID:
        rr = [r for r in rows if r["eps"] == eps]
        ref = refs[str(eps)]
        entry = {"eps": eps, "truth_policy_contrast_pi": ref["pc_pi"], "truth_shapley_pi": ref["shapley_pi"]}
        for k in ("union3_raw_pi", "union3_clip_pi", "ess", "E3_rollout_pi"):
            vals = np.array([r[k] for r in rr], float)
            entry[k] = summarize(vals, seed=1)
            entry[k]["finite_fraction"] = float(np.mean(np.isfinite(vals)))
        entry["coverage"] = {"E3_pi_vs_policy_contrast": coverage(rr, "E3_pi", ref["pc_pi"]["value"]),
                             "union3_raw_pi_vs_shapley": coverage(rr, "u_raw_pi", ref["shapley_pi"]["value"]),
                             "union3_clip_pi_vs_shapley": coverage(rr, "u_clip_pi", ref["shapley_pi"]["value"])}
        summary.append(entry)
    jobs = [(n_index, n, rep, args) for n_index, n in enumerate(N_SCALING) for rep in range(args.reps_n)]
    nrows = pmap(n_scaling_replicate, jobs, args.workers)
    target = refs["0.0"]["shapley_pi"]["value"]
    nsum = []
    for n in N_SCALING:
        rr = [r for r in nrows if r["n"] == n]
        est = summarize([r["union3_raw_pi"] for r in rr], target=target, seed=2)
        nsum.append({"n": n, "union3_raw_pi": est, "truth_shapley_pi": target,
                     "analyst_ci95_median": [float(np.median([r["bootstrap_ci95"][0] for r in rr])),
                                             float(np.median([r["bootstrap_ci95"][1] for r in rr]))],
                     "analyst_ci_covers_truth": float(np.mean([r["bootstrap_ci95"][0] <= target <= r["bootstrap_ci95"][1] for r in rr])),
                     "alert_only_patients_mean": float(np.mean([r["alert_only_patients"] for r in rr])),
                     "alert_only_treated_max": int(max(r["alert_only_treated"] for r in rr))})
    out["B"] = {"references": refs, "summary": summary, "runs": rows,
                "n_scaling": {"summary": nsum, "runs": nrows}}


def run_d(args, out):
    cells = [(t, p) for t in (0.5, 1.0, 1.5, 2.5) for p in (0.1, 0.3, 0.5, 0.7, 0.85, 0.9)]
    jobs = [(c_index, t, p, rep, args) for c_index, (t, p) in enumerate(cells) for rep in range(args.reps_d)]
    rows = pmap(d_replicate, jobs, args.workers)
    summary = []
    for t, p in cells:
        rr = [r for r in rows if r["theta"] == t and r["p_alert"] == p]
        entry = {"theta": t, "p_alert": p}
        for k in ("auroc_e0", "auroc_e1", "auroc_cf", "averted_observed", "averted_expected",
                  "extra_unnecessary", "nb_std", "nb_cf", "nb_standard_care"):
            entry[k] = summarize([r[k] for r in rr], seed=3)
        entry["net_averted_pp_by_harm"] = {str(h): summarize([100 * (r["averted_expected"] - h * r["extra_unnecessary"]) for r in rr], seed=4)
                                            for h in HARM_GRID}
        summary.append(entry)
    out["D"] = {"summary": summary, "runs": rows}


def run_e(args, out):
    rows = pmap(e_replicate, [(rep, args) for rep in range(args.reps_e)], args.workers)
    rd = np.array([r["rd_local"] for r in rows])
    policy = np.array([r["policy_effect_alerted"] for r in rows])
    curve = np.array([r["policy_effect_curve"] for r in rows], float)
    out["E"] = {"policy_effect_alerted": summarize(policy, seed=5),
                "treatment_effect_alerted": summarize([r["treatment_effect_alerted"] for r in rows], seed=8),
                "rd_local": summarize(rd, seed=6),
                "rd_minus_policy": summarize(rd - policy, seed=7),
                "rd_relative_to_policy": float(rd.mean() / policy.mean()),
                "policy_effect_curve": {"bin_edges": [float(v) for v in rd_grid(make_cfg(deploy=True))],
                                        "mean": [float(v) for v in np.nanmean(curve, 0)]},
                "runs": rows}


def run_f(args, out):
    rows_spec = [(setting, tg, kw) for setting, kw in (("endogenous only", dict(delta=0.0, dy=0.0)),
                                                       ("mixed", dict(delta=0.5, dy=0.6)))
                 for tg in (0.0, 0.1, 0.2, 0.3, 0.5)]
    refs = []
    for row_index, (setting, tg, kw) in enumerate(rows_spec):
        cfg = make_cfg(triage=tg, deploy=True, **kw)
        tables = oracle_tables(cfg, args.n_truth, args.truth_blocks, args.root_seed, 200 + row_index)
        ref = reference_summary(tables)
        pros = [prospective_expected(cfg, args.n_truth, rng_for(args.root_seed, EXPERIMENT, 920, row_index, b))
                for b in range(args.truth_blocks)]
        retro = [tv[frozenset(P3)] - tv[frozenset({"X", "Y"})] for tv in tables]
        refs.append({"setting": setting, "triage": tg, "references": ref,
                     "prospective": {"value": float(np.mean(pros)), "mc_se": float(np.std(pros, ddof=1) / np.sqrt(len(pros)))},
                     "retrospective": {"value": float(np.mean(retro)), "mc_se": float(np.std(retro, ddof=1) / np.sqrt(len(retro)))}})
    jobs = [(row_index, setting, tg, kw, rep, args) for row_index, (setting, tg, kw) in enumerate(rows_spec)
            for rep in range(args.reps_f)]
    rows = pmap(f_replicate, jobs, args.workers)
    summary = []
    for ref in refs:
        rr = [r for r in rows if r["setting"] == ref["setting"] and r["triage"] == ref["triage"]]
        entry = dict(ref)
        for k in ("uo_X", "uo_pi", "ua_X", "ua_pi", "e3_X", "e3_Y", "e3_pi", "n2_X", "n2_Ygx"):
            entry[k] = summarize([r[k] for r in rr], seed=8)
        entry["e3_pi_minus_prospective"] = summarize([r["e3_pi"] - ref["prospective"]["value"] for r in rr], seed=9)
        summary.append(entry)
    out["F"] = {"summary": summary, "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", default="A,B,D,E,F")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--n-truth", type=int, default=1_000_000)
    parser.add_argument("--truth-blocks", type=int, default=4)
    parser.add_argument("--reps-a", type=int, default=100)
    parser.add_argument("--boot-reps-a", type=int, default=100, help="replicates in A that get a patient bootstrap")
    parser.add_argument("--reps-b", type=int, default=30)
    parser.add_argument("--boot-reps-b", type=int, default=30)
    parser.add_argument("--reps-n", type=int, default=10)
    parser.add_argument("--reps-d", type=int, default=20)
    parser.add_argument("--n-d", type=int, default=40000)
    parser.add_argument("--reps-e", type=int, default=40)
    parser.add_argument("--n-e", type=int, default=80000)
    parser.add_argument("--reps-f", type=int, default=30)
    parser.add_argument("--boots", type=int, default=300)
    parser.add_argument("--boots-n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--root-seed", type=int, default=2026092907)
    parser.add_argument("--b0", type=float, default=None,
                        help="part A at another outcome intercept (e.g. -3.5, event rate about 7%%); use with its own --root-seed and --output")
    parser.add_argument("--alert-rate", type=float, default=0.2, help="alert rate that sets the threshold when --b0 is given")
    parser.add_argument("--output", default="merged_expE7.json")
    parser.add_argument("--merge", action="store_true",
                        help="keep the parts not rerun from the existing figures/<output>")
    args = parser.parse_args()
    if args.b0 is not None and args.parts != "A":
        parser.error("--b0 applies to part A only; pass --parts A")
    out = {"design": __doc__.split("\n\n")[1].replace("\n", " "),
           "parameters": {k: v for k, v in vars(args).items() if k not in ("output", "merge")} | {
               "experiment_block": EXPERIMENT, "harm_grid": list(HARM_GRID), "eps_grid": list(EPS_GRID),
               "n_scaling": list(N_SCALING)}}
    existing = ROOT / "figures" / args.output
    if args.merge and existing.exists():
        old = json.loads(existing.read_text())
        rerun = set(args.parts.split(","))
        out.update({k: v for k, v in old.items() if k in ("A", "B", "D", "E", "F") and k not in rerun})
        out["parameters"]["parts"] = ",".join(sorted(set(old["parameters"]["parts"].split(",")) | rerun))
    start = time.time()
    parts = {"A": run_a, "B": run_b, "D": run_d, "E": run_e, "F": run_f}
    for part in args.parts.split(","):
        parts[part](args, out)
        print(f"finished {part} ({time.time() - start:.0f}s)", flush=True)
    write_json(args.output, out)
    print(f"wrote {args.output} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
