"""E6 step 2: semi-synthetic Experiments A and C on the MIMIC-IV ward cohort.

    python3 sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz

Run locally, next to credentialed data (see sim2/e6_mimic_cohort.py). The
script writes figures/merged_expE6.json and figures/fig7_mimic.pdf (with
copies in aaai/figures/) only when the cohort comes from MIMIC-IV and has at
least --min-admissions admissions; --tag writes merged_expE6_<tag>.json and
fig7_mimic_<tag>.pdf instead, for sensitivity runs. Cohorts built from the
synthetic fixture (sim2/e6_fixture.py) or the demo are refused unless
--output and --figure point elsewhere.

What is real and what is simulated.
  Real:       covariates, the outcome under historical care (ICU admission or
              death within 12 h), and the period (anchor_year_group).
  Simulated:  (1) deployment of a logistic score fitted to part of the early
              period; (2) the alert threshold (a fixed alert rate on that
              training part); (3) clinician response: an alert is acted upon
              with probability p_act; (4) the effect of the alert-triggered
              action: it prevents an event that would otherwise have occurred
              with probability rrr and never causes one; (5) the randomised
              unalerted arm (30% of post-deployment patients); (6) the
              retrain-and-redeploy loop.
Potential outcomes follow: Y(standard care) is the MIMIC-IV outcome and
Y(alert policy) = Y x (1 - A B), with A ~ Bernoulli(p_act) if alerted and
B ~ Bernoulli(rrr). Before deployment no patient receives the alert-triggered
action, so the union-graph estimator faces the eps = 0 positivity failure.

Experiment A analogue (attribution). Pre and post samples:
  deploy_only        both drawn from the same early-period pool, post under
                     the alert policy (S3 analogue: the exogenous change is
                     exactly zero, so the policy term is the whole change);
  drift_only         pre early, post late, no deployment (real drift);
  drift_and_deploy   pre early, post late under the alert policy (S4 analogue).
Estimators follow merged.py: the monitor's two-player game with a classifier
density ratio, the proposed decomposition with the randomised arm, and (in
deploy_only, where no outcome-mechanism ratio is needed) the union-graph
estimator with oracle policy weights. Targets: the policy contrast
R_post(alert) - R_post(standard care) and the exogenous change
R_post(standard care) - R_pre(standard care), both over the finite pools and
in expectation over the simulated components. The X / outcome split of the
exogenous change has no ground truth in real data and is reported, not scored.
Replicates resample patients (with all their admissions) from the pools and
redraw the simulation.

Experiment C analogue (retraining). The late period is split by patient into
a training pool and an evaluation pool. Eight rounds; fixed threshold or
rate-held; rules keep, naive refit, refit on A = 0, and refit on A = 0 weighted
by 1 / P(A = 0 | X). Events averted are expected values on the evaluation
pool against its historical (standard-care) outcomes.

Periods. --period-by anchor_group (default) uses the patient's
anchor_year_group. --period-by admission_year places each admission by its
approximate year, anchor_year_group shifted by year(admittime) - anchor_year,
and drops admissions whose three-year window straddles the early/late
boundary.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURE_DIRS, ROOT, expected_auroc, rng_for, save_figure, summarize, write_json

EXPERIMENT = 6
EARLY = ("2008 - 2010", "2011 - 2013")
LATE = ("2014 - 2016", "2017 - 2019")
LOG_FEATURES = ("hours_since_admission", "creatinine", "urea_nitrogen", "glucose", "wbc", "platelets",
                "hours_in_icu", "hours_before_icu", "lactate")      # the last three: open stand-in cohort
MIMIC = "MIMIC-IV"
SCENARIOS = (("deploy_only", False, True), ("drift_only", True, False), ("drift_and_deploy", True, True))
ARMS_C = ("keep", "naive", "untreated", "untreated_ipw")
STATE = {}


# ----------------------------------------------------------------------------- data
def subject_split(subject_id, fraction, salt):
    """Deterministic patient-level split: True for about `fraction` of patients.

    A keyed BLAKE2 hash per patient, so splits with different salts are independent.
    """
    import hashlib
    ids = np.asarray(subject_id)
    uniq, inverse = np.unique(ids, return_inverse=True)
    u = np.array([int.from_bytes(hashlib.blake2b(f"{salt}:{s}".encode(), digest_size=8).digest(), "big") / 2 ** 64
                  for s in uniq.tolist()])
    return (u < fraction)[inverse]


def year_range(groups):
    """Years covered by contiguous anchor_year_group labels such as '2008 - 2010'."""
    spans = sorted(tuple(int(v) for v in re.findall(r"\d{4}", g)) for g in groups)
    if any(len(s) != 2 for s in spans) or any(b[0] != a[1] + 1 for a, b in zip(spans, spans[1:])):
        raise ValueError(f"period groups must be contiguous 'YYYY - YYYY' labels: {groups}")
    return spans[0][0], spans[-1][1]


def period_masks(period, years_from_anchor, early, late, mode):
    if mode == "anchor_group":
        return np.isin(period, early), np.isin(period, late)
    if years_from_anchor is None:
        raise SystemExit("--period-by admission_year needs a cohort built by the current e6_mimic_cohort.py")
    spans = np.array([[int(v) for v in re.findall(r"\d{4}", p)][:2] if len(re.findall(r"\d{4}", p)) >= 2 else [0, 0]
                      for p in period.tolist()], float)
    lo, hi = spans[:, 0] + years_from_anchor, spans[:, 1] + years_from_anchor
    (e0, e1), (l0, l1) = year_range(early), year_range(late)
    valid = spans[:, 0] > 0
    return valid & (lo >= e0) & (hi <= e1), valid & (lo >= l0) & (hi <= l1)


class Cohort:
    def __init__(self, path, early, late, train_fraction, period_by="anchor_group"):
        z = np.load(path, allow_pickle=False)
        self.synthetic = bool(z["synthetic_fixture"]) if "synthetic_fixture" in z.files else False
        names = [str(n) for n in z["feature_names"]]
        x = np.array(z["X_raw"], float)
        for name in LOG_FEATURES:
            if name in names:
                j = names.index(name)
                x[:, j] = np.log1p(np.clip(x[:, j], 0, None))
        period = z["period"].astype(str)
        years = np.array(z["years_from_anchor"], float) if "years_from_anchor" in z.files else None
        self.y = z["y"].astype(float)
        self.subject = z["subject_id"]
        self.early, self.late = period_masks(period, years, early, late, period_by)
        self.model_train = self.early & subject_split(self.subject, train_fraction, 11)
        ref = x[self.model_train]
        med = np.nanmedian(ref, axis=0)
        miss = np.isnan(x)
        keep_ind = miss[self.early].mean(0) > 0.01
        x = np.where(miss, med, x)
        mu, sd = x[self.model_train].mean(0), x[self.model_train].std(0) + 1e-9
        x = (x - mu) / sd
        self.x = np.column_stack([x, miss[:, keep_ind].astype(float)])
        self.feature_names = names + [f"missing_{n}" for n, k in zip(names, keep_ind) if k]
        self.n_periods = {p: int((period == p).sum()) for p in np.unique(period)}
        self.rate_by_period = {p: float(self.y[period == p].mean()) for p in np.unique(period)}
        self.outcome_share = ({"icu_admission": float(z["y_icu"].mean()), "death": float(z["y_death"].mean())}
                              if "y_icu" in z.files else None)
        self.extraction = json.loads(str(z["meta_json"])) if "meta_json" in z.files else None
        self.source = (self.extraction or {}).get("source", MIMIC)


class Pool:
    """Admissions of one pool grouped by patient, for patient-level resampling."""

    def __init__(self, idx, subject):
        idx = np.asarray(idx)
        self.idx = idx[np.argsort(subject[idx], kind="stable")]
        _, self.start, self.count = np.unique(subject[self.idx], return_index=True, return_counts=True)

    def __len__(self):
        return len(self.idx)

    def draw(self, rng):
        """All admissions of len(patients) patients drawn with replacement."""
        k = rng.integers(0, len(self.start), len(self.start))
        start, count = self.start[k], self.count[k]
        offset = np.repeat(start - (np.cumsum(count) - count), count) + np.arange(count.sum())
        return self.idx[offset]


def fit_logistic(x, y, weight=None, c=1.0):
    model = LogisticRegression(C=c, max_iter=5000).fit(x, y, sample_weight=weight)
    return lambda xx: model.predict_proba(xx)[:, 1]


# ----------------------------------------------------------------------------- A analogue
def ratio_classifier(x0, x1):
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack([x0, x1]), np.r_[np.zeros(len(x0)), np.ones(len(x1))])

    def ratio(x):
        p = np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)
        return p / (1 - p) * len(x0) / len(x1)
    return ratio


def wmean(v, w):
    return float(np.sum(v * w) / np.sum(w))


def monitor(x0, l0, x1, l1, rx=None):
    """Two players {P(X), P(Y|X)}, classifier ratio, post outcomes used directly (merged.naive2_est)."""
    r0, r1 = l0.mean(), l1.mean()
    rx = rx or ratio_classifier(x0, x1)
    vx = wmean(l0, rx(x0)) - r0
    vy = wmean(l1, 1 / rx(x1)) - r0
    va = r1 - r0
    return {"X": 0.5 * (vx + va - vy), "Ygx": 0.5 * (vy + va - vx), "dR": va}


def proposed(x0, l0, x1, l1, arm, rx=None):
    """Exogenous Shapley with the policy at standard care, plus the randomised-arm policy term (merged.E3)."""
    m = arm == 1
    c = arm == 0 if np.any(arm == 0) else np.ones(len(arm), bool)
    r0, r1 = l0.mean(), l1[m].mean()
    rx = rx or ratio_classifier(x0, x1[m])
    wx0, wx1 = rx(x0), 1 / rx(x1)
    r_e1_pi0, r_mixed = l1[c].mean(), wmean(l1[c], wx1[c])
    vx, vy, vexo = wmean(l0, wx0) - r0, r_mixed - r0, r_e1_pi0 - r0
    return {"X": 0.5 * (vx + vexo - vy), "Y": 0.5 * (vy + vexo - vx), "pi": r1 - r_e1_pi0, "dR": r1 - r0}


def union_policy_term(l0, alert0, p_act):
    """Union-graph IS policy term with oracle weights when no pre-deployment patient got the action.

    All pre-deployment patients have A = 0, so the policy weight is
    pi1(A=0|x) / pi0(A=0|x) = 1 - p_act * alert(x); treated alert-only patients
    have no pre-deployment support and are silently omitted.
    """
    w = 1 - p_act * alert0
    return wmean(l0, w) - float(l0.mean())


def a_truths(f_pre, y_pre, f_post, y_post, alert_post, deploy, p_prevent):
    l_pre = (f_pre - y_pre) ** 2
    l_std = (f_post - y_post) ** 2
    if deploy:
        prevented = alert_post * (y_post == 1)
        l_alert = np.where(prevented, p_prevent * f_post ** 2 + (1 - p_prevent) * (1 - f_post) ** 2, l_std)
    else:
        l_alert = l_std
    return {"pi": float(l_alert.mean() - l_std.mean()), "exogenous": float(l_std.mean() - l_pre.mean()),
            "alert_rate_post": float(alert_post.mean()) if deploy else 0.0}


def a_pools(c):
    """Pre and post pools per scenario. Pre and post are resampled independently in each replicate."""
    early_pool = np.flatnonzero(c.early & ~c.model_train)
    late = np.flatnonzero(c.late)
    return {"deploy_only": (early_pool, early_pool), "drift_only": (early_pool, late),
            "drift_and_deploy": (early_pool, late)}


def a_replicate(job):
    s_index, rep = job
    c, f0, tau, args = STATE["cohort"], STATE["f0"], STATE["tau"], STATE["args"]
    name, _drift, deploy = SCENARIOS[s_index]
    pre, post = STATE["a_pools"][name]
    rng = rng_for(args.root_seed, EXPERIMENT, 1, s_index, rep)
    i0, i1 = pre.draw(rng), post.draw(rng)
    x0, y0, x1, y1 = c.x[i0], c.y[i0], c.x[i1], c.y[i1]
    s0, s1 = f0(x0), f0(x1)
    alert1 = s1 > tau
    arm = (rng.random(len(i1)) >= args.control_frac).astype(int) if deploy else np.ones(len(i1), int)
    acted = (arm == 1) & alert1 & (rng.random(len(i1)) < args.p_act) if deploy else np.zeros(len(i1), bool)
    prevented = acted & (rng.random(len(i1)) < args.rrr)
    y_obs = np.where(prevented, 0.0, y1)
    l0, l1 = (s0 - y0) ** 2, (s1 - y_obs) ** 2
    d = arm == 1
    rx = ratio_classifier(x0, x1[d])            # monitor and proposed fit the same ratio on the same data
    mon = monitor(x0, l0, x1[d], l1[d], rx)
    pro = proposed(x0, l0, x1, l1, arm, rx)
    row = {"scenario": name, "rep": rep, "monitor_X": mon["X"], "monitor_Ygx": mon["Ygx"],
           "proposed_X": pro["X"], "proposed_Y": pro["Y"], "proposed_pi": pro["pi"],
           "proposed_exogenous": pro["X"] + pro["Y"], "dR_observed": pro["dR"],
           "auroc_change_observed": float(roc_auc_score(y_obs[d], s1[d]) - roc_auc_score(y0, s0)),
           "n_pre": int(len(i0)), "n_post": int(len(i1)), "alert_rate_post": float(alert1.mean())}
    if name == "deploy_only":
        row["union_pi"] = union_policy_term(l0, (s0 > tau).astype(float), args.p_act)
    return row


def run_a(c, f0, tau, args):
    truths = {}
    p_prevent = args.p_act * args.rrr
    for name, _drift, deploy in SCENARIOS:
        pre, post = STATE["a_pools"][name]
        f_pre, f_post = f0(c.x[pre.idx]), f0(c.x[post.idx])
        truths[name] = a_truths(f_pre, c.y[pre.idx], f_post, c.y[post.idx], (f_post > tau).astype(float), deploy,
                                p_prevent)
        truths[name].update(n_pre=int(len(pre)), n_post=int(len(post)),
                            patients_pre=int(len(pre.start)), patients_post=int(len(post.start)))
    jobs = [(s_index, rep) for s_index in range(len(SCENARIOS)) for rep in range(args.reps_a)]
    rows = pmap(a_replicate, jobs, args.workers)
    print("  A done", flush=True)
    summary = {}
    for name, *_ in SCENARIOS:
        rr = [r for r in rows if r["scenario"] == name]
        t = truths[name]
        summary[name] = {k: summarize([r[k] for r in rr],
                                      target=(t["pi"] if k in ("proposed_pi", "union_pi") else
                                              t["exogenous"] if k == "proposed_exogenous" else None), seed=i)
                         for i, k in enumerate(rr[0]) if k not in ("scenario", "rep")}
    return {"truth": truths, "summary": summary, "runs": rows}


# ----------------------------------------------------------------------------- C analogue
def c_replicate(job):
    t_index, rep = job
    c, f0, tau, args = STATE["cohort"], STATE["f0"], STATE["tau"], STATE["args"]
    rule = ("fixed", "rate")[t_index]
    train_pool, eval_pool, held_rate = STATE["c_train"], STATE["c_eval"], STATE["held_rate"]
    n_train = args.n_train if args.n_train > 0 else len(train_pool)
    x_eval, y_eval = c.x[eval_pool], c.y[eval_pool]
    rows = []
    for arm in ARMS_C:
        score = f0
        for rnd in range(1, args.rounds + 1):
            thr = tau if rule == "fixed" else float(np.quantile(score(c.x[train_pool]), 1 - held_rate))
            s = score(x_eval)
            alert = (s > thr).astype(float)
            q_obs = y_eval * (1 - args.p_act * args.rrr * alert)
            rows.append({"threshold_rule": rule, "rep": rep, "arm": arm, "round": rnd,
                         "events_averted_pp": float(100 * (y_eval.mean() - q_obs.mean())),
                         "alert_rate": float(alert.mean()),
                         "observed_auroc": expected_auroc(s, q_obs),
                         "calibration_in_large": float(s.mean() - y_eval.mean())})
            if arm == "keep" or rnd == args.rounds:
                continue
            # common random numbers across arms: the same patients and uniforms in each round
            rng = rng_for(args.root_seed, EXPERIMENT, 2, t_index, rep, rnd)
            idx = rng.choice(train_pool, size=n_train, replace=True)
            u_act, u_prev = rng.random(n_train), rng.random(n_train)
            xb, yb = c.x[idx], c.y[idx]
            alert_b = score(xb) > thr
            p_a = np.where(alert_b, args.p_act, 0.0)
            acted = u_act < p_a
            y_obs = np.where(acted & (u_prev < args.rrr), 0.0, yb)
            if arm == "naive":
                score = fit_logistic(xb, y_obs)
            elif arm == "untreated":
                score = fit_logistic(xb[~acted], y_obs[~acted])
            else:
                score = fit_logistic(xb[~acted], y_obs[~acted], weight=1 / (1 - p_a[~acted]))
    return rows


def run_c(c, f0, tau, args):
    jobs = [(t_index, rep) for t_index in range(2) for rep in range(args.reps_c)]
    rows = [r for chunk in pmap(c_replicate, jobs, args.workers) for r in chunk]
    print("  C done", flush=True)
    summary = []
    for rule in ("fixed", "rate"):
        for arm in ARMS_C:
            for rnd in range(1, args.rounds + 1):
                cell = [r for r in rows if r["threshold_rule"] == rule and r["arm"] == arm and r["round"] == rnd]
                summary.append({"threshold_rule": rule, "arm": arm, "round": rnd,
                                **{k: summarize([r[k] for r in cell], seed=rnd)
                                   for k in ("events_averted_pp", "alert_rate", "observed_auroc", "calibration_in_large")}})
    return {"held_alert_rate": STATE["held_rate"], "n_train_pool": int(len(STATE["c_train"])),
            "n_train_per_round": int(args.n_train if args.n_train > 0 else len(STATE["c_train"])),
            "n_eval_pool": int(len(STATE["c_eval"])), "event_rate_eval_pool": float(c.y[STATE["c_eval"]].mean()),
            "summary": summary, "runs": rows}


# ----------------------------------------------------------------------------- driver
def init_worker(state):
    STATE.update(state)
    model = STATE.pop("f0_model")
    STATE["f0"] = lambda xx: model.predict_proba(xx)[:, 1]


def pmap(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(STATE["shared"],)) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (8 * workers))))


# ----------------------------------------------------------------------------- figure
def figure(result, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    A = result["A"]
    items = (("monitor_Ygx", "monitor: $P(Y\\mid X)$", "#EE6677", "o"),
             ("proposed_pi", "proposed: policy term", "#228833", "s"),
             ("proposed_exogenous", "proposed: exogenous terms", "#4477AA", "^"),
             ("union_pi", "union graph, oracle weights", "#999999", "D"))
    for s_i, name in enumerate(("deploy_only", "drift_and_deploy")):
        summ, truth = A["summary"][name], A["truth"][name]
        for k_i, (key, label, color, marker) in enumerate(items):
            if key not in summ:
                continue
            v = summ[key]
            y = s_i * 5 + k_i
            ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt=marker,
                        color=color, ms=4, capsize=1.5, label=label if s_i == 0 or key != "union_pi" else None)
        ax.plot([truth["pi"]] * 2, [s_i * 5 - 0.5, s_i * 5 + 3.5], color="#228833", ls="--", lw=0.8)
        ax.plot([truth["exogenous"]] * 2, [s_i * 5 - 0.5, s_i * 5 + 3.5], color="#4477AA", ls=":", lw=0.8)
    ax.axvline(0, color="black", lw=0.5)
    labels = result.get("scenario_labels") or {}
    ax.set_yticks([1.5, 6.5])
    ax.set_yticklabels([labels.get("deploy_only", "deployment only\n(early period)"),
                        labels.get("drift_and_deploy", "real drift and\ndeployment")])
    ax.set_xlabel("attributed change in Brier score")
    ax.set_title("A: attribution (dashed/dotted: targets)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:4], fontsize=5.6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax = axes[1]
    styles = (("keep", "keep", "#228833", "-"), ("naive", "naive refit", "#EE6677", "-"),
              ("untreated", "refit on $A=0$", "#66CCEE", "--"), ("untreated_ipw", "refit on $A=0$, weighted", "#AA3377", ":"))
    for arm, label, color, ls in styles:
        cells = sorted((e for e in result["C"]["summary"] if e["threshold_rule"] == "fixed" and e["arm"] == arm),
                       key=lambda e: e["round"])
        r = np.array([e["round"] for e in cells])
        m = np.array([e["events_averted_pp"]["mean"] for e in cells])
        lo = np.array([e["events_averted_pp"]["ci95"][0] for e in cells])
        hi = np.array([e["events_averted_pp"]["ci95"][1] for e in cells])
        ax.plot(r, m, color=color, ls=ls, marker="o", ms=2, lw=1.1, label=label)
        ax.fill_between(r, lo, hi, color=color, alpha=0.18, lw=0)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("deployment round")
    ax.set_ylabel("events averted vs historical care (pp)")
    ax.set_title("C: retrain and redeploy, fixed threshold")
    ax.legend(fontsize=5.8, frameon=False)
    fig.tight_layout()
    save_figure(fig, Path(paths[0]).name, dirs=tuple(Path(p).parent for p in paths))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--early", nargs="+", default=list(EARLY))
    parser.add_argument("--late", nargs="+", default=list(LATE))
    parser.add_argument("--period-by", choices=("anchor_group", "admission_year"), default="anchor_group")
    parser.add_argument("--train-fraction", type=float, default=0.4, help="share of early-period patients used to fit the score")
    parser.add_argument("--alert-rate", type=float, default=0.10)
    parser.add_argument("--p-act", type=float, default=0.85)
    parser.add_argument("--rrr", type=float, default=0.5, help="probability an acted-upon alert prevents the event")
    parser.add_argument("--control-frac", type=float, default=0.3)
    parser.add_argument("--reps-a", type=int, default=200)
    parser.add_argument("--reps-c", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--n-train", type=int, default=8000, help="training sample per refit; 0 = the whole training pool")
    parser.add_argument("--root-seed", type=int, default=2026092906)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", default=None, help="sensitivity run: write merged_expE6_<tag>.json and fig7_mimic_<tag>.pdf")
    parser.add_argument("--output", default=None, help="JSON path outside figures/ (fixture and demo runs)")
    parser.add_argument("--figure", default=None, help="figure path outside figures/ (fixture and demo runs)")
    parser.add_argument("--min-admissions", type=int, default=5000,
                        help="smallest cohort allowed to write into figures/ (the open demo has about 100 patients)")
    args = parser.parse_args()
    if (args.output is None) != (args.figure is None):
        parser.error("pass both --output and --figure, or neither")
    if args.tag is not None and args.output is not None:
        parser.error("--tag writes into figures/; it cannot be combined with --output")
    if args.tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        parser.error("--tag may contain letters, digits, '.', '_' and '-' only")
    start = time.time()
    c = Cohort(args.cohort, args.early, args.late, args.train_fraction, args.period_by)
    to_figures = args.output is None
    if c.synthetic and to_figures:
        parser.error("this cohort was built from the synthetic fixture; pass --output and --figure outside figures/")
    if c.source != MIMIC and to_figures:
        parser.error(f"this cohort comes from {c.source}, not MIMIC-IV; pass --output and --figure outside the E6 slot")
    if len(c.y) < args.min_admissions and to_figures:
        parser.error(f"the cohort has {len(c.y)} admissions (fewer than --min-admissions={args.min_admissions}), "
                     "as for the MIMIC-IV demo or a test extract; pass --output and --figure outside figures/")
    if min(c.early.sum(), c.late.sum(), c.model_train.sum()) < 50:
        parser.error("too few admissions in the early or late period for this run (check --early, --late, --period-by)")
    model = LogisticRegression(C=1.0, max_iter=5000).fit(c.x[c.model_train], c.y[c.model_train])
    f0 = lambda xx: model.predict_proba(xx)[:, 1]
    tau = float(np.quantile(f0(c.x[c.model_train]), 1 - args.alert_rate))
    late = np.flatnonzero(c.late)
    train_mask = subject_split(c.subject[late], 0.6, 37)
    c_train, c_eval = late[train_mask], late[~train_mask]
    held_rate = float(np.mean(f0(c.x[c_train]) > tau))
    pools = {k: (Pool(a, c.subject), Pool(b, c.subject)) for k, (a, b) in a_pools(c).items()}
    shared = {"cohort": c, "f0_model": model, "tau": tau, "args": args, "a_pools": pools,
              "c_train": c_train, "c_eval": c_eval, "held_rate": held_rate}
    init_worker(dict(shared))
    STATE["shared"] = shared
    held_out = c.early & ~c.model_train
    result = {
        "design": __doc__.split("\n\n")[2].replace("\n", " "),
        "semi_synthetic": True, "synthetic_fixture": c.synthetic,
        "simulated_components": ["deployment of a logistic score fitted to part of the early period",
                                 f"alert threshold at a {args.alert_rate:.0%} alert rate on that part",
                                 f"clinician response to alerts (acted upon with probability {args.p_act})",
                                 f"effect of the alert-triggered action (prevents an event with probability {args.rrr}, never causes one)",
                                 f"randomised unalerted arm ({args.control_frac:.0%} of post-deployment patients)",
                                 "retrain-and-redeploy loop"],
        "data_source": c.source,
        "real_components": [s.format(early="/".join(args.early), late="/".join(args.late)) for s in
                            (c.extraction or {}).get("real_components",
                                                     ["covariates", "outcome under historical care (ICU admission or death within 12 h)",
                                                      "period (anchor_year_group)"])],
        "scenario_labels": {k: v.format(early="/".join(args.early), late="/".join(args.late))
                            for k, v in (c.extraction or {}).get("scenario_labels", {}).items()},
        "parameters": {k: v for k, v in vars(args).items() if k not in ("cohort", "output", "figure", "workers")}
                      | {"experiment_block": EXPERIMENT},
        "cohort": {"admissions": int(len(c.y)), "patients": int(len(np.unique(c.subject))),
                   "periods": c.n_periods, "event_rate_by_period": c.rate_by_period,
                   "outcome_share": c.outcome_share, "features": c.feature_names,
                   "admissions_early": int(c.early.sum()), "admissions_late": int(c.late.sum()),
                   "admissions_model_train": int(c.model_train.sum()),
                   "event_rate_early": float(c.y[c.early].mean()), "event_rate_late": float(c.y[c.late].mean()),
                   "score_auroc_early_heldout": float(roc_auc_score(c.y[held_out], f0(c.x[held_out]))),
                   "score_auroc_late": float(roc_auc_score(c.y[c.late], f0(c.x[c.late]))),
                   "alert_rate_early_heldout": float(np.mean(f0(c.x[held_out]) > tau)),
                   "alert_rate_late": float(np.mean(f0(c.x[c.late]) > tau)),
                   "threshold": tau, "extraction": c.extraction},
    }
    result["A"] = run_a(c, f0, tau, args)
    result["C"] = run_c(c, f0, tau, args)
    if to_figures:
        suffix = f"_{args.tag}" if args.tag else ""
        write_json(f"merged_expE6{suffix}.json", result)
        fig_paths = [Path(d) / f"fig7_mimic{suffix}.pdf" for d in FIGURE_DIRS]
    else:
        write_json(Path(args.output).name, result, dirs=(Path(args.output).parent,))
        fig_paths = [Path(args.figure)]
    figure(result, fig_paths)
    print(f"wrote {', '.join(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in fig_paths)} "
          f"in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
