"""
Experiment F -- the triage axis: deployment also shifts P(X).

Violates the "(independent) shifts" clause of Zhang et al. Assumption 3.2:
one event (turning the alert on) moves BOTH pi and P(X), so the two players
share a common ancestor and are not separately manipulable.

In dgp.mu_of the covariate mean is  delta * 1{X in S} + triage * 1{pi in S},
i.e. the DGP splits the covariate shift into an exogenous part carried by the
X player and an endogenous (deployment-caused) part carried by the pi player.
That split is the ground truth. An analyst cannot see it: they observe only
the total shift delta + triage.

Estimators compared, at the same operating point and seeds as exp_A:
  union3_oracle    IS estimator given the oracle split (dgp.is_values)
  union3_analyst   IS estimator that assigns the WHOLE observed covariate
                   shift to the X player -- what a real monitor computes
  naive2           two players, no action log
  E3_rollout       exogenous Shapley + policy contrast from the randomised arm

Two policy-contrast estimands, which coincide only when triage = 0:
  retrospective  v(all) - v({X,Y})        "what did deployment contribute"
                 (swaps the policy out AND undoes its triage effect)
  prospective    R_e1(pi1) - R_e1(pi0)    "what if we switch the alert off now"
                 (keeps the current patient mix; this is what the randomised
                  arm identifies and what the keep-or-retrain decision needs)
"""
import itertools, json, sys, numpy as np
sys.path.insert(0, ".")
import dgp
from dgp import p_treat, p_y1, mu_of, logpdf_x, brier, shapley, truth_values, is_values, naive2_values
from merged import make_cfg, sample_po, E3, truth_E3_decomp

P3 = ["X", "pi", "Y"]


def prospective_contrast(cfg, n=400_000, seed=0):
    """R_e1(pi1) - R_e1(pi0): both arms drawn from e1's covariate mix."""
    rng = np.random.default_rng(seed)
    mu = mu_of(cfg, None, {"X", "pi"})                 # e1 covariate mean
    x = rng.normal(size=(n, 2)) + mu
    out = {}
    for lab, deployed in [("pi1", True), ("pi0", False)]:
        a = (rng.random(n) < p_treat(cfg, x, deployed=deployed and cfg.deploy)).astype(float)
        y = (rng.random(n) < p_y1(cfg, x, a, shifted=(cfg.dy != 0))).astype(float)
        out[lab] = brier(cfg, x, y)
    return out["pi1"] - out["pi0"]


def _w_analyst(cfg, x, a, y, S):
    """Analyst's importance weights: the entire observed covariate shift is
    booked to the X player; the policy is assumed not to move P(X)."""
    w = np.ones(len(x))
    if "X" in S:
        mu_tot = mu_of(cfg, None, {"X", "pi"})         # total observed shift
        w = w * np.exp(logpdf_x(cfg, x, mu_tot) - logpdf_x(cfg, x, np.zeros(2)))
    if "pi" in S:
        p1 = p_treat(cfg, x, deployed=cfg.deploy); p0 = p_treat(cfg, x, deployed=False)
        w = w * (np.where(a == 1, p1, 1 - p1) / np.where(a == 1, p0, 1 - p0))
    if "Y" in S:
        q1 = p_y1(cfg, x, a, shifted=True); q0 = p_y1(cfg, x, a, shifted=False)
        w = w * (np.where(y == 1, q1, 1 - q1) / np.where(y == 1, q0, 1 - q0))
    return w


def union3_analyst(cfg, n=20_000, seed=1, clip=None):
    rng = np.random.default_rng(seed)
    x, a, y = dgp.sample_env(cfg, set(), n, rng)
    vals = {}
    for k in range(4):
        for S in itertools.combinations(P3, k):
            w = _w_analyst(cfg, x, a, y, set(S))
            if clip is not None:
                w = np.minimum(w, clip)
            vals[frozenset(S)] = brier(cfg, x, y, w=w)
    return shapley(vals, P3)


def run(triages=(0.0, 0.1, 0.2, 0.3, 0.5), seeds=20, n=20_000):
    out = []
    for setting, kw in [("endogenous only", dict(delta=0.0, dy=0.0)),
                        ("mixed", dict(delta=0.5, dy=0.6))]:
        for tg in triages:
            cfg = make_cfg(triage=tg, deploy=True, **kw)
            tv = truth_values(cfg, P3, n=400_000, metric="brier", seed=0)
            t3 = shapley(tv, P3); tE = truth_E3_decomp(tv)
            retro = tv[frozenset(P3)] - tv[frozenset({"X", "Y"})]
            prosp = prospective_contrast(cfg)
            acc = {k: [] for k in ["uo_X", "uo_pi", "ua_X", "ua_pi", "ua_Y",
                                   "n2_X", "n2_Y", "e3_X", "e3_Y", "e3_pi"]}
            for sd in range(seeds):
                uo = shapley(is_values(cfg, P3, n=n, metric="brier", clip=None, seed=sd)[0], P3)
                ua = union3_analyst(cfg, n=n, seed=sd)
                n2 = shapley(naive2_values(cfg, n=n, metric="brier", seed=sd), ["X", "Ygx"])
                rng = np.random.default_rng(100 + sd)
                s0 = sample_po(cfg, set(), n, rng)
                s1 = sample_po(cfg, {"X", "pi", "Y"}, n, rng, control_frac=0.3)
                e3 = E3(cfg, s0, s1, "rollout")
                for k, v in [("uo_X", uo["X"]), ("uo_pi", uo["pi"]), ("ua_X", ua["X"]),
                             ("ua_pi", ua["pi"]), ("ua_Y", ua["Y"]), ("n2_X", n2["X"]),
                             ("n2_Y", n2["Ygx"]), ("e3_X", e3["X"]), ("e3_Y", e3["Y"]),
                             ("e3_pi", e3["pi"])]:
                    acc[k].append(v)
            row = dict(setting=setting, triage=tg, dR=tv[frozenset(P3)] - tv[frozenset()],
                       truth_X=t3["X"], truth_pi=t3["pi"], truth_Y=t3["Y"],
                       truthE3_X=tE["X"], truthE3_Y=tE["Y"],
                       retrospective=retro, prospective=prosp)
            for k, v in acc.items():
                row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v))
            out.append(row)
            print(f"{setting:16s} triage={tg:.1f} | dR {row['dR']:+.4f} | truth X {t3['X']:+.4f} pi {t3['pi']:+.4f}"
                  f" | retro {retro:+.4f} prosp {prosp:+.4f}"
                  f" | analyst X {row['ua_X']:+.4f} pi {row['ua_pi']:+.4f}"
                  f" | E3 pi {row['e3_pi']:+.4f}±{row['e3_pi_sd']:.4f}")
    return out


if __name__ == "__main__":
    res = run()
    json.dump(res, open("../figures/merged_expF.json", "w"), indent=1, default=float)
