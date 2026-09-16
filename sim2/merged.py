"""
Merged experiment. DGP from dgp.py (union graph X -> r_hat -> A -> Y; alert-only region where the
model flags what the clinician misses; pre-deployment treatment probability eps there).
Added here:
  * potential-outcome sampler with a randomised unalerted arm inside e1
  * E3: exogenous Shapley over {P(X), P(Y|X,A)} with the policy held at pi0 + policy term inside e1
  * naive2_est: two-player game with density ratios ESTIMATED by a classifier (no oracle ratios)
  * ESS diagnostic for the union-graph importance-sampling estimator, over an eps continuum incl. eps = 0
  * retrain-and-redeploy, counterfactual net benefit, threshold discontinuity
"""
import numpy as np, itertools, json, sys
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "."); import dgp
from dgp import sigmoid, r_hat, suspicion, p_treat, p_y1, mu_of, logpdf_x, brier, auroc, shapley, truth_values, is_values, naive2_values


def make_cfg(**kw):
    base = dict(tau_r=0.45, tau_s=1.2, p_alert=0.85, b0=-0.6, eps=1e-3, theta=2.5)
    base.update(kw); cfg = dgp.Config(**base)
    cfg.w_r = cfg.w_r * 1.5; cfg.w_y = cfg.w_r.copy()
    return cfg


# ------------------------------------------------------------------ sampler with potential outcomes
def sample_po(cfg, mechs, n, rng, control_frac=0.0):
    mu = mu_of(cfg, None, mechs); x = rng.normal(size=(n, 2)) + mu
    shifted = "Y" in mechs; deployed = ("pi" in mechs) and cfg.deploy
    q0 = p_y1(cfg, x, np.zeros(n), shifted); q1 = p_y1(cfg, x, np.ones(n), shifted)
    u = rng.random(n); y0 = (u < q0).astype(float); y1 = (u < q1).astype(float)      # monotone coupling
    arm = np.ones(n, int)
    if deployed and control_frac > 0: arm = (rng.random(n) >= control_frac).astype(int)
    p0 = p_treat(cfg, x, False); p1 = p_treat(cfg, x, True) if deployed else p0.copy()   # policy actually in force in e1
    p_apply = np.where(arm == 1, p1, p0)
    a = (rng.random(n) < p_apply).astype(float); y = np.where(a == 1, y1, y0)
    return dict(x=x, a=a, y=y, y0=y0, y1=y1, arm=arm, r=r_hat(cfg, x), p0=p0, p1=p1)


def wmean(v, w): return float(np.sum(v * w) / np.sum(w))
def brier_vec(cfg, x, y): return (r_hat(cfg, x) - y) ** 2


def ratio_X_est(x0, x1):
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack([x0, x1]), np.r_[np.zeros(len(x0)), np.ones(len(x1))])
    def f(x):
        p = np.clip(clf.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4); return p / (1 - p) * len(x0) / len(x1)
    return f


# ------------------------------------------------------------------ estimators on (e0 sample, e1 sample)
def naive2_est(cfg, s0, s1):
    """Two players {P(X), P(Y|X)} with the X density ratio estimated by a classifier and e1 outcomes used directly."""
    m = s1["arm"] == 1
    L0 = brier_vec(cfg, s0["x"], s0["y"]); L1 = brier_vec(cfg, s1["x"][m], s1["y"][m]); R0 = L0.mean(); R1 = L1.mean()
    rx = ratio_X_est(s0["x"], s1["x"][m])
    vX = wmean(L0, rx(s0["x"])) - R0; vY = wmean(L1, 1 / rx(s1["x"][m])) - R0; vAll = R1 - R0
    return {"X": 0.5 * (vX + vAll - vY), "Ygx": 0.5 * (vY + vAll - vX), "dR": vAll}


def E3(cfg, s0, s1, mode="rollout"):
    """Exogenous Shapley over {P(X), P(Y|X,A)} with policy at pi0, plus Delta_pi = R_e1(pi1) - R_e1(pi0)."""
    m = s1["arm"] == 1
    L0 = brier_vec(cfg, s0["x"], s0["y"]); L1 = brier_vec(cfg, s1["x"], s1["y"]); R0 = L0.mean(); R1 = L1[m].mean()
    rx = ratio_X_est(s0["x"], s1["x"][m]); wX0 = rx(s0["x"]); wX1 = 1 / rx(s1["x"])
    if mode == "rollout":
        c = s1["arm"] == 0
        if c.sum() == 0: c = np.ones(len(L1), bool)          # e1 still under pi0: the whole sample is the pi0 arm
        R_e1_pi0 = L1[c].mean(); R_mixed = wmean(L1[c], wX1[c])
    else:  # oracle policy weights pi0/pi1 on the deployed arm
        w = np.where(s1["a"] == 1, s1["p0"] / s1["p1"], (1 - s1["p0"]) / (1 - s1["p1"]))
        R_e1_pi0 = wmean(L1[m], w[m]); R_mixed = wmean(L1[m], (w * wX1)[m])
    vX = wmean(L0, wX0) - R0; vY = R_mixed - R0; vExo = R_e1_pi0 - R0
    return {"X": 0.5 * (vX + vExo - vY), "Y": 0.5 * (vY + vExo - vX), "pi": R1 - R_e1_pi0, "dR": R1 - R0}


def truth_E3_decomp(tv):
    """Ground-truth version of the proposed decomposition from the coalition table (players X, pi, Y)."""
    e, X, Y, XY, all_ = tv[frozenset()], tv[frozenset({"X"})], tv[frozenset({"Y"})], tv[frozenset({"X", "Y"})], tv[frozenset({"X", "pi", "Y"})]
    vX, vY, vExo = X - e, Y - e, XY - e
    return {"X": 0.5 * (vX + vExo - vY), "Y": 0.5 * (vY + vExo - vX), "pi": all_ - XY, "dR": all_ - e}


def union3_with_ess(cfg, n, clip, seed):
    """Their union-graph IS estimator plus the ESS of the policy-switch weights."""
    vals, diags = is_values(cfg, ["X", "pi", "Y"], n=n, metric="brier", clip=clip, seed=seed)
    # ESS of the policy-switch weights, computed here because dgp.py records it before multiplying by w_pi
    rng = np.random.default_rng(seed); x, a, y = dgp.sample_env(cfg, set(), n, rng)
    p1 = p_treat(cfg, x, cfg.deploy); p0 = p_treat(cfg, x, False)
    w = np.where(a == 1, p1, 1 - p1) / np.where(a == 1, p0, 1 - p0)
    if clip is not None: w = np.minimum(w, clip)
    ess = float(w.sum() ** 2 / (w ** 2).sum()) if np.isfinite(w).all() else np.nan
    return shapley(vals, ["X", "pi", "Y"]), ess, diags[frozenset({"pi"})]


# ------------------------------------------------------------------ scenarios
SCEN = {"S1": dict(delta=0.5, dy=0.0, deploy=False), "S2": dict(delta=0.0, dy=0.6, deploy=False),
        "S3": dict(delta=0.0, dy=0.0, deploy=True), "S4": dict(delta=0.5, dy=0.6, deploy=True)}


def exp_A(seeds=30, n=20000):
    out = {}
    for sc, kw in SCEN.items():
        cfg = make_cfg(**kw); tv = truth_values(cfg, ["X", "pi", "Y"], n=400_000, metric="brier", seed=0)
        truth3 = shapley(tv, ["X", "pi", "Y"]); truthE3 = truth_E3_decomp(tv)
        rows = {"truth3": truth3, "truthE3": truthE3, "runs": []}
        for sd in range(seeds):
            rng = np.random.default_rng(100 + sd)
            s0 = sample_po(cfg, set(), n, rng); s1 = sample_po(cfg, {"X", "pi", "Y"}, n, rng, control_frac=0.3)
            u_raw, ess, _ = union3_with_ess(cfg, n, None, sd); u_clip, _, _ = union3_with_ess(cfg, n, 100.0, sd)
            nv = shapley(naive2_values(cfg, n=n, metric="brier", seed=sd), ["X", "Ygx"])
            rows["runs"].append({"union3_raw": u_raw, "union3_clip": u_clip, "ess": ess, "naive2_oracle": nv,
                                 "naive2_est": naive2_est(cfg, s0, s1), "E3_rollout": E3(cfg, s0, s1, "rollout"), "E3_oracle": E3(cfg, s0, s1, "oracle")})
        out[sc] = rows
        def ms(est, key):
            v = np.array([r[est][key] for r in rows["runs"]]); return f"{v.mean():+.4f}±{v.std():.4f}"
        print(f"\n{sc}: truth  X {truth3['X']:+.4f}  pi {truth3['pi']:+.4f}  Y {truth3['Y']:+.4f} | truthE3  X {truthE3['X']:+.4f}  Y {truthE3['Y']:+.4f}  pi {truthE3['pi']:+.4f}  dR {truthE3['dR']:+.4f}")
        print(f"  union3 raw   X {ms('union3_raw','X')}  pi {ms('union3_raw','pi')}  Y {ms('union3_raw','Y')}   ESS {np.mean([r['ess'] for r in rows['runs']]):.1f}")
        print(f"  union3 clip  X {ms('union3_clip','X')}  pi {ms('union3_clip','pi')}  Y {ms('union3_clip','Y')}")
        print(f"  naive2 orac  X {ms('naive2_oracle','X')}  Y|X {ms('naive2_oracle','Ygx')}")
        print(f"  naive2 est   X {ms('naive2_est','X')}  Y|X {ms('naive2_est','Ygx')}")
        print(f"  E3 rollout   X {ms('E3_rollout','X')}  Y {ms('E3_rollout','Y')}  pi {ms('E3_rollout','pi')}")
        print(f"  E3 oracle    X {ms('E3_oracle','X')}  Y {ms('E3_oracle','Y')}  pi {ms('E3_oracle','pi')}")
    return out


def exp_B(seeds=20, n=20000):
    """eps continuum: ESS and estimator behaviour, S3."""
    out = []
    for eps in [0.0, 1e-4, 1e-3, 1e-2, 5e-2]:
        cfg = make_cfg(eps=eps, deploy=True); tv = truth_values(cfg, ["X", "pi", "Y"], n=400_000, metric="brier", seed=0)
        truthE3 = truth_E3_decomp(tv); acc = []
        for sd in range(seeds):
            rng = np.random.default_rng(300 + sd)
            s0 = sample_po(cfg, set(), n, rng); s1 = sample_po(cfg, {"X", "pi", "Y"}, n, rng, control_frac=0.3)
            with np.errstate(divide="ignore", invalid="ignore"):
                u_raw, ess, d = union3_with_ess(cfg, n, None, sd); u_clip, _, _ = union3_with_ess(cfg, n, 100.0, sd)
            e3 = E3(cfg, s0, s1, "rollout")
            acc.append([u_raw["pi"], u_clip["pi"], ess, e3["pi"], d.get("n_eff_divergent", np.nan)])
        a = np.array(acc, float)
        row = dict(eps=eps, truth_pi=truthE3["pi"], raw_mean=np.nanmean(a[:, 0]), raw_sd=np.nanstd(a[:, 0]), raw_finite=float(np.mean(np.isfinite(a[:, 0]))),
                   clip_mean=a[:, 1].mean(), clip_sd=a[:, 1].std(), ess=np.nanmean(a[:, 2]), e3_mean=a[:, 3].mean(), e3_sd=a[:, 3].std())
        out.append(row); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return out


def exp_C(seeds=20, n=40000, rounds=8):
    """Retrain and redeploy in S3, five arms, matched draws across arms (same seed sequence per arm)."""
    cfg = make_cfg(deploy=True); res = []
    for sd in range(seeds):
        base = sample_po(cfg, set(), n, np.random.default_rng(500 + sd))["y"].mean()
        def deploy(f, rng):
            x = rng.normal(size=(n, 2)); r = f(x); s = suspicion(cfg, x)
            p = np.where(s > cfg.tau_s, cfg.q_hi, np.where(r > cfg.tau_r, cfg.p_alert, cfg.eps))
            a = (rng.random(n) < p).astype(float); q = p_y1(cfg, x, a, False); y = (rng.random(n) < q).astype(float)
            return x, a, y, r
        f0 = lambda x: r_hat(cfg, x); arms = {}
        for arm in ["keep", "retrain", "unalerted", "untreated", "cond"]:
            rng = np.random.default_rng(10_000 + sd); f = f0; traj = []
            for t in range(rounds):
                x, a, y, r = deploy(f, rng); traj.append((float(y.mean()), float((r > cfg.tau_r).mean()), _auroc(r, y)))
                if arm == "retrain": clf = LogisticRegression(max_iter=1000).fit(x, y); f = (lambda c: lambda xx: c.predict_proba(xx)[:, 1])(clf)
                elif arm == "unalerted":
                    m = r <= cfg.tau_r; clf = LogisticRegression(max_iter=1000).fit(x[m], y[m]); f = (lambda c: lambda xx: c.predict_proba(xx)[:, 1])(clf)
                elif arm == "untreated":
                    m = a == 0; clf = LogisticRegression(max_iter=1000).fit(x[m], y[m]); f = (lambda c: lambda xx: c.predict_proba(xx)[:, 1])(clf)
                elif arm == "cond":
                    clf = LogisticRegression(max_iter=1000).fit(np.c_[x, a], y); f = (lambda c: lambda xx: c.predict_proba(np.c_[xx, np.zeros(len(xx))])[:, 1])(clf)
            arms[arm] = traj
        res.append(dict(base=float(base), **arms))
    b = np.mean([r["base"] for r in res]); print("\nexp C: standard-care event rate", round(b, 4))
    for arm in ["keep", "retrain", "unalerted", "untreated", "cond"]:
        ev = np.array([[t[0] for t in r[arm]] for r in res]); al = np.array([[t[1] for t in r[arm]] for r in res]); au = np.array([[t[2] for t in r[arm]] for r in res])
        print(f"  {arm:10s} averted(pp) {np.round(100*(b-ev.mean(0)),2)}  alert% {np.round(100*al.mean(0),1)}  AUROC {np.round(au.mean(0),3)}")
    return res


def exp_M():
    """Marginal contribution of the policy player to every coalition it can join (oracle coalition table)."""
    out = {}
    for sc, kw in SCEN.items():
        cfg = make_cfg(**kw); tv = truth_values(cfg, ["X", "pi", "Y"], n=600_000, metric="brier", seed=0)
        mc = {"to_empty": tv[frozenset({"pi"})] - tv[frozenset()], "to_X": tv[frozenset({"X", "pi"})] - tv[frozenset({"X"})],
              "to_Y": tv[frozenset({"pi", "Y"})] - tv[frozenset({"Y"})], "to_XY": tv[frozenset({"X", "pi", "Y"})] - tv[frozenset({"X", "Y"})]}
        mc["shapley"] = shapley(tv, ["X", "pi", "Y"])["pi"]; mc["I_Xpi"] = mc["to_X"] - mc["to_empty"]; mc["I_piY"] = mc["to_Y"] - mc["to_empty"]
        out[sc] = mc; print(sc, {k: round(v, 5) for k, v in mc.items()})
    return out


def _auroc(r, y):
    from sklearn.metrics import roc_auc_score; return float(roc_auc_score(y, r))


def exp_D(seeds=8, n=40000):
    """Observed AUROC, events averted, standard vs counterfactual net benefit across p_alert and theta."""
    out = []
    for theta in [0.5, 1.0, 1.5, 2.5]:
        for pa in [0.1, 0.3, 0.5, 0.7, 0.9]:
            cfg = make_cfg(theta=theta, p_alert=pa, deploy=True); acc = []
            for sd in range(seeds):
                rng = np.random.default_rng(700 + sd); s0 = sample_po(cfg, set(), n, rng); s1 = sample_po(cfg, {"pi"}, n, rng)
                t = cfg.tau_r; flag = s1["r"] > t
                nb_std = np.mean(flag & (s1["y"] == 1)) - np.mean(flag & (s1["y"] == 0)) * t / (1 - t)
                nb_cf = np.mean(flag & (s1["y0"] == 1)) - np.mean(flag & (s1["y0"] == 0)) * t / (1 - t)
                acc.append([_auroc(s0["r"], s0["y"]), _auroc(s1["r"], s1["y"]), _auroc(s1["r"], s1["y0"]), s0["y"].mean() - s1["y"].mean(), nb_std, nb_cf])
            a = np.mean(acc, 0); out.append(dict(theta=theta, p_alert=pa, auroc_e0=a[0], auroc_e1=a[1], auroc_cf=a[2], averted=a[3], nb_std=a[4], nb_cf=a[5]))
            print(f"theta={theta} p_alert={pa}: AUROC {a[0]:.3f}->{a[1]:.3f} (cf {a[2]:.3f}) averted {100*a[3]:.1f}pp NB std {a[4]:+.3f} cf {a[5]:+.3f}")
    return out


def exp_E(seeds=20, n=80000, h=0.04):
    """Threshold discontinuity: local jump vs global policy effect among the alerted (outcome scale)."""
    cfg = make_cfg(deploy=True); out = []
    for sd in range(seeds):
        rng = np.random.default_rng(900 + sd); s1 = sample_po(cfg, {"pi"}, n, rng); al = s1["r"] > cfg.tau_r
        true_global = float(np.mean((s1["y"] - s1["y0"])[al]))
        m = np.abs(s1["r"] - cfg.tau_r) < h; rr = s1["r"][m] - cfg.tau_r; side = (rr > 0).astype(float)
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(m.sum()), rr, side, rr * side], s1["y"][m], rcond=None)
        grid = np.linspace(cfg.tau_r, 0.95, 13); curve = []
        for g0, g1 in zip(grid[:-1], grid[1:]):
            mm = al & (s1["r"] >= g0) & (s1["r"] < g1); curve.append(float(np.mean((s1["y"] - s1["y0"])[mm])) if mm.sum() > 30 else np.nan)
        out.append(dict(true_global=true_global, theta_tau=float(coef[2]), curve=curve, grid=list(grid)))
    tg = np.mean([o["true_global"] for o in out]); tt = np.mean([o["theta_tau"] for o in out])
    print(f"\nexp E: true global {tg:+.4f}, RD local {tt:+.4f} ± {np.std([o['theta_tau'] for o in out]):.4f}, curve {np.round(np.nanmean([o['curve'] for o in out],0),3)}")
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    fn = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "M": exp_M}[which]
    res = fn(); json.dump(res, open(f"../figures/merged_exp{which}.json", "w"), indent=1, default=float)
