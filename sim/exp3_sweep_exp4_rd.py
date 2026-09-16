import numpy as np, json, sys
sys.path.insert(0, ".")
from core import *
from exp1_scenarios import BASE


def envs_S3(beta_A, p_adh, tau=0.3, rollout_control=0.3):
    kw = dict(BASE); kw["beta_A"] = beta_A; kw["p_adh"] = p_adh; kw["tau"] = tau
    return Env(policy="pi0", **kw), Env(policy="pi1", rollout_control=rollout_control, **kw)


def sweep(seeds=8, n=30000):
    rows = []
    for beta_A in [-0.5, -1.0, -1.5, -2.5]:
        for p_adh in [0.1, 0.3, 0.5, 0.7, 0.9]:
            acc = []
            for sd in range(seeds):
                rng = np.random.default_rng(1000 + sd)
                e0, e1 = envs_S3(beta_A, p_adh)
                st = sample_env(e0, lambda X: np.zeros(len(X)), n, rng); f, _ = train_risk_model(st.X, st.Y)
                s0 = sample_env(e0, f, n, rng); s1 = sample_env(e1, f, n, rng); ratio = density_ratio_X(s0.X, s1.X)
                e1r = E1_marginal(s0, s1, f, ratio); e3 = E3_proposed(s0, s1, f, ratio, e1, "oracle")
                share_Y = e1r["phi_YgX"] / e1r["dR"] if abs(e1r["dR"]) > 1e-4 else np.nan
                m = s1.arm == 1
                s1m = Sample(*[getattr(s1, k)[m] for k in ["X", "r", "A", "Y", "Y0", "Y1", "alerted", "arm", "p_pi0", "p_pi1"]])
                acc.append([e1r["dR"], share_Y, e3["delta_pi"] / e1r["dR"] if abs(e1r["dR"]) > 1e-4 else np.nan,
                            auroc(s0), auroc(s1m), auroc(s1m, True),
                            policy_value_events(s0) - policy_value_events(s1m),
                            counterfactual_net_benefit(s1m, e1.tau, False), counterfactual_net_benefit(s1m, e1.tau, True),
                            counterfactual_net_benefit(s0, e0.tau, True)])
            a = np.nanmean(np.array(acc), 0)
            rows.append(dict(beta_A=beta_A, p_adh=p_adh, dR=a[0], E1_share_YgX=a[1], E3_share_policy=a[2], auroc_e0=a[3],
                             auroc_e1_obs=a[4], auroc_e1_cf=a[5], events_averted=a[6], nb_standard=a[7], nb_counterfactual=a[8], nb_e0=a[9]))
            print(f"beta_A={beta_A:5.1f} p_adh={p_adh:.1f}  dR={a[0]:.4f}  E1 share->Y|X={a[1]:.2f}  E3 share->policy={a[2]:.2f}  "
                  f"AUROC {a[3]:.3f}->{a[4]:.3f} (cf {a[5]:.3f})  averted={a[6]:.4f}  NB std={a[7]:.3f} cf={a[8]:.3f}")
    return rows


def rd_experiment(seeds=20, n=60000, beta_A=-1.5, p_adh=0.7, tau=0.3):
    """Compare the threshold-discontinuity local effect with the global policy term in outcome units."""
    out = []
    for sd in range(seeds):
        rng = np.random.default_rng(2000 + sd)
        e0, e1 = envs_S3(beta_A, p_adh, tau, rollout_control=0.0)
        st = sample_env(e0, lambda X: np.zeros(len(X)), n, rng); f, _ = train_risk_model(st.X, st.Y)
        s1 = sample_env(e1, f, n, rng)
        # truth: global policy effect on events among the alerted (oracle potential outcomes)
        al = s1.alerted
        true_global = float(np.mean((s1.Y - s1.Y0)[al]))                      # mean change in outcome among alerted
        # effect curve theta(r): E[Y - Y0 | r] on a grid
        grid = np.linspace(tau, 0.95, 14); theta_r = []
        for g0, g1 in zip(grid[:-1], grid[1:]):
            m = al & (s1.r >= g0) & (s1.r < g1); theta_r.append(float(np.mean((s1.Y - s1.Y0)[m])) if m.sum() > 30 else np.nan)
        theta_tau = rd_local_effect(s1, tau, h=0.05)
        # homogeneity extrapolation and monotone bounds
        homog = theta_tau
        theta_max = float(np.nanmin(theta_r))                                   # most negative (largest) effect along the curve
        mass_near = float(np.mean(al & (s1.r < tau + 0.1)) / np.mean(al))
        out.append(dict(true_global=true_global, theta_tau=theta_tau, homog=homog, bound_lo=theta_max, bound_hi=theta_tau,
                        mass_near=mass_near, theta_curve=theta_r, grid=list(grid)))
    tg = np.mean([o["true_global"] for o in out]); tt = np.mean([o["theta_tau"] for o in out])
    print(f"\nRD: true global effect among alerted = {tg:.4f}; local jump at tau = {tt:.4f} (sd {np.std([o['theta_tau'] for o in out]):.4f}); "
          f"homogeneity extrapolation error = {tt - tg:+.4f}; monotone bounds [{np.mean([o['bound_lo'] for o in out]):.4f}, {np.mean([o['bound_hi'] for o in out]):.4f}]")
    print("effect curve theta(r):", np.round(np.nanmean([o["theta_curve"] for o in out], 0), 3))
    return out


if __name__ == "__main__":
    rows = sweep(); json.dump(rows, open("../figures/exp3_sweep.json", "w"), indent=1, default=float)
    rd = rd_experiment(); json.dump(rd, open("../figures/exp4_rd.json", "w"), indent=1, default=float)
