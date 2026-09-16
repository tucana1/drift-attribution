import numpy as np, json, sys
sys.path.insert(0, ".")
from core import *

D = 5
BASE = dict(mu=np.zeros(D), beta=np.array([1.2, 0.8, -0.6, 0.5, 0.0]), b=-1.6, beta_A=-1.5,
            tau=0.3, p_adh=0.7)


def make_envs(scenario, rollout_control=0.0):
    e0 = Env(policy="pi0", **BASE)
    kw = dict(BASE)
    if scenario in ("S1", "S4"):
        kw["mu"] = BASE["mu"] + np.array([0.6, 0.3, 0.0, 0.0, 0.0])
    if scenario in ("S2", "S4"):
        kw["beta"] = BASE["beta"] + np.array([0.0, -0.5, 0.0, 0.0, 0.6]); kw["b"] = BASE["b"] + 0.4
    policy = "pi1" if scenario in ("S3", "S4") else "pi0"
    e1 = Env(policy=policy, rollout_control=rollout_control if policy == "pi1" else 0.0, **kw)
    return e0, e1


def one_run(scenario, seed, n=40000, rollout_control=0.3):
    rng = np.random.default_rng(seed)
    e0, e1 = make_envs(scenario, rollout_control)
    # train f on a separate pre-deployment sample under standard care
    s_train = sample_env(e0, lambda X: np.zeros(len(X)), n, rng)
    f, clf = train_risk_model(s_train.X, s_train.Y)
    s0 = sample_env(e0, f, n, rng); s1 = sample_env(e1, f, n, rng)
    ratio = density_ratio_X(s0.X, s1.X)
    out = {"scenario": scenario, "seed": seed}
    out["E1"] = E1_marginal(s0, s1, f, ratio)
    out["E2"] = E2_zhang_with_action(s0, s1, f, ratio, e1)
    out["E3_oracle"] = E3_proposed(s0, s1, f, ratio, e1, "oracle")
    out["E3_rollout"] = E3_proposed(s0, s1, f, ratio, e1, "rollout") if e1.policy == "pi1" else None
    out["metrics"] = {"auroc_e0": auroc(s0), "auroc_e1_obs": auroc(s1), "auroc_e1_untreated_truth": auroc(s1, True),
                      "events_e0": policy_value_events(s0), "events_e1": policy_value_events(s1),
                      "nb_e1_standard": counterfactual_net_benefit(s1, e1.tau, False),
                      "nb_e1_counterfactual": counterfactual_net_benefit(s1, e1.tau, True),
                      "alert_rate_e1": float(s1.alerted.mean()), "treat_rate_e0": float(s0.A.mean()),
                      "treat_rate_e1": float(s1.A.mean())}
    return out


if __name__ == "__main__":
    seeds = range(20)
    res = [one_run(sc, sd) for sc in ["S1", "S2", "S3", "S4"] for sd in seeds]
    json.dump(res, open("../figures/exp1_results.json", "w"), indent=1, default=float)

    def agg(sc, path):
        vals = []
        for r in res:
            if r["scenario"] != sc: continue
            v = r
            for k in path: v = v[k] if v is not None else None
            if v is not None: vals.append(v)
        vals = np.array(vals); return vals.mean(), vals.std()

    print(f"{'scen':<5}{'dR':>9}{'E1 phi_X':>11}{'E1 phi_Y|X':>12}{'E3 phi_X':>11}{'E3 phi_Y|XA':>12}{'E3 d_pi':>10}{'E2 undef':>10}{'nosupp':>8}")
    for sc in ["S1", "S2", "S3", "S4"]:
        dR = agg(sc, ["E1", "dR"]); e1x = agg(sc, ["E1", "phi_X"]); e1y = agg(sc, ["E1", "phi_YgX"])
        e3x = agg(sc, ["E3_oracle", "phi_X"]); e3y = agg(sc, ["E3_oracle", "phi_YgXA"]); e3p = agg(sc, ["E3_oracle", "delta_pi"])
        und = agg(sc, ["E2", "frac_undefined_weights"]); ns = agg(sc, ["E2", "no_support_mass_e1"])
        print(f"{sc:<5}{dR[0]:>9.4f}{e1x[0]:>11.4f}{e1y[0]:>12.4f}{e3x[0]:>11.4f}{e3y[0]:>12.4f}{e3p[0]:>10.4f}{und[0]:>10.3f}{ns[0]:>8.3f}")
    print("\nrollout-arm estimate of the policy term (S3, S4):")
    for sc in ["S3", "S4"]:
        print(sc, "oracle d_pi = %.4f +- %.4f" % agg(sc, ["E3_oracle", "delta_pi"]), " rollout d_pi = %.4f +- %.4f" % agg(sc, ["E3_rollout", "delta_pi"]))
    print("\nutility metrics (S3):")
    for k in ["auroc_e0", "auroc_e1_obs", "auroc_e1_untreated_truth", "events_e0", "events_e1", "nb_e1_standard", "nb_e1_counterfactual", "alert_rate_e1", "treat_rate_e0", "treat_rate_e1"]:
        print(f"  {k:<26} {agg('S3', ['metrics', k])[0]:.4f}")
