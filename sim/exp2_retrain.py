import numpy as np, json, sys
sys.path.insert(0, ".")
from core import *
from exp1_scenarios import make_envs, BASE


def deploy_round(env, f, n, rng):
    return sample_env(env, f, n, rng)


def run(seed, n=40000, rounds=4):
    rng = np.random.default_rng(seed)
    e0, e1 = make_envs("S3", rollout_control=0.0)
    s_train = sample_env(e0, lambda X: np.zeros(len(X)), n, rng)
    f0, _ = train_risk_model(s_train.X, s_train.Y)
    base_events = policy_value_events(sample_env(e0, f0, n, rng))            # standard care, no model
    # arm K: keep f0 deployed every round
    keep = []; f = f0
    for t in range(rounds):
        s = deploy_round(e1, f, n, rng); keep.append((policy_value_events(s), float(s.alerted.mean()), auroc(s)))
    # arm R: naive retraining on the previous round's observational (X, Y) each round (misattribution acted on)
    retrain = []; f = f0
    for t in range(rounds):
        s = deploy_round(e1, f, n, rng); retrain.append((policy_value_events(s), float(s.alerted.mean()), auroc(s)))
        f, _ = train_risk_model(s.X, s.Y)                                      # learns E_e1[Y|X] under pi1
    # arm C: retraining that respects the policy: fit on untreated-behaviour data, i.e. condition on A and predict at A=0
    cond = []; f = f0
    for t in range(rounds):
        s = deploy_round(e1, f, n, rng); cond.append((policy_value_events(s), float(s.alerted.mean()), auroc(s)))
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.c_[s.X, s.A], s.Y)
        f = (lambda c: (lambda X: c.predict_proba(np.c_[X, np.zeros(len(X))])[:, 1]))(clf)   # predict under A=0
    return {"seed": seed, "base_events": base_events, "keep": keep, "retrain": retrain, "cond": cond}


if __name__ == "__main__":
    res = [run(s) for s in range(20)]
    json.dump(res, open("../figures/exp2_results.json", "w"), indent=1, default=float)
    base = np.mean([r["base_events"] for r in res])
    print("standard care event rate (no model): %.4f" % base)
    for arm in ["keep", "retrain", "cond"]:
        ev = np.array([[x[0] for x in r[arm]] for r in res]); al = np.array([[x[1] for x in r[arm]] for r in res]); au = np.array([[x[2] for x in r[arm]] for r in res])
        print(f"\n{arm}: events per round", np.round(ev.mean(0), 4), "\n      events averted vs standard care", np.round(base - ev.mean(0), 4),
              "\n      alert rate", np.round(al.mean(0), 3), "\n      observed AUROC", np.round(au.mean(0), 3))
