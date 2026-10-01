"""Does a working alert make the deployed score look worse? Exact before/after metrics.

    python3 sim2/check_performance_drop.py

For a deployment that only changes the action policy (S3), compares the
score's performance on outcomes before deployment (standard care) and after
(alert policy), in expectation over outcomes and actions: Brier score, AUROC,
observed / expected events and calibration slope. Synthetic: the calibrated
operating point and the prevalence grid of exp_prevalence.py (alert rate 20%).
Open stand-in (if data/e6_sepsis_cohort.npz exists): each hospital as the
deployment site, score fitted to 40% of its patients, 10% alert rate, an alert
acted upon with probability 0.85 that prevents the event with probability 0.5,
metrics on the held-out patients. Writes figures/performance_drop.json.

Discrimination and calibration fall at every prevalence; the Brier score rises
at the calibrated operating point and falls at low prevalence, where preventing
events among alerted patients with predicted risk below 0.5 lowers the squared
error more than the miscalibration raises it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, expected_auroc, rng_for, write_json
from dgp import p_treat, p_y1, r_hat
from merged import make_cfg, make_cfg_at

EXPERIMENT = 9


def calibration_slope(f, q):
    """Slope of a logistic recalibration of the outcome on logit(score), outcome probabilities q as weights."""
    z = logit(np.clip(f, 1e-6, 1 - 1e-6))[:, None]
    model = LogisticRegression(C=1e6, max_iter=1000).fit(np.r_[z, z], np.r_[np.ones(len(q)), np.zeros(len(q))],
                                                         sample_weight=np.r_[q, 1 - q])
    return float(model.coef_[0, 0])


def metrics(f, q):
    return {"brier": float(np.mean(f ** 2 + (1 - 2 * f) * q)), "auroc": expected_auroc(f, q),
            "observed_over_expected": float(q.mean() / f.mean()), "calibration_slope": calibration_slope(f, q),
            "event_rate": float(q.mean())}


def before_after(f, q_before, q_after):
    b, a = metrics(f, q_before), metrics(f, q_after)
    return {"before": b, "after": a, "change": {k: a[k] - b[k] for k in b}}


def synthetic(n=400_000):
    rows = []
    x = rng_for(2026092909, EXPERIMENT, 0).normal(size=(n, 2))
    settings = [("calibrated operating point", make_cfg(deploy=True))]
    settings += [(f"b0 = {b0:g}, alert rate 20%", make_cfg_at(b0, 0.2, deploy=True)) for b0 in (-0.6, -1.5, -2.5, -3.5)]
    for label, cfg in settings:
        f = r_hat(cfg, x)
        q0, q1 = p_y1(cfg, x, np.zeros(n), False), p_y1(cfg, x, np.ones(n), False)
        rows.append({"setting": label, "b0": cfg.b0, "tau_r": cfg.tau_r,
                     **before_after(f, q0 + p_treat(cfg, x, False) * (q1 - q0), q0 + p_treat(cfg, x, True) * (q1 - q0))})
    return rows


def open_stand_in(path=ROOT / "data/e6_sepsis_cohort.npz", p_prevent=0.85 * 0.5, alert_rate=0.1):
    if not path.exists():
        return None
    from exp_E6_mimic import Cohort
    rows = []
    for site, other in (("A", "B"), ("B", "A")):
        c = Cohort(path, [site], [other], 0.4)
        model = LogisticRegression(C=1.0, max_iter=5000).fit(c.x[c.model_train], c.y[c.model_train])
        f_train = model.predict_proba(c.x[c.model_train])[:, 1]
        tau = float(np.quantile(f_train, 1 - alert_rate))
        pool = c.early & ~c.model_train
        f, y = model.predict_proba(c.x[pool])[:, 1], c.y[pool]
        rows.append({"deployment_site": site, "patients": int(pool.sum()), "tau": tau,
                     **before_after(f, y, y * (1 - p_prevent * (f > tau)))})
    return rows


def main():
    out = {"design": __doc__.split("\n\n")[1].replace("\n", " "), "synthetic": synthetic(), "open_stand_in": open_stand_in()}
    write_json("performance_drop.json", out)
    for r in out["synthetic"] + (out["open_stand_in"] or []):
        label = r.get("setting") or f"open stand-in, hospital {r['deployment_site']}"
        ch = r["change"]
        print(f"{label:34s} event rate {r['before']['event_rate']:.3f}  Brier {ch['brier']:+.4f}  AUROC {ch['auroc']:+.3f}  "
              f"O/E {r['before']['observed_over_expected']:.2f} -> {r['after']['observed_over_expected']:.2f}  "
              f"slope {r['before']['calibration_slope']:.2f} -> {r['after']['calibration_slope']:.2f}")


if __name__ == "__main__":
    main()
