"""Sign of the deployment-induced Brier change across outcome prevalence (appendix, with E4 and E6).

At the calibrated operating point the untreated event rate is about 40% and
alerted patients mostly have predicted risk above 0.5, so preventing their
events raises the Brier score (S3 is a performance drop). The derivative of
E[(r - Y)^2] in P(Y = 1) is 1 - 2r, so where alerted risks sit below 0.5, as
at realistic ward-deterioration prevalence, the same success lowers it. This
script lowers the outcome intercept b0, holds the alert rate at 20% (the
threshold is the 80th percentile of the score), and reports the oracle S3
change, the policy player's Shapley value and the monitor's outcome term,
each averaged over independent blocks of expected Brier scores (no outcome
noise), plus the expected observed AUROC before and after deployment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import expected_auroc, rng_for, seed_block, summarize, write_json
from dgp import p_treat, p_y1, r_hat, shapley
from exp_AB_robustness import truth_values_expected
from exp_E7_intervals import naive2_reference
from merged import make_cfg

EXPERIMENT = 8
B0_GRID = (-0.6, -1.5, -2.5, -3.5)
ALERT_RATE = 0.20
N, BLOCKS = 400_000, 4


def main():
    rows = []
    for i, b0 in enumerate(B0_GRID):
        base = make_cfg(b0=b0, deploy=True)
        x_ref = rng_for(2026092908, EXPERIMENT, i, 99).normal(size=(N, base.d))
        tau = float(np.quantile(r_hat(base, x_ref), 1 - ALERT_RATE))
        cfg = make_cfg(b0=b0, tau_r=tau, deploy=True)
        blocks = []
        for b in range(BLOCKS):
            tv = truth_values_expected(cfg, N, seed_block(2026092908, EXPERIMENT, i, b))
            mon = naive2_reference(cfg, N, rng_for(2026092908, EXPERIMENT, 100 + i, b))
            x = rng_for(2026092908, EXPERIMENT, 200 + i, b).normal(size=(N, cfg.d))
            r = r_hat(cfg, x)
            q0, q1 = p_y1(cfg, x, np.zeros(N), False), p_y1(cfg, x, np.ones(N), False)
            blocks.append({"dR": tv[frozenset({"X", "pi", "Y"})] - tv[frozenset()],
                           "policy_shapley": shapley(tv, ["X", "pi", "Y"])["pi"],
                           "monitor_Ygx": mon["Ygx"],
                           "auroc_before": expected_auroc(r, q0 + p_treat(cfg, x, False) * (q1 - q0)),
                           "auroc_after": expected_auroc(r, q0 + p_treat(cfg, x, True) * (q1 - q0)),
                           "event_rate_untreated": float(q0.mean()),
                           "mean_score_alerted": float(r[r > tau].mean())})
        row = {"b0": b0, "tau_r": tau, "alert_rate": ALERT_RATE}
        for k in blocks[0]:
            row[k] = summarize([blk[k] for blk in blocks], seed=i)
        rows.append(row)
        print(f"b0={b0:+.1f} event rate {row['event_rate_untreated']['mean']:.3f} dR {row['dR']['mean']:+.4f} "
              f"monitor Y|X {row['monitor_Ygx']['mean']:+.4f}", flush=True)
    write_json("merged_expP.json", {"design": __doc__.split("\n\n")[1].replace("\n", " "),
                                    "parameters": {"b0_grid": list(B0_GRID), "alert_rate": ALERT_RATE, "n": N,
                                                   "blocks": BLOCKS, "root_seed": 2026092908,
                                                   "experiment_block": EXPERIMENT},
                                    "rows": rows})
    print("wrote merged_expP.json")


if __name__ == "__main__":
    main()
