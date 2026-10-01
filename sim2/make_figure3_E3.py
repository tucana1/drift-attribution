"""Figure 3 from merged_expH.json (E3): update rules after deployment.

Rows: fixed alert threshold, rate-held threshold. The first three columns share
one vertical scale (events averted against standard care, percentage points):
homogeneous effect, heterogeneous effect (kappa = 2), misspecified score class.
The fourth column shows what monitoring sees for naive refitting: observed
AUROC under the fixed threshold, calibration against the untreated risk under
the rate-held threshold. Bands are 95% bootstrap intervals over seeds.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_figure

ROOT = Path(__file__).resolve().parents[1]
STYLES = (
    ("keep", "keep original score", "#228833", "-", 1.6),
    ("naive", "naive refit", "#EE6677", "-", 1.2),
    ("unalerted", "refit below threshold", "#CCBB44", "--", 1.0),
    ("conditional_main", "condition on $A$, main effects", "#999999", "-.", 1.0),
    ("untreated", "refit on $A=0$", "#66CCEE", "--", 1.0),
    ("conditional_interactions", "condition on $A$ with $X\\times A$", "#4477AA", ":", 1.2),
    ("untreated_ipw", "refit on $A=0$, weighted by $1/P(A=0\\mid X)$", "#AA3377", (0, (1, 1)), 1.4),
)
COLUMNS = (("k0", "homogeneous effect"), ("k2", "heterogeneous effect, $\\kappa=2$"),
           ("nl", "misspecified score class"))


def main():
    data = json.loads((ROOT / "figures/merged_expH.json").read_text())
    summary = data["summary"]
    rounds = np.arange(1, data["parameters"]["rounds"] + 1)

    def series(setting, arm, key):
        cells = sorted((e for e in summary if e["setting"] == setting and e["arm"] == arm), key=lambda e: e["round"])
        mean = np.array([c[key]["mean"] for c in cells])
        lo = np.array([c[key]["ci95"][0] for c in cells])
        hi = np.array([c[key]["ci95"][1] for c in cells])
        return mean, lo, hi

    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titlesize": 7.2})
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.55), gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
    values = [e["events_averted_pp"]["ci95"][k] for e in summary for k in (0, 1)]
    ylim = (min(-1.0, min(values) - 0.5), max(values) + 0.8)
    for row, rule in enumerate(("fixed", "rate")):
        for col, (suffix, title) in enumerate(COLUMNS):
            ax = axes[row, col]
            setting = f"{rule}_{suffix}"
            for arm, label, color, ls, lw in STYLES:
                m, lo, hi = series(setting, arm, "events_averted_pp")
                ax.plot(rounds, m, color=color, ls=ls, lw=lw, marker="o", ms=1.8, label=label)
                ax.fill_between(rounds, lo, hi, color=color, alpha=0.18, lw=0)
            ax.axhline(0, color="black", lw=0.6, ls="--")
            ax.set_ylim(*ylim)
            ax.set_xticks(rounds)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(("fixed threshold" if rule == "fixed" else "rate-held threshold")
                              + "\nevents averted (pp)")
            else:
                ax.set_yticklabels([])
            if row == 1:
                ax.set_xlabel("deployment round")
    # monitoring view, naive refit versus keep, homogeneous effect
    ax = axes[0, 3]
    for arm, color in (("keep", "#228833"), ("naive", "#EE6677")):
        m, lo, hi = series("fixed_k0", arm, "observed_auroc")
        ax.plot(rounds, m, color=color, marker="o", ms=1.8, lw=1.1)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.2, lw=0)
    ax.set_title("dashboard: observed AUROC\n(fixed, homogeneous)")
    ax.set_xticks(rounds)
    ax = axes[1, 3]
    for arm, color in (("keep", "#228833"), ("naive", "#EE6677")):
        m, lo, hi = series("rate_k0", arm, "calibration_in_large")
        ax.plot(rounds, m, color=color, marker="o", ms=1.8, lw=1.1)
        ax.fill_between(rounds, lo, hi, color=color, alpha=0.2, lw=0)
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_title("mean score minus untreated risk\n(rate-held, homogeneous)")
    ax.set_xticks(rounds)
    ax.set_xlabel("deployment round")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, fontsize=6.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.085, 1, 1), h_pad=0.6, w_pad=0.4)
    save_figure(fig, "fig3_retrain.pdf")
    plt.close(fig)
    print("wrote fig3_retrain.pdf")


if __name__ == "__main__":
    main()
