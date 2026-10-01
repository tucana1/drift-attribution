"""Figure 6 (E1): keep-or-refit decision rules across S1-S4, outcome level.

Events averted against standard care in the window after the decision, per
scenario and rule, under the fixed alert threshold (top) and the rate-held
threshold (bottom). Points are means over 200 independent replicates per
scenario; bars are 95% percentile bootstrap intervals over replicates.
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
RULES = (
    ("never", "never refit", "#228833", "o", True),
    ("always", "always refit (naive)", "#EE6677", "v", True),
    ("exo_proposed", "refit if exogenous share > 1/2 (proposed)", "#4477AA", "s", False),
    ("outcome_proposed", "refit if outcome-mechanism share > 1/2 (proposed)", "#4477AA", "s", True),
    ("outcome_monitor", "same rule, monitor without action log", "#EE7733", "D", True),
    ("refit_untreated", "reference: refit on $A=0$ (action log)", "#AA3377", "*", True),
)
SCENARIOS = (("S1", "S1\ncovariate"), ("S2", "S2\noutcome"), ("S3", "S3\ndeployment"), ("S4", "S4\nall three"))


def main():
    data = json.loads((ROOT / "figures/merged_expE1.json").read_text())
    table = data["table"]
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "axes.titlesize": 7.2})
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.9), sharex=True)
    offsets = np.linspace(-0.3, 0.3, len(RULES))
    for ax, rule_name, title in zip(axes, ("fixed", "rate"), ("fixed alert threshold", "rate-held alert threshold")):
        for (rule, label, color, marker, filled), off in zip(RULES, offsets):
            xs, ms, lo, hi = [], [], [], []
            for i, (sc, _) in enumerate(SCENARIOS):
                e = next(t for t in table if t["threshold_rule"] == rule_name and t["scenario"] == sc and t["rule"] == rule)
                v = e["events_averted_pp"]
                xs.append(i + off)
                ms.append(v["mean"])
                lo.append(v["mean"] - v["ci95"][0])
                hi.append(v["ci95"][1] - v["mean"])
            ax.errorbar(xs, ms, yerr=[lo, hi], fmt=marker, ms=4.2 if marker != "*" else 6, color=color,
                        mfc=color if filled else "white", mew=0.9, elinewidth=0.8, capsize=1.2, label=label, lw=0)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(title)
        ax.set_ylabel("events averted after\nthe decision (pp)")
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels([s[1] for s in SCENARIOS])
        for i in range(1, len(SCENARIOS)):
            ax.axvline(i - 0.5, color="#DDDDDD", lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.8, frameon=False, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.2, 1, 1), h_pad=0.5)
    save_figure(fig, "fig6_decision.pdf")
    plt.close(fig)
    print("wrote fig6_decision.pdf")


if __name__ == "__main__":
    main()
