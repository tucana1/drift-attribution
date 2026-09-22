"""Regenerate Figure 3 from the saved E3 run, using a shared y scale."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "figures/merged_expH.json").read_text())
styles = (
    ("keep", "Keep original score", "#228833", "-"),
    ("naive", "Naive refit", "#EE6677", "-"),
    ("unalerted", "Refit below alert threshold", "#CCBB44", "--"),
    ("untreated", "Refit untreated patients", "#66CCEE", "--"),
    ("conditional_interactions", "Condition on action and interactions", "#4477AA", ":"),
)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharex=True, sharey=True)
all_values = [row["mean_events_averted_pp"] for row in data["summary"]]
ymin = min(-0.5, min(all_values) - 0.5)
ymax = max(all_values) + 1.0
for ax, kappa in zip(axes, (0.0, 2.0)):
    for arm, label, color, linestyle in styles:
        cells = [row for row in data["summary"] if row["kappa"] == kappa and row["arm"] == arm]
        x = np.array([row["round"] for row in cells])
        y = np.array([row["mean_events_averted_pp"] for row in cells])
        se = np.array([row["mcse_pp"] for row in cells])
        ax.plot(x, y, marker="o", markersize=2.2, linewidth=1.0,
                color=color, linestyle=linestyle, label=label)
        ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=color, alpha=0.10)
    ax.axhline(0, color="black", linewidth=0.65, linestyle="--")
    ax.set_title(rf"Treatment heterogeneity $\kappa={kappa:g}$")
    ax.set_xlabel("Deployment round")
    ax.set_xticks(range(1, data["parameters"]["rounds"] + 1))
    ax.set_ylim(ymin, ymax)
axes[0].set_ylabel("Events averted vs standard care (pp)")
axes[1].legend(fontsize=6.2, frameon=False, loc="lower right")
fig.tight_layout()
for folder in (ROOT / "figures", ROOT / "aaai/figures"):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / "fig3_retrain.pdf", bbox_inches="tight")
plt.close(fig)
print("Wrote Figure 3 in both figure directories")
