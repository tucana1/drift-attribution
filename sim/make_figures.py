import json, numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
FIG = Path(__file__).resolve().parents[1] / "figures"
MIRROR = FIG.parent / "aaai" / "figures"


def save(name):
    """Save without a creation date (reproducible bytes) and copy to aaai/figures/."""
    plt.savefig(FIG / name, metadata={"CreationDate": None}); plt.close()
    (MIRROR / name).write_bytes((FIG / name).read_bytes())


C_X, C_Y, C_P = "#4477AA", "#EE6677", "#228833"   # Tol bright: blue, red, green

# ---------------------------------------------------------------- Fig 1: attribution by scenario, E1 vs E3
res = json.loads((FIG / "exp1_results.json").read_text())
scen = ["S1", "S2", "S3", "S4"]; labels = ["S1 covariate\nshift", "S2 outcome\nmechanism", "S3 performative\nsuccess", "S4 all three"]
def stats(sc, est, key):
    v = np.array([r[est][key] for r in res if r["scenario"] == sc]); return v.mean(), v.std()
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=False)
for ax, sc, lab in zip(axes, scen, labels):
    e1 = [stats(sc, "E1", "phi_X"), stats(sc, "E1", "phi_YgX")]
    e3 = [stats(sc, "E3_oracle", "phi_X"), stats(sc, "E3_oracle", "phi_YgXA"), stats(sc, "E3_oracle", "delta_pi")]
    xs = np.array([0, 1]); ax.bar(xs - 0.18, [e1[0][0], e1[1][0]], 0.34, yerr=[e1[0][1], e1[1][1]], color=[C_X, C_Y], alpha=0.45, hatch="//", edgecolor="k", lw=0.4, label="E1 marginal")
    xs3 = np.array([0, 1, 2]); ax.bar(xs3 + 0.18, [e3[0][0], e3[1][0], e3[2][0]], 0.34, yerr=[e3[0][1], e3[1][1], e3[2][1]], color=[C_X, C_Y, C_P], edgecolor="k", lw=0.4, label="E3 proposed")
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["P(X)", "P(Y|X)\nor P(Y|X,A)", "policy\n$\\pi_0\\to\\pi_1$"], fontsize=6.5)
    ax.set_title(lab, fontsize=8); ax.tick_params(labelsize=6.5)
    dR = stats(sc, "E1", "dR")[0]; ax.text(0.98, 0.95, f"$\\Delta R$ = {dR:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.5)
axes[0].set_ylabel("attributed change in log loss")
h = [plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=0.45, hatch="//", ec="k", lw=0.4), plt.Rectangle((0, 0), 1, 1, fc="grey", ec="k", lw=0.4)]
axes[0].legend(h, ["E1: marginal Shapley (A not logged)", "E3: exogenous Shapley + policy term"], loc="upper left", fontsize=6, frameon=False, bbox_to_anchor=(0, 1.02))
plt.tight_layout(); save("fig_attribution_scenarios.pdf")

# ---------------------------------------------------------------- Fig 2: retrain and redeploy
r2 = json.loads((FIG / "exp2_results.json").read_text()); base = np.mean([r["base_events"] for r in r2])
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
arms = [("keep", "keep $f$ (attribution says: policy, do not retrain)", C_P), ("retrain", "naive retrain on post-deployment $(X,Y)$", C_Y), ("cond", "retrain conditioning on $A$, predict at $A{=}0$", C_X)]
for arm, lab, c in arms:
    ev = np.array([[x[0] for x in r[arm]] for r in r2]); al = np.array([[x[1] for x in r[arm]] for r in r2]); au = np.array([[x[2] for x in r[arm]] for r in r2])
    t = np.arange(1, ev.shape[1] + 1)
    for ax, y, name in zip(axes, [100 * (base - ev), 100 * al, au], ["events averted vs standard care (pp)", "alert rate (%)", "observed AUROC on deployed population"]):
        ax.errorbar(t, y.mean(0), y.std(0), marker="o", ms=3, lw=1, capsize=2, color=c, label=lab); ax.set_title(name, fontsize=7.5); ax.set_xticks(t); ax.set_xlabel("deployment round")
axes[0].legend(fontsize=6, frameon=False, loc="lower left"); plt.tight_layout(); save("fig_retrain_redeploy.pdf")

# ---------------------------------------------------------------- Fig 3: AUROC vs utility across adherence
sw = json.loads((FIG / "exp3_sweep.json").read_text())
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
for beta_A, c in zip([-0.5, -1.0, -1.5, -2.5], ["#CCBB44", "#66CCEE", "#4477AA", "#AA3377"]):
    rows = sorted([r for r in sw if r["beta_A"] == beta_A], key=lambda r: r["p_adh"]); p = [r["p_adh"] for r in rows]
    axes[0].plot(p, [r["auroc_e1_obs"] for r in rows], marker="o", ms=3, color=c, label=f"$\\beta_A$ = {beta_A}")
    axes[1].plot(p, [100 * r["events_averted"] for r in rows], marker="o", ms=3, color=c)
    axes[2].plot(p, [r["nb_standard"] for r in rows], marker="o", ms=3, color=c, ls="--"); axes[2].plot(p, [r["nb_counterfactual"] for r in rows], marker="s", ms=3, color=c)
axes[0].axhline(sw[0]["auroc_e0"], color="k", lw=0.6, ls=":"); axes[0].text(0.12, sw[0]["auroc_e0"] + 0.003, "pre-deployment", fontsize=6)
axes[0].set_title("observed AUROC after deployment", fontsize=7.5); axes[1].set_title("events averted (pp)", fontsize=7.5); axes[2].set_title("net benefit at $\\tau$: standard DCA (dashed)\nvs counterfactual (solid)", fontsize=7.5)
axes[2].axhline(0, color="k", lw=0.5)
for ax in axes: ax.set_xlabel("clinician adherence $p_{adh}$")
axes[0].legend(fontsize=6, frameon=False); plt.tight_layout(); save("fig_auroc_vs_utility.pdf")

# ---------------------------------------------------------------- Fig 4: RD local effect vs global policy term
rd = json.loads((FIG / "exp4_rd.json").read_text()); grid = np.array(rd[0]["grid"]); mid = (grid[:-1] + grid[1:]) / 2
curve = np.nanmean([o["theta_curve"] for o in rd], 0); curve_sd = np.nanstd([o["theta_curve"] for o in rd], 0)
tg = np.mean([o["true_global"] for o in rd]); tt = np.mean([o["theta_tau"] for o in rd]); tt_sd = np.std([o["theta_tau"] for o in rd])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.errorbar(mid, curve, curve_sd, marker="o", ms=3, lw=1, color=C_P, capsize=2, label="effect curve $\\theta(r)$ among alerted (oracle)")
ax.errorbar([grid[0]], [tt], [tt_sd], marker="D", ms=5, color=C_Y, capsize=3, label="RD local jump at $\\tau$")
ax.axhline(tg, color=C_X, lw=1, label="true global policy term (mean over alerted)")
ax.axhspan(np.mean([o["bound_lo"] for o in rd]), np.mean([o["bound_hi"] for o in rd]), color="grey", alpha=0.15, label="bounds [min $\\theta$, $\\theta(\\tau)$]")
ax.set_xlabel("risk score $\\hat r$"); ax.set_ylabel("change in event probability"); ax.legend(fontsize=5.5, frameon=False, loc="lower left")
plt.tight_layout(); save("fig_rd_bounds.pdf")
print("figures written")
