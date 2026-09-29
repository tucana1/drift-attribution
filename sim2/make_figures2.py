"""Regenerate every main-text figure from the saved JSON evidence (one command, task R4).

    python3 sim2/make_figures2.py

Figures 1, 2, 4 and 5 read merged_expE7.json (independent seed block per row,
replicates stored); Figure 3 reads merged_expH.json (E3); Figure 6 reads
merged_expE1.json (E1); Figure 7 is written by exp_E6_mimic.py next to the
credentialed data and is only copied if present. Unless a caption says
otherwise, points are means over replicates and thick bars or bands are 95%
percentile bootstrap intervals of that mean; thin light bars show the central
95% of single replicates, i.e. the spread one analyst's estimate could have.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
FIG_DIRS = (ROOT / "figures", ROOT / "aaai/figures")
CX, CP, CY, CG = "#4477AA", "#228833", "#EE6677", "#888888"   # X, policy, Y|X(,A), grey
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.titlesize": 8})


def save(fig, name):
    from common import save_figure
    save_figure(fig, name, FIG_DIRS)
    plt.close(fig)
    print(f"wrote {name}")


def spread(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return np.quantile(a, [0.025, 0.975]) if len(a) else (np.nan, np.nan)


def figure1(E7):
    A = E7["A"]
    rows = (("Shapley game, oracle coalition values", "ref_shapley"),
            ("policy-contrast estimand (target)", "ref_pc"),
            ("union graph IS, unclipped", "union3_raw"),
            ("union graph IS, weights clipped at 100", "union3_clip"),
            ("no action log: $\\{P(X), P(Y\\mid X)\\}$", "naive2_est"),
            ("proposed, randomised arm", "E3_rollout"))
    comps = (("X", CX, "o", "patient mix $P(X)$"), ("pi", CP, "D", "alerting policy $\\pi$"),
             ("Y", CY, "s", "outcome mechanism $P(Y\\mid X,A)$, or $P(Y\\mid X)$ without $A$"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), sharey=True)
    for ax, sc, title in zip(axes, ("S3", "S4"), ("S3: performative success only",
                                                   "S4: covariate + outcome shift + deployment")):
        ref, runs = A["references"][sc], [r for r in A["runs"] if r["scenario"] == sc]
        summ = A["summary"][sc]
        ys = np.arange(len(rows))[::-1]
        for y, (label, key) in zip(ys, rows):
            for j, (comp, color, marker, _) in enumerate(comps):
                dy = (j - 1) * 0.2
                if key == "ref_shapley":
                    ax.plot(ref[f"shapley_{comp}"]["value"], y + dy, marker, color=color, ms=4, mfc="white", mew=1.0)
                    continue
                if key == "ref_pc":
                    ax.plot(ref[f"pc_{comp}"]["value"], y + dy, marker, color=color, ms=4)
                    continue
                col = f"{key}_{'Ygx' if (key == 'naive2_est' and comp == 'Y') else comp}"
                if key == "naive2_est" and comp == "pi":
                    continue                               # the monitor has no policy player
                s = summ[col]
                lo, hi = spread([r[col] for r in runs])
                ax.plot([lo, hi], [y + dy] * 2, color=color, lw=0.6, alpha=0.35)
                ax.errorbar(s["mean"], y + dy, xerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]],
                            fmt=marker, color=color, ms=4, elinewidth=1.3, capsize=0)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(ref["dR"]["value"], color=CG, lw=0.7, ls="--")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
        ax.set_title(title)
        ax.set_xlabel("attributed change in Brier score")
    handles = [plt.Line2D([], [], marker=m, color=c, ls="", ms=4) for _, c, m, _ in comps]
    fig.legend(handles, [c[3] for c in comps], fontsize=6.3, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_attribution.pdf")


def figure2(E7):
    B = E7["B"]
    summ = B["summary"]
    eps = [r["eps"] for r in summ]
    xe = np.array([1e-5 if e == 0 else e for e in eps])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    ax = axes[0]
    for key, label, color, marker, off in (("union3_raw_pi", "union graph IS, unclipped", CG, "o", 0.92),
                                          ("union3_clip_pi", "weights clipped at 100", CY, "s", 1.0),
                                          ("E3_rollout_pi", "proposed, randomised arm", CP, "^", 1.08)):
        m = np.array([r[key]["mean"] for r in summ])
        lo = np.array([r[key]["ci95"][0] for r in summ])
        hi = np.array([r[key]["ci95"][1] for r in summ])
        ax.errorbar(xe * off, m, yerr=[m - lo, hi - m], marker=marker, ms=3, color=color, capsize=2, lw=0.9, label=label)
    ax.plot(xe, [r["truth_policy_contrast_pi"]["value"] for r in summ], "k--", lw=0.8, label="target")
    ax.set_xscale("log")
    ax.set_xticks(xe)
    ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "5e-2"])
    ax.set_xlabel("$\\varepsilon$: pre-deployment treatment prob.\nin the alert-only region")
    ax.set_ylabel("policy contribution")
    ax.set_title("across the positivity continuum")
    ax = axes[1]
    m = np.array([r["ess"]["mean"] for r in summ])
    lo = np.array([r["ess"]["ci95"][0] for r in summ])
    hi = np.array([r["ess"]["ci95"][1] for r in summ])
    ax.errorbar(xe, m, yerr=[m - lo, hi - m], marker="o", ms=3, color=CG, capsize=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xe)
    ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "5e-2"])
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel(f"effective sample size (of {E7['parameters']['n']:,})")
    ax.set_title("ESS highest where the estimate is wrong")
    ax = axes[2]
    ns = B["n_scaling"]["summary"]
    n = np.array([r["n"] for r in ns])
    m = np.array([r["union3_raw_pi"]["mean"] for r in ns])
    alo = np.array([r["analyst_ci95_median"][0] for r in ns])
    ahi = np.array([r["analyst_ci95_median"][1] for r in ns])
    ax.errorbar(n, m, yerr=[m - alo, ahi - m], marker="o", ms=3, color=CG, capsize=2,
                label="union graph IS at $\\varepsilon=0$,\nwith an analyst's 95% bootstrap CI")
    ax.axhline(ns[0]["truth_shapley_pi"], color="k", ls="--", lw=0.8, label="target")
    ax.axhline(0, color=CG, lw=0.4)
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{v // 1000:g}k" for v in n], fontsize=6)
    ax.minorticks_off()
    ax.set_xlabel("$n$ (pre-deployment sample)")
    ax.set_title("more data, same wrong answer")
    ax.legend(fontsize=5.4, frameon=False, loc="center right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig2_positivity.pdf")


def figure4(E7):
    F = E7["F"]["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.35))
    for setting, alpha in (("endogenous only", 1.0), ("mixed", 0.5)):
        rows = [r for r in F if r["setting"] == setting]
        tg = np.array([r["triage"] for r in rows])
        axes[0].plot(tg, [r["references"]["shapley_X"]["value"] for r in rows], marker="o", ms=3, color=CX,
                     alpha=alpha, label=f"Shapley target, {setting}")
        for key, color, ls, marker, lab in (("ua_X", CG, "--", "s", "monitor"), ("e3_X", CP, ":", "^", "proposed $X$ term")):
            m = np.array([r[key]["mean"] for r in rows])
            lo = np.array([r[key]["ci95"][0] for r in rows])
            hi = np.array([r[key]["ci95"][1] for r in rows])
            axes[0].plot(tg, m, marker=marker, ms=3, ls=ls, color=color, alpha=alpha, label=f"{lab}, {setting}")
            axes[0].fill_between(tg, lo, hi, color=color, alpha=0.15 * alpha, lw=0)
    rows = [r for r in F if r["setting"] == "endogenous only"]
    tg = np.array([r["triage"] for r in rows])
    axes[1].plot(tg, [r["retrospective"]["value"] for r in rows], marker="o", ms=3, color="#AA3377",
                 label="retrospective: undo deployment")
    axes[1].plot(tg, [r["prospective"]["value"] for r in rows], marker="o", ms=3, color=CP,
                 label="prospective: switch alert off now")
    m = np.array([r["e3_pi"]["mean"] for r in rows])
    lo = np.array([r["e3_pi"]["ci95"][0] for r in rows])
    hi = np.array([r["e3_pi"]["ci95"][1] for r in rows])
    axes[1].errorbar(tg, m, yerr=[m - lo, hi - m], marker="s", ms=3, ls="--", color=CP, capsize=2,
                     label="randomised arm estimate")
    axes[0].set_xlabel("triage: covariate shift caused by deployment")
    axes[0].set_ylabel("attributed to patient mix $P(X)$")
    axes[0].set_title("deployment-caused mix shift is booked to $P(X)$")
    axes[1].set_xlabel("triage: covariate shift caused by deployment")
    axes[1].set_ylabel("policy contribution")
    axes[1].set_title("two policy estimands; randomisation recovers the decision one")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.8, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.27, -0.02))
    axes[1].legend(fontsize=5.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "fig4_triage.pdf")


def figure5(E7):
    D = E7["D"]["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    for th, c in zip((0.5, 1.0, 1.5, 2.5), ("#CCBB44", "#66CCEE", CX, "#AA3377")):
        rows = sorted((r for r in D if r["theta"] == th), key=lambda r: r["p_alert"])
        p = np.array([r["p_alert"] for r in rows])
        for ax, key, scale, ls, marker, lab in ((axes[0], "auroc_e1", 1, "-", "o", f"treatment effect $\\theta$ = {th}"),
                                                (axes[1], "averted_observed", 100, "-", "o", None),
                                                (axes[2], "nb_std", 1, "--", "o", None),
                                                (axes[2], "nb_cf", 1, "-", "s", None)):
            m = scale * np.array([r[key]["mean"] for r in rows])
            lo = scale * np.array([r[key]["ci95"][0] for r in rows])
            hi = scale * np.array([r[key]["ci95"][1] for r in rows])
            ax.plot(p, m, marker=marker, ms=2.8, color=c, ls=ls, label=lab)
            ax.fill_between(p, lo, hi, color=c, alpha=0.2, lw=0)
        axes[0].plot(p, [r["auroc_e0"]["mean"] for r in rows], ls=":", lw=0.8, color=c)
    cf = np.mean([r["auroc_cf"]["mean"] for r in D])
    axes[0].axhline(cf, color="k", lw=0.6, ls="-.")
    axes[0].text(0.12, cf + 0.004, "vs untreated potential outcome", fontsize=5.5)
    axes[0].set_title("observed AUROC (dotted: pre-deployment)")
    axes[1].set_title("events averted (pp)")
    axes[2].set_title("net benefit at $\\tau$: standard DCA (dashed)\nvs counterfactual (solid)")
    axes[2].axhline(0, color="k", lw=0.5)
    for ax in axes:
        ax.set_xlabel("clinician adherence to alerts")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "fig5_utility.pdf")


def main():
    E7 = json.loads((ROOT / "figures/merged_expE7.json").read_text())
    figure1(E7)
    figure2(E7)
    import make_figure3_E3
    make_figure3_E3.main()
    figure4(E7)
    figure5(E7)
    import make_figure_E1
    make_figure_E1.main()
    if (ROOT / "figures/fig7_mimic.pdf").exists():
        (ROOT / "aaai/figures/fig7_mimic.pdf").write_bytes((ROOT / "figures/fig7_mimic.pdf").read_bytes())
        print("copied fig7_mimic.pdf")
    else:
        print("fig7_mimic.pdf not present: run sim2/exp_E6_mimic.py next to credentialed MIMIC-IV data")
    print("ok")


if __name__ == "__main__":
    main()
