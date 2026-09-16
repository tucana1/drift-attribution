import json, csv, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "axes.titlesize": 8})
CX, CP, CY, CG = "#4477AA", "#228833", "#EE6677", "#888888"   # X, policy, Y|X(,A), grey
F = "../figures/"

# ------------------------------------------------------------ Fig 1: attribution under S3 and S4, two references
A = json.load(open(F + "merged_expA.json")); M = json.load(open(F + "merged_expM.json"))
def ms(sc, est, key): v = np.array([r[est][key] for r in A[sc]["runs"]]); return v.mean(), v.std()
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1, 1]})
for ax, sc, ttl in zip(axes, ["S3", "S4"], ["S3: performative success only", "S4: covariate + outcome shift + deployment"]):
    rows = [("Shapley game, oracle coalition values", [A[sc]["truth3"]["X"], A[sc]["truth3"]["pi"], A[sc]["truth3"]["Y"]], None),
            ("policy-contrast estimand (target)", [A[sc]["truthE3"]["X"], A[sc]["truthE3"]["pi"], A[sc]["truthE3"]["Y"]], None),
            ("union graph IS, unclipped", [ms(sc, "union3_raw", "X"), ms(sc, "union3_raw", "pi"), ms(sc, "union3_raw", "Y")], "sd"),
            ("union graph IS, weights clipped", [ms(sc, "union3_clip", "X"), ms(sc, "union3_clip", "pi"), ms(sc, "union3_clip", "Y")], "sd"),
            ("no action log: {P(X), P(Y|X)}", [ms(sc, "naive2_est", "X"), (0, 0), ms(sc, "naive2_est", "Ygx")], "sd"),
            ("proposed, randomised arm", [ms(sc, "E3_rollout", "X"), ms(sc, "E3_rollout", "pi"), ms(sc, "E3_rollout", "Y")], "sd")]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, vals, kind) in zip(y, rows):
        left = 0.0
        for v, c in zip(vals, [CX, CP, CY]):
            m, s = (v if kind else (v, 0.0))
            ax.barh(yi, m, left=left, color=c, height=0.62, edgecolor="white", lw=0.4); left += m
        if kind:
            tot = sum(v[0] for v in vals); sd = np.sqrt(sum(v[1] ** 2 for v in vals)); ax.errorbar(left, yi, xerr=ms(sc, "union3_raw", "pi")[1] if "unclipped" in lab else 0, color="k", lw=0.6, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=6.5); ax.axvline(0, color="k", lw=0.5)
    ax.axvline(A[sc]["truthE3"]["dR"], color=CG, lw=0.6, ls="--"); ax.set_title(ttl); ax.set_xlabel("attributed change in Brier score")
h = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in [CX, CP, CY]]
fig.legend(h, ["patient mix P(X)", "alerting policy $\\pi$", "outcome mechanism P(Y|X,A), or marginal P(Y|X) when A is not logged"], fontsize=6.3, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(F + "fig1_attribution.pdf", bbox_inches="tight"); plt.close()

# ------------------------------------------------------------ Fig 2: eps continuum and n scaling
B = json.load(open(F + "merged_expB.json")); ns = list(csv.DictReader(open(F + "nonidentification_n_scaling.csv")))
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
eps = [r["eps"] for r in B]; xe = [1e-5 if e == 0 else e for e in eps]
ax = axes[0]; ax.errorbar(xe, [r["raw_mean"] for r in B], [r["raw_sd"] for r in B], marker="o", ms=3, color=CG, capsize=2, label="union graph IS, unclipped")
ax.errorbar(xe, [r["clip_mean"] for r in B], [r["clip_sd"] for r in B], marker="s", ms=3, color=CY, capsize=2, label="weights clipped at 100")
ax.errorbar(xe, [r["e3_mean"] for r in B], [r["e3_sd"] for r in B], marker="^", ms=3, color=CP, capsize=2, label="proposed, randomised arm")
ax.plot(xe, [r["truth_pi"] for r in B], "k--", lw=0.8, label="truth"); ax.set_xscale("log"); ax.set_xticks(xe); ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "5e-2"])
ax.set_xlabel("$\\varepsilon$: pre-deployment treatment prob. in alert-only region"); ax.set_ylabel("policy contribution"); ax.legend(fontsize=5.3, frameon=False, loc="lower right"); ax.set_title("across the positivity continuum")
ax = axes[1]; ax.plot(xe, [r["ess"] for r in B], marker="o", ms=3, color=CG); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(xe); ax.set_xticklabels(["0", "1e-4", "1e-3", "1e-2", "5e-2"])
ax.set_xlabel("$\\varepsilon$"); ax.set_ylabel("effective sample size (of 20,000)"); ax.set_title("ESS highest where the estimate is wrong")
ax = axes[2]; n = [int(r["n"]) for r in ns]; ax.errorbar(n, [float(r["pi"]) for r in ns], [float(r["sd"]) for r in ns], marker="o", ms=3, color=CG, capsize=2, label="union graph IS at $\\varepsilon = 0$")
ax.axhline(float(ns[0]["truth_pi"]), color="k", ls="--", lw=0.8, label="truth"); ax.axhline(0, color=CG, lw=0.4); ax.set_xscale("log"); ax.set_xlabel("$n$ (pre-deployment sample)"); ax.set_title("more data, same wrong answer")
ax.legend(fontsize=5.5, frameon=False); plt.tight_layout(); plt.savefig(F + "fig2_positivity.pdf"); plt.close()

# ------------------------------------------------------------ Fig 3: retrain and redeploy, 8 rounds, 5 arms
C = json.load(open(F + "merged_expC.json")); base = np.mean([r["base"] for r in C])
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
arms = [("keep", "keep $f$", CP, "-"), ("retrain", "naive retrain", CY, "-"), ("unalerted", "retrain on $\\hat r \\leq \\tau$ (no action log)", "#CCBB44", "--"),
        ("untreated", "retrain on $A = 0$ (action log)", "#66CCEE", "--"), ("cond", "retrain on $(X, A)$, predict at $A{=}0$", CX, ":")]
for arm, lab, c, ls in arms:
    ev = np.array([[t[0] for t in r[arm]] for r in C]); al = np.array([[t[1] for t in r[arm]] for r in C]); au = np.array([[t[2] for t in r[arm]] for r in C]); t = np.arange(1, ev.shape[1] + 1)
    for ax, yv, name in zip(axes, [100 * (base - ev), 100 * al, au], ["events averted vs standard care (pp)", "alert rate (%)", "observed AUROC, deployed population"]):
        ax.errorbar(t, yv.mean(0), yv.std(0), marker="o", ms=2.5, lw=1, ls=ls, capsize=1.5, color=c, label=lab); ax.set_title(name); ax.set_xticks(t); ax.set_xlabel("deployment round")
axes[0].legend(fontsize=5.5, frameon=False, loc="center right"); plt.tight_layout(); plt.savefig(F + "fig3_retrain.pdf"); plt.close()

# ------------------------------------------------------------ Fig 4: triage axis
Fj = json.load(open(F + "merged_expF.json"))
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for setting, alpha in [("endogenous only", 1.0), ("mixed", 0.45)]:
    rows = [r for r in Fj if r["setting"] == setting]; tg = [r["triage"] for r in rows]
    axes[0].plot(tg, [r["truth_X"] for r in rows], marker="o", ms=3, color=CX, alpha=alpha, label=f"truth, {setting}")
    axes[0].plot(tg, [r["ua_X"] for r in rows], marker="s", ms=3, ls="--", color=CG, alpha=alpha, label=f"monitor and proposed X term, {setting}")
    axes[0].plot(tg, [r["e3_X"] for r in rows], marker="^", ms=3, ls=":", color=CP, alpha=alpha)
rows = [r for r in Fj if r["setting"] == "endogenous only"]; tg = [r["triage"] for r in rows]
axes[1].plot(tg, [r["retrospective"] for r in rows], marker="o", ms=3, color="#AA3377", label="retrospective: undo deployment")
axes[1].plot(tg, [r["prospective"] for r in rows], marker="o", ms=3, color=CP, label="prospective: switch alert off now")
axes[1].errorbar(tg, [r["e3_pi"] for r in rows], [r["e3_pi_sd"] for r in rows], marker="s", ms=3, ls="--", color=CP, capsize=2, label="randomised arm estimate")
axes[0].set_xlabel("triage: covariate shift caused by deployment"); axes[0].set_ylabel("attributed to patient mix P(X)"); axes[0].set_title("deployment-caused mix shift is booked to P(X)")
axes[1].set_xlabel("triage: covariate shift caused by deployment"); axes[1].set_ylabel("policy contribution"); axes[1].set_title("two policy estimands; randomisation recovers the decision one")
axes[0].legend(fontsize=5.3, frameon=False); axes[1].legend(fontsize=5.5, frameon=False); plt.tight_layout(); plt.savefig(F + "fig4_triage.pdf"); plt.close()

# ------------------------------------------------------------ Fig 5: AUROC vs utility
D = json.load(open(F + "merged_expD.json"))
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
for th, c in zip([0.5, 1.0, 1.5, 2.5], ["#CCBB44", "#66CCEE", CX, "#AA3377"]):
    rows = sorted([r for r in D if r["theta"] == th], key=lambda r: r["p_alert"]); p = [r["p_alert"] for r in rows]
    axes[0].plot(p, [r["auroc_e1"] for r in rows], marker="o", ms=3, color=c, label=f"treatment effect $\\theta$ = {th}")
    axes[0].plot(p, [r["auroc_e0"] for r in rows], ls=":", lw=0.8, color=c)
    axes[1].plot(p, [100 * r["averted"] for r in rows], marker="o", ms=3, color=c)
    axes[2].plot(p, [r["nb_std"] for r in rows], marker="o", ms=3, color=c, ls="--"); axes[2].plot(p, [r["nb_cf"] for r in rows], marker="s", ms=3, color=c)
axes[0].axhline(D[0]["auroc_cf"], color="k", lw=0.6, ls="-."); axes[0].text(0.12, D[0]["auroc_cf"] + 0.004, "vs untreated potential outcome", fontsize=5.5)
axes[0].set_title("observed AUROC (dotted: pre-deployment)"); axes[1].set_title("events averted (pp)"); axes[2].set_title("net benefit at $\\tau$: standard DCA (dashed)\nvs counterfactual (solid)"); axes[2].axhline(0, color="k", lw=0.5)
for ax in axes: ax.set_xlabel("clinician adherence to alerts")
axes[0].legend(fontsize=5.3, frameon=False); plt.tight_layout(); plt.savefig(F + "fig5_utility.pdf"); plt.close()
print("ok")
