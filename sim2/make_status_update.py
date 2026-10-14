"""Status-update figures for the team channel: workflow, missing input, main results.

    python3 sim2/make_status_update.py

Writes docs/status_update/01_workflow.png ... 08_real_covariates.png and
docs/status_update/README.md (one short caption per image, ready to paste).
Every number in the charts and captions is read from the result files in
figures/; nothing is recomputed, so this takes a few seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Wedge
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "docs" / "status_update"
DATE = "13 October 2026"

# Reference palette, light surface. Categorical slots 1-3 pass the colour-vision
# and normal-vision checks for every pair; aqua sits below 3:1 on the surface, so
# every aqua mark carries a visible value label. Status colours appear only with
# an icon and a word.
SURFACE, WHITE = "#fcfcfb", "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"                      # diverging pole opposite blue
GRAY = "#c3c2b7"                     # de-emphasised marks
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
SYMBOLS = "DejaVu Sans"              # has the check and cross glyphs

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 11,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 10.5,
})

FIGSIZE = (10, 5.625)                # 2000 x 1125 px at 200 dpi
CAPTIONS: list[tuple[str, str]] = []


def load(name):
    return json.loads((FIG / name).read_text())


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote docs/status_update/{name}")


def header(fig, title, subtitle, source):
    fig.text(0.035, 0.955, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.885, subtitle, fontsize=11.5, color=INK2, va="top", linespacing=1.4)
    fig.text(0.035, 0.025, source, fontsize=9, color=MUTED, va="bottom")


def fmt(v, digits=3, sign=True):
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return s.replace("-", "\u2212")


# ----------------------------------------------------------------------------- bars
def rounded_bar(ax, center, width, value, color, horizontal=False, base=0.0, radius_px=4.0):
    """Bar from the baseline, square at the baseline and rounded (radius_px at 1x) at the data end."""
    if value == 0:
        return
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px_x, px_y = bbox.width / abs(x1 - x0), bbox.height / abs(y1 - y0)
    # along the bar (data end) and across it
    along_px, across_px = (px_x, px_y) if horizontal else (px_y, px_x)
    r_along = min(radius_px / along_px, abs(value))
    r_across = min(radius_px / across_px, width / 2)
    a, b = center - width / 2, center + width / 2
    end, s = base + value, np.sign(value)
    pts = [(a, base), (a, end - s * r_along), (a, end), (a + r_across, end),
           (b - r_across, end), (b, end), (b, end - s * r_along), (b, base), (a, base)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    if horizontal:
        pts = [(p[1], p[0]) for p in pts]
    ax.add_patch(PathPatch(MPath(pts, codes), facecolor=color, edgecolor="none", zorder=3))


def baseline(ax, horizontal=False):
    (ax.axvline if horizontal else ax.axhline)(0, color=AXIS, lw=1.0, zorder=2)


# ----------------------------------------------------------------------------- 1 workflow
def badge(ax, x, y, status, r=0.95):
    if status == "done":
        ax.add_patch(Circle((x, y), r, color=GOOD, zorder=5))
        ax.text(x, y - 0.05, "✓", ha="center", va="center", color=WHITE, fontsize=11 * r / 0.95,
                fontweight="bold", fontfamily=SYMBOLS, zorder=6)
    elif status == "partial":
        ax.add_patch(Circle((x, y), r, facecolor=WHITE, edgecolor=WARN, lw=1.8, zorder=5))
        ax.add_patch(Wedge((x, y), r, 90, 270, color=WARN, zorder=6))
    else:
        ax.add_patch(Circle((x, y), r, color=CRIT, zorder=5))
        ax.text(x, y - 0.03, "✕", ha="center", va="center", color=WHITE, fontsize=10 * r / 0.95,
                fontweight="bold", fontfamily=SYMBOLS, zorder=6)


def node(ax, x, y, w, h, title, detail=(), status="done", tag=None, alert=False, title_size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.7",
                                facecolor=to_rgba(CRIT, 0.07) if alert else WHITE,
                                edgecolor=CRIT if alert else AXIS, lw=2.0 if alert else 1.0, zorder=2))
    ty = y + h - 1.55
    badge(ax, x + 1.55, ty, status)
    title_artist = ax.text(x + 3.0, ty, title, fontsize=title_size, fontweight="bold", color=INK, va="center",
                           zorder=4)
    if tag:
        ax.text(x + w - 0.8, ty, tag, fontsize=9.5, color=MUTED, va="center", ha="right", zorder=4)
    for i, line in enumerate(detail):
        ax.text(x + 3.0, ty - 2.05 - 1.75 * i, line, fontsize=10, color=INK2, va="center", zorder=4)
    return title_artist


def arrow(ax, p0, p1, color=MUTED, dashed=False, lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
                                 linestyle=(0, (4, 3)) if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))


def figure_workflow():
    stand_in = load("e6_open/merged_expE6_sepsis2019_AtoB.json")["cohort"]["admissions"]
    todo = (ROOT / "aaai/drift_attribution_aaai.tex").read_text().count("\\todo{")
    fig = plt.figure(figsize=(13.333, 7.85))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 58.875)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(2, 57.0, "Drift attribution: where the project stands", fontsize=21, fontweight="bold",
            color=INK, va="top")
    ax.text(2, 53.5, "The question: when clinicians act on a hospital AI alert, the model's measured accuracy changes. "
                     "Does monitoring blame the right cause?\n"
                     f"Status on {DATE}: all the simulation work is done and reproducible. "
                     "Still missing: MIMIC-IV, the real hospital data.",
            fontsize=12.5, color=INK2, va="top", linespacing=1.45)
    for x, status, label in ((70.5, "done", "Done"), (78.2, "partial", "Partly done"),
                             (88.2, "missing", "Missing or blocked")):
        badge(ax, x, 56.6, status, r=0.8)
        ax.text(x + 1.5, 56.6, label, fontsize=10.5, color=INK2, va="center")
    for x, label in ((2, "DATA"), (26, "EXPERIMENTS"), (57, "RESULTS"), (80, "PAPER")):
        ax.text(x, 47.6, label, fontsize=10, fontweight="bold", color=MUTED, va="center")

    # inputs
    node(ax, 2, 24.9, 19, 7.4, "Simulated hospital",
         ("the true answer is known, so", "each method can be checked"))
    node(ax, 2, 12.0, 19, 7.4, "Public ICU data",
         ("PhysioNet 2019 sepsis challenge,", f"2 hospitals, {stand_in:,} ICU stays"))
    mimic = node(ax, 2, 0.9, 19, 9.0, "MIMIC-IV data", ("real hospital records; needs",
                                                         "approved access, must run on",
                                                         "our own computers; script ready"),
                 status="missing", alert=True)
    fig.canvas.draw()
    right = mimic.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).x1
    pill_x, pill_w, pill_y = right + 0.8, 5.3, 9.9 - 1.55
    assert pill_x + pill_w <= 2 + 19 - 0.5, "MISSING pill does not fit beside the title"
    ax.add_patch(FancyBboxPatch((pill_x, pill_y - 0.8), pill_w, 1.6, boxstyle="round,pad=0,rounding_size=0.45",
                                facecolor=CRIT, edgecolor="none", zorder=4))
    ax.text(pill_x + pill_w / 2, pill_y, "MISSING", fontsize=9, fontweight="bold", color=WHITE, ha="center",
            va="center", zorder=5)

    # experiments
    boxes = (("Which cause gets the blame?", "A, B · E7", "standard monitoring vs our method", "done"),
             ("Retraining after launch", "C · E3", "7 ways to retrain the model, compared", "done"),
             ("Keep or retrain?", "E1", "does knowing the cause help decide?", "done"),
             ("Triage and staged rollout", "F · E5", "alert changes who is seen; 48 sites", "done"),
             ("Stress tests", "E4", "more variables; outcome rates 7% to 40%", "done"),
             ("Real hospital data", "E6", "public data done; MIMIC-IV run blocked", "partial"))
    centers = []
    for i, (title, tag, detail, status) in enumerate(boxes):
        top = 46.0 - 7.2 * i
        node(ax, 26, top - 6.0, 27, 6.0, title, (detail,), status=status, tag=tag, title_size=11.5)
        centers.append(top - 3.0)
    # the simulator feeds experiments 1-5
    ax.plot([25.0, 24.6, 24.6, 25.0], [centers[0] + 2.6, centers[0] + 2.6, centers[4] - 2.6, centers[4] - 2.6],
            color=MUTED, lw=1.2, solid_capstyle="butt")
    arrow(ax, (21.0, 28.6), (24.5, 28.6))
    # real-data inputs feed experiment 6
    arrow(ax, (21.0, 16.1), (25.9, 8.4), rad=0.0)
    arrow(ax, (21.0, 4.4), (25.9, 5.2), color=CRIT, dashed=True, lw=1.6)
    ax.text(23.4, 3.2, "blocked", fontsize=9.5, color=INK2, style="italic", ha="center")

    # collector bus into the result files
    for c in centers:
        ax.plot([53.0, 55.0], [c, c], color=MUTED, lw=1.2)
    ax.plot([55.0, 55.0], [centers[-1], centers[0]], color=MUTED, lw=1.2)
    arrow(ax, (55.0, 40.5), (56.9, 40.5))

    # outputs
    node(ax, 57, 37.0, 20, 7.4, "Saved results", ("one script per result file;", "reruns give the same numbers"),
         tag="R1, R2")
    node(ax, 57, 23.6, 20, 7.4, "Figures and tables", ("rebuilt with one command,", "identical every time"),
         tag="R4")
    node(ax, 57, 10.2, 20, 7.4, "Automatic checks", ("key numbers in the draft are", "checked against the results"))
    arrow(ax, (67, 37.0), (67, 31.1))
    arrow(ax, (67, 23.6), (67, 17.7))

    # paper
    x0, y0, w, h = 80, 9.2, 19, 35.8
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.7", facecolor=WHITE,
                                edgecolor=AXIS, lw=1.0, zorder=2))
    ax.text(x0 + 1.2, y0 + h - 1.7, "Draft paper", fontsize=12, fontweight="bold", color=INK, va="center")
    ax.text(x0 + 1.2, y0 + h - 3.7, "AAAI-27 workshop, due about 20 Nov", fontsize=10, color=INK2, va="center")
    items = (("done", ["Results section written: 6 of", "7 findings, all with error bars"]),
             ("done", ["Existing references checked"]),
             ("missing", ["Finding 7 (real hospital data)", "waits on the MIMIC-IV run"]),
             ("partial", ["Still to write: abstract, intro,", "theory and proofs, related",
                          "work, responsible use"]),
             ("partial", ["A few references still to add"]))
    y = y0 + h - 7.2
    for status, lines in items:
        badge(ax, x0 + 1.9, y, status, r=0.8)
        for j, line in enumerate(lines):
            ax.text(x0 + 3.4, y - 1.75 * j, line, fontsize=10.2, color=INK, va="center")
        y -= 1.75 * len(lines) + 1.6
    ax.text(x0 + 1.2, y0 + 1.4, f"{todo} TODO notes left in the draft", fontsize=9.5, color=MUTED, va="center")
    arrow(ax, (77.0, 27.3), (79.9, 27.3))
    arrow(ax, (77.0, 13.9), (79.9, 13.9))
    ax.text(97.5, 2.2, "The main simulations use one hospital setup (the stress tests vary it); with the public data, "
                       "only the alert and its effect are simulated.", fontsize=9, color=MUTED, ha="right")
    save(fig, "01_workflow.png")
    CAPTIONS.append((
        "01_workflow.png",
        "All the simulation work is done; the one missing piece (red) is the MIMIC-IV run, which needs someone "
        "with approved access to the data to run our script on their own computer."))


# ----------------------------------------------------------------------------- 2 and 8 attribution bars
COMPONENTS = (("The alert being acted on", BLUE), ("Underlying risk changed", ORANGE), ("Different patients", AQUA))
METHODS = ("What really\nhappened", "Standard\nmonitoring", "Our method")


def attribution_chart(name, panels, title, subtitle, source, xlabel, digits, label_min, tick_step, note=None):
    """Horizontal grouped bars: one group per method, one bar per component, value at the tip.

    panels: (panel title, rows, errors); rows[method][component] is a value or None (no such
    term), errors[method][component] is a 95% interval or None.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=FIGSIZE, sharey=True)
    fig.subplots_adjust(left=0.2, right=0.975, top=0.69, bottom=0.2 if note else 0.165, wspace=0.1)
    ends = [0.0]
    for _, rows, errs in panels:
        for r, e in zip(rows, errs):
            ends += [v for v in r if v is not None] + [x for ci in e if ci for x in ci]
    lo, hi = min(ends), max(ends)
    span = hi - lo
    # room for a value label beyond the bar end on each side that has labelled bars
    xlim = (lo - (0.24 if lo < -label_min / 2 else 0.03) * span, hi + (0.2 if hi > label_min / 2 else 0.12) * span)
    # notes at zero go on the side of the axis with more room
    side = 1 if hi >= -lo else -1
    at_zero = dict(va="center", ha="left" if side > 0 else "right", color=MUTED)
    pitch, thick = 0.27, 0.2
    for ax, (ptitle, rows, errs) in zip(axes, panels):
        ax.set_xlim(*xlim)
        ax.set_ylim(2.62, -0.62)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        baseline(ax, horizontal=True)
        for g, (row, err) in enumerate(zip(rows, errs)):
            for k, (_, color) in enumerate(COMPONENTS):
                y = g + (k - 1) * pitch
                v, ci = row[k], err[k]
                if v is None:
                    ax.text(0.012 * span * side, y, "doesn't consider the alert", fontsize=9.5, style="italic",
                            **at_zero)
                    continue
                rounded_bar(ax, y, thick, v, color, horizontal=True)
                if ci is not None:
                    ax.plot(ci, [y, y], color=INK2, lw=1.0, zorder=4)
                if abs(v) >= label_min:
                    tip = max([v] + list(ci or [])) if v > 0 else min([v] + list(ci or []))
                    ax.text(tip + (0.015 if v > 0 else -0.015) * span, y, fmt(v, digits), va="center",
                            ha="left" if v > 0 else "right", fontsize=10, color=INK, zorder=5)
                elif g == 0 and v == 0:
                    ax.text(0.012 * span * side, y, "0", fontsize=9.5, **at_zero)
        ax.set_title(ptitle, fontsize=12, color=INK, loc="left", pad=8)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_step))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+.{max(digits - 1, 2)}f}".replace("-", "\u2212")))
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(METHODS, fontsize=10.5, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in COMPONENTS]
    fig.legend(handles, [l for l, _ in COMPONENTS], loc="upper right", bbox_to_anchor=(0.975, 0.835),
               ncol=3, handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    header(fig, title, subtitle, source)
    if note:
        fig.text(0.035, 0.062, note, fontsize=9.5, color=INK2, va="bottom")
    save(fig, name)


def figure_attribution():
    E7 = load("merged_expE7.json")
    A = E7["A"]
    reps = len([r for r in A["runs"] if r["scenario"] == "S3"])
    panels, out = [], {}
    for sc, ptitle in (("S3", "Only change: clinicians act on the alert"),
                       ("S4", "Alert, plus different patients and changed risk")):
        ref, s = A["references"][sc], A["summary"][sc]
        truth = [ref["pc_pi"]["value"], ref["pc_Y"]["value"], ref["pc_X"]["value"]]
        mon = [None, s["naive2_est_Ygx"]["mean"], s["naive2_est_X"]["mean"]]
        pro = [s["E3_rollout_pi"]["mean"], s["E3_rollout_Y"]["mean"], s["E3_rollout_X"]["mean"]]
        err = [[None] * 3, [None, s["naive2_est_Ygx"]["ci95"], s["naive2_est_X"]["ci95"]],
               [s["E3_rollout_pi"]["ci95"], s["E3_rollout_Y"]["ci95"], s["E3_rollout_X"]["ci95"]]]
        panels.append((ptitle, [truth, mon, pro], err))
        out[sc] = (truth, mon, pro)
    t3, m3, p3 = out["S3"]
    # S4: the patient-mix shift moves more patients into the alert region; the policy contrast that the
    # truth row and our method use gives that interaction to the alert (Shapley symmetry would halve it)
    overlap = A["references"]["S4"]["I_Xpi"]["value"]
    assert t3[0] > 0 and m3[1] > 0.9 * t3[0] and p3[0] > 0.9 * t3[0], "the captions describe this pattern"
    attribution_chart(
        "02_attribution.png", panels,
        "Standard monitoring blames the wrong cause",
        "In a simulated hospital we know what really happened. The bars show which cause each method blames for the"
        "\nchange in the model's prediction error. Our method gets it right; standard monitoring does not.",
        f"Source: figures/merged_expE7.json · {reps} simulated runs per panel, {E7['parameters']['n']:,} patients "
        "per period · thin lines: 95% intervals",
        "change in prediction error (Brier score)", 3, 0.002, 0.01,
        note=f"Right: the new patients trigger more alerts, so {fmt(overlap, 3, False)} of the change comes from "
             "both together; it is counted as the alert's, as our method does.")
    CAPTIONS.append((
        "02_attribution.png",
        "In a simulated hospital where the only change is clinicians acting on the alert, standard monitoring "
        "blames a change in patients' risk, while our method correctly blames the alert."))


def figure_real_covariates():
    panels, rows = [], {}
    for tag, ptitle in (("AtoB", "Hospital A"), ("BtoA", "Hospital B")):
        d = load(f"e6_open/merged_expE6_sepsis2019_{tag}.json")
        t, s = d["A"]["truth"]["deploy_only"], d["A"]["summary"]["deploy_only"]
        truth = [t["pi"], 0.0, 0.0]        # one hospital split at random: no real change in X or Y | X
        mon = [None, s["monitor_Ygx"]["mean"], s["monitor_X"]["mean"]]
        pro = [s["proposed_pi"]["mean"], s["proposed_Y"]["mean"], s["proposed_X"]["mean"]]
        err = [[None] * 3, [None, s["monitor_Ygx"]["ci95"], s["monitor_X"]["ci95"]],
               [s["proposed_pi"]["ci95"], s["proposed_Y"]["ci95"], s["proposed_X"]["ci95"]]]
        panels.append((ptitle, [truth, mon, pro], err))
        rows[tag] = (truth, mon, pro, d["cohort"])
    ca = rows["AtoB"][3]
    attribution_chart(
        "08_real_covariates.png", panels,
        "The same mistake on real patient data (public stand-in for MIMIC-IV)",
        "Real ICU patients and outcomes from two hospitals; only the alert and its effect are simulated.\nHere the "
        "alert lowers prediction error, and standard monitoring credits that to lower underlying risk.",
        f"Source: figures/e6_open/ · PhysioNet 2019 challenge, {ca['admissions']:,} ICU stays, sepsis in "
        f"{100 * ca['event_rate_early']:.1f}% (A) and {100 * ca['event_rate_late']:.1f}% (B) · thin lines: 95% "
        "intervals",
        "change in prediction error (lower = better)", 4, 0.0008, 0.001)
    for truth, mon, pro, _ in rows.values():
        assert truth[0] < 0 and mon[1] < 0 and pro[0] < 0, "the captions describe this pattern"
    CAPTIONS.append((
        "08_real_covariates.png",
        "On real ICU patients from two hospitals (a public stand-in until the MIMIC-IV run), standard monitoring "
        "again mistakes the alert's effect for a change in risk, while our method gets it right."))


# ----------------------------------------------------------------------------- 3 more data
def figure_more_data():
    B = load("merged_expE7.json")["B"]
    ns = B["n_scaling"]["summary"]
    target = ns[0]["truth_shapley_pi"]
    n = np.array([r["n"] for r in ns])
    m = np.array([r["union3_raw_pi"]["mean"] for r in ns])
    lo = np.array([r["analyst_ci95_median"][0] for r in ns])
    hi = np.array([r["analyst_ci95_median"][1] for r in ns])
    covered = max(r["analyst_ci_covers_truth"] for r in ns)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.085, right=0.975, top=0.72, bottom=0.165, wspace=0.28)
    ticks = [f"{v // 1000:,}k" for v in n]
    ax = axes[0]
    ax.set_xscale("log")
    ax.set_xlim(14_000, 560_000)
    ax.set_ylim(-0.016, 0.036)
    baseline(ax)
    ax.axhline(target, color=INK, lw=1.4, zorder=2)
    ax.text(16_000, target + 0.0012, f"true effect of the alert  {fmt(target)}", fontsize=10.5, color=INK, va="bottom")
    ax.plot(n, m, "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=4)
    ax.text(16_000, m[0] - 0.0022, f"what the usual correction reports  {fmt(m.mean())}", fontsize=10.5,
            color=INK, va="top")
    ax.set_title("Its answer vs the true effect", fontsize=12, color=INK, loc="left", pad=8)
    ax.set_ylabel("effect of the alert on prediction error")
    ax2 = axes[1]
    ax2.set_xscale("log")
    ax2.set_xlim(14_000, 700_000)
    ax2.set_ylim(-0.0108, -0.0070)
    for xi, l, h_ in zip(n, lo, hi):
        ax2.plot([xi, xi], [l, h_], color=BLUE, lw=2.0, solid_capstyle="butt", zorder=3)
        ax2.plot([xi / 1.07, xi * 1.07], [l, l], color=BLUE, lw=1.6, zorder=3)
        ax2.plot([xi / 1.07, xi * 1.07], [h_, h_], color=BLUE, lw=1.6, zorder=3)
    ax2.plot(n, m, "o", color=BLUE, ms=8, mec=SURFACE, mew=2, zorder=4)
    ax2.text(n[0] * 1.15, hi[0] + 0.0001, f"width {hi[0] - lo[0]:.4f}", fontsize=10, color=INK, va="bottom")
    ax2.text(n[-1], lo[-1] - 0.0001, f"width {hi[-1] - lo[-1]:.4f}", fontsize=10, color=INK, va="top",
             ha="center")
    ax2.text(0.98, 0.04, f"the true effect ({fmt(target)}) is far above this window;\nnone of the error bars "
             "reaches it", transform=ax2.transAxes, fontsize=10, color=INK2, ha="right", va="bottom")
    ax2.set_title("Its error bars (zoomed in)", fontsize=12, color=INK, loc="left", pad=8)
    for a, digits in ((ax, 3), (ax2, 4)):
        a.set_xticks(n)
        a.set_xticklabels(ticks)
        a.minorticks_off()
        a.set_xlabel("patients before the alert (log scale)")
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _, d=digits: "0" if abs(v) < 1e-12 else f"{v:+.{d}f}".replace("-", "\u2212")))
    header(fig, "More data makes the wrong answer more confident",
           "The usual correction fails when patients like the ones the alert flags were never treated before it "
           "existed.\nIts answer does not move toward the truth as data grows; only its error bars shrink.",
           f"Source: figures/merged_expE7.json (part B) · {len(B['n_scaling']['runs']) // len(n)} simulated runs "
           "per data size · right: 95% error bars (median over runs)")
    save(fig, "03_more_data.png")
    assert m.max() < 0 < target and covered == 0, "the caption describes this pattern"
    CAPTIONS.append((
        "03_more_data.png",
        "When the patients the alert flags were never treated before it existed, the usual correction gets the "
        f"alert's effect backwards ({fmt(m.mean())} instead of {fmt(target)}), and more data only makes it more "
        "confident."))


# ----------------------------------------------------------------------------- 4 retraining
def figure_retraining():
    H = load("merged_expH.json")
    g = {(e["setting"], e["arm"], e["round"]): e for e in H["summary"]}
    rounds = np.arange(1, H["parameters"]["rounds"] + 1)
    series = lambda arm, key: [g[("fixed_k0", arm, r)][key]["mean"] for r in rounds]
    arms = (("keep", "Keep the original model", BLUE), ("naive", "Retrain on the latest data each round", ORANGE))
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.68, bottom=0.16, wspace=0.22)
    ax = axes[0]
    for arm, label, color in arms:
        ax.plot(rounds, series(arm, "events_averted_pp"), color=color, marker="o", ms=7, mec=SURFACE, mew=1.8,
                label=label, zorder=3)
    keep, naive = series("keep", "events_averted_pp"), series("naive", "events_averted_pp")
    weighted = series("untreated_ipw", "events_averted_pp")
    for r in (2, 3):
        ax.text(r, naive[r - 1] + (0.6 if r == 3 else -0.9), f"{naive[r - 1]:.1f}", ha="center",
                va="bottom" if r == 3 else "top", fontsize=10, color=INK)
    ax.text(rounds[-1], keep[-1] + 0.6, f"{keep[-1]:.1f}", ha="center", va="bottom", fontsize=10, color=INK)
    assert min(weighted) > 0.95 * min(keep), "the note says the weighted refit stays on the blue line"
    ax.text(2.4, 10.15, "Retraining only on untreated patients (with a\nstatistical correction) stays on the blue line",
            fontsize=9.5, color=INK2, va="center", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    ax.set_ylim(-1.5, 13.8)
    ax.set_xticks(rounds)
    ax.set_xlabel("round (model retrained after each)")
    ax.set_title("Bad outcomes prevented per 100 patients", fontsize=12, color=INK, loc="left", pad=8)
    baseline(ax)
    ax = axes[1]
    for arm, label, color in arms:
        ax.plot(rounds, series(arm, "observed_auroc"), color=color, marker="o", ms=7, mec=SURFACE, mew=1.8,
                zorder=3)
    au = series("naive", "observed_auroc")
    ax.set_ylim(0.6, 0.78)
    ax.set_xticks(rounds)
    ax.set_xlabel("round (model retrained after each)")
    ax.set_title("What the dashboard shows: accuracy (AUROC)", fontsize=12, color=INK, loc="left", pad=8)
    ax.annotate("best accuracy exactly when\nthe alert prevents nothing", xy=(2, au[1]), xytext=(3.3, 0.765),
                fontsize=10, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0, shrinkA=2, shrinkB=5))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.03, 0.815), ncol=2, columnspacing=2.2,
               handlelength=2.2)
    header(fig, "Retraining on recent data throws the benefit away",
           "Flagged patients are treated and do fine, so a retrained model thinks they are low risk and stops flagging "
           "them.\nIt keeps flipping back and forth, and the accuracy score looks best exactly when the alert helps "
           "no one.",
           f"Source: figures/merged_expH.json · {H['parameters']['seeds']} simulated runs, alert threshold kept "
           "fixed; 95% intervals are narrower than the markers")
    save(fig, "04_retraining.png")
    CAPTIONS.append((
        "04_retraining.png",
        "Retraining on recent data teaches the model that treated patients are low risk, so the alert keeps "
        "switching off and on, and the accuracy score looks best exactly when the alert helps no one."))


# ----------------------------------------------------------------------------- 5 refit rules
def figure_refit_rules():
    E1 = load("merged_expE1.json")
    worst = {w["rule"]: w["max_mean_regret_pp"] for w in E1["worst_case"] if w["threshold_rule"] == "fixed"}
    rules = (("always", "Always retrain", False),
             ("outcome_monitor", "Retrain if risk changed, per standard monitoring", False),
             ("exo_proposed", "Retrain unless the alert did most of it, per our method", False),
             ("never", "Never retrain", False),
             ("outcome_proposed", "Retrain if risk changed, per our method", True),
             ("refit_untreated", "Retrain on untreated patients (needs treatment records)", False))
    vals = [max(0.0, worst[r]) for r, _, _ in rules]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.46, right=0.95, top=0.74, bottom=0.15)
    ys = np.arange(len(rules))[::-1]
    ax.set_xlim(-0.3, 13.2)
    ax.set_ylim(-0.6, len(rules) - 0.4)
    ax.grid(axis="y", visible=False)
    for y, v, (rule, label, hl) in zip(ys, vals, rules):
        color = BLUE if hl else MUTED
        ax.plot([0, v], [y, y], color=color, lw=2, zorder=3)
        ax.plot(v, y, "o", color=color, ms=10, mec=SURFACE, mew=2, zorder=4)
        ax.text(v + 0.35, y, f"{v:.1f}", va="center", ha="left", fontsize=11, color=INK,
                fontweight="bold" if hl else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in rules], fontsize=10.5)
    for t, (_, _, hl) in zip(ax.get_yticklabels(), rules):
        t.set_color(INK)
        t.set_fontweight("bold" if hl else "normal")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    baseline(ax, horizontal=True)
    ax.set_xlabel("extra bad outcomes per 100 patients, worst case (lower is better)")
    header(fig, "Knowing whether risk really changed tells you when to retrain",
           "After a monitoring period, should the hospital keep the model or retrain it? Each rule makes that call."
           "\nScore: extra bad outcomes compared with the better choice, worst case over four simulated situations.",
           f"Source: figures/merged_expE1.json · {E1['parameters']['reps']} simulated runs per situation, alert "
           "threshold kept fixed")
    save(fig, "05_refit_rules.png")
    assert worst["outcome_proposed"] < 0.05 and worst["outcome_monitor"] > 0.95 * worst["always"], \
        "the caption describes this pattern"
    CAPTIONS.append((
        "05_refit_rules.png",
        "Deciding when to retrain with our method never cost extra bad outcomes in our tests, while the same rule "
        f"using standard monitoring cost up to {worst['outcome_monitor']:.1f} extra per 100 patients."))


# ----------------------------------------------------------------------------- 6 auroc vs benefit
def figure_auroc_vs_benefit():
    runs = load("merged_expE7.json")["D"]
    D = sorted((r for r in runs["summary"] if r["theta"] == 2.5), key=lambda r: r["p_alert"])
    reps = sum(r["theta"] == 2.5 and r["p_alert"] == D[0]["p_alert"] for r in runs["runs"])
    p = np.array([r["p_alert"] for r in D])
    au = np.array([r["auroc_e1"]["mean"] for r in D])
    au0 = float(np.mean([r["auroc_e0"]["mean"] for r in D]))
    av = np.array([100 * r["averted_observed"]["mean"] for r in D])
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.74, bottom=0.16, wspace=0.22)
    ax = axes[0]
    ax.axhline(au0, color=INK2, lw=1.0, zorder=2)
    ax.text(0.5, au0 + 0.004, f"before the alert: {au0:.2f}", fontsize=10, color=INK2, va="bottom", ha="center")
    ax.plot(p, au, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=1.8, zorder=3)
    ax.text(p[0], au[0] - 0.008, f"{au[0]:.2f}", ha="center", va="top", fontsize=10.5, color=INK)
    ax.text(p[-1], au[-1] - 0.008, f"{au[-1]:.2f}", ha="center", va="top", fontsize=10.5, color=INK)
    ax.set_ylim(0.6, 0.77)
    ax.set_title("Accuracy after the alert goes live (AUROC)", fontsize=12, color=INK, loc="left", pad=8)
    ax = axes[1]
    ax.plot(p, av, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=1.8, zorder=3)
    ax.text(p[0], av[0] + 0.5, f"{av[0]:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.text(p[-1], av[-1] + 0.5, f"{av[-1]:.1f}", ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_ylim(0, 14)
    baseline(ax)
    ax.set_title("Bad outcomes prevented per 100 patients", fontsize=12, color=INK, loc="left", pad=8)
    for a in axes:
        a.set_xlim(0.03, 0.97)
        a.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
        a.set_xticklabels(["10%", "30%", "50%", "70%", "90%"])
        a.set_xlabel("share of alerts clinicians act on")
    header(fig, "Measured accuracy falls as the alert helps more patients",
           "Same model, same patients; the only thing that changes is how often clinicians act on an alert.",
           f"Source: figures/merged_expE7.json (part D) · {reps} simulated runs per setting; 95% intervals are "
           "narrower than the markers")
    save(fig, "06_auroc_vs_benefit.png")
    assert all(np.diff(au) < 0) and all(np.diff(av) > 0), "the caption describes this pattern"
    CAPTIONS.append((
        "06_auroc_vs_benefit.png",
        "The more often clinicians act on alerts, the more bad outcomes are prevented and the lower the model's "
        "measured accuracy, so falling accuracy after launch can mean the alert is working."))


# ----------------------------------------------------------------------------- 7 prevalence
def figure_prevalence():
    P = load("merged_expP.json")
    rows = P["rows"]
    rate = [r["event_rate_untreated"]["mean"] for r in rows]
    dr = [r["dR"]["mean"] for r in rows]
    share = [r["monitor_Ygx"]["mean"] / r["dR"]["mean"] for r in rows]
    auroc_fall = all(r["auroc_after"]["mean"] < r["auroc_before"]["mean"] for r in rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.75, bottom=0.165)
    x = np.arange(len(rows))
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(-0.012, 0.024)
    ax.grid(axis="x", visible=False)
    baseline(ax)
    for xi, v in zip(x, dr):
        rounded_bar(ax, xi, 0.11, v, RED if v > 0 else BLUE)
        ax.text(xi, v + (0.0009 if v > 0 else -0.0009), fmt(v, 4), ha="center", va="bottom" if v > 0 else "top",
                fontsize=11, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{100 * r:.0f}%" for r in rate], fontsize=12, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("share of patients who would have the bad outcome without the alert")
    ax.set_ylabel("change in prediction error")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+.3f}".replace("-", "\u2212")))
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["looks like a performance drop", "looks like an improvement"], loc="upper right",
               bbox_to_anchor=(0.97, 0.835), ncol=2, handlelength=1.0, handleheight=1.0, columnspacing=1.6)
    assert auroc_fall, "the subtitle says AUROC falls in every case"
    assert dr[0] > 0 > dr[2] and dr[3] < 0, "the caption describes this pattern"
    header(fig, "When the bad outcome is uncommon, a working alert looks like an improvement",
           f"Same alert (it flags {P['parameters']['alert_rate']:.0%} of patients); only how common the bad outcome "
           f"is changes.\nAccuracy (AUROC) falls every time, and standard monitoring blames changed risk for "
           f"{min(share):.0%} or more of each change.",
           f"Source: figures/merged_expP.json · averaged over {P['parameters']['blocks']} batches of "
           f"{P['parameters']['n']:,} simulated patients")
    save(fig, "07_prevalence.png")
    CAPTIONS.append((
        "07_prevalence.png",
        f"When the bad outcome is uncommon ({100 * rate[3]:.0f} to {100 * rate[2]:.0f}% of patients, as on real "
        "wards), a working alert lowers prediction error, which standard monitoring reads as patients' risk going "
        "down."))


# ----------------------------------------------------------------------------- captions
def write_captions():
    order = sorted(CAPTIONS)
    lines = [f"# Status update, {DATE}", "",
             "Figures for the team channel, generated by `python3 sim2/make_status_update.py` from the result "
             "files in `figures/` (every number below is read from those files).", "",
             "## Post", "",
             "When clinicians act on a hospital AI alert, the model's measured accuracy changes even though nothing "
             "is broken, and we show that standard monitoring blames the wrong cause, while our method gets it right "
             "by holding the alert back for a random group of patients.",
             "",
             "All the simulation experiments are done; the one missing piece is MIMIC-IV, real hospital records that "
             "someone with approved PhysioNet access needs to run through `sim2/run_e6.sh all <username>` on their "
             "own computer (about 3 GB to download, 20 to 40 minutes to run, and only summary numbers are saved).",
             "", "## Captions", ""]
    for name, text in order:
        lines += [f"`{name}`", "", text, ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print("wrote docs/status_update/README.md")


def main():
    figure_workflow()
    figure_attribution()
    figure_more_data()
    figure_retraining()
    figure_refit_rules()
    figure_auroc_vs_benefit()
    figure_prevalence()
    figure_real_covariates()
    write_captions()


if __name__ == "__main__":
    main()
