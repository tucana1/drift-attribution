"""Check the saved E1-E7 evidence, mirrored files, generated tables and quoted numbers.

    python3 sim2/validate_josh_results.py

Every check reads files in figures/, aaai/figures/, aaai/tables/ and the
manuscript; nothing is recomputed from scratch except the generated tables,
which are rebuilt in memory and compared byte for byte.
"""
import importlib
import io
import json
import math
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
TEX = (ROOT / "aaai/drift_attribution_aaai.tex").read_text()


def load(name):
    root_path = ROOT / "figures" / name
    mirror_path = ROOT / "aaai/figures" / name
    assert root_path.read_bytes() == mirror_path.read_bytes(), f"{name} differs between figures/ and aaai/figures/"
    return json.loads(root_path.read_text())


def same_figure(name):
    a, b = ROOT / "figures" / name, ROOT / "aaai/figures" / name
    assert a.exists() and b.exists() and a.read_bytes() == b.read_bytes(), f"{name} missing or not mirrored"


def close(a, b, tol=1e-9):
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) < tol, (a, b)


def rows_of(columns):
    keys = list(columns)
    return [dict(zip(keys, vals)) for vals in zip(*(columns[k] for k in keys))]


def quoted(text, where="manuscript"):
    assert text in TEX, f"{where}: expected to find {text!r}"


def signed(v, digits=4):
    return f"${v:+.{digits}f}$"


def with_ci(entry, digits=4, sign=True):
    f = (lambda v: f"${v:+.{digits}f}$") if sign else (lambda v: f"${v:.{digits}f}$")
    return f"{f(entry['mean'])} [{f(entry['ci95'][0])}, {f(entry['ci95'][1])}]"


def check_h():
    d = load("merged_expH.json")
    p = d["parameters"]
    runs = rows_of(d["runs"])
    assert len(runs) == len(p["settings"]) * p["seeds"] * p["rounds"] * len(p["arms"])
    refs = {}
    for r in runs:
        close(r["events_averted_pp"], 100 * (r["standard_care_events"] - r["events"]))
        refs.setdefault((r["setting"], r["seed"]), set()).add(r["standard_care_events"])
    assert all(len(v) == 1 for v in refs.values()), "one standard-care reference per setting and seed"
    for e in d["summary"][:: 7]:
        vals = [r["events_averted_pp"] for r in runs
                if (r["setting"], r["arm"], r["round"]) == (e["setting"], e["arm"], e["round"])]
        close(e["events_averted_pp"]["mean"], mean(vals))
    first_keep = {(r["setting"], r["seed"]): r["events"] for r in runs if r["arm"] == "keep" and r["round"] == 1}
    for r in runs:
        if r["arm"] == "keep":
            close(r["events"], first_keep[(r["setting"], r["seed"])], 1e-15)
    same_figure("fig3_retrain.pdf")
    g = {(e["setting"], e["arm"], e["round"]): e for e in d["summary"]}
    fmt = lambda v: f"{v:.2f}"
    quoted(f"${fmt(g[('fixed_k0', 'keep', 1)]['events_averted_pp']['mean'])}$ [${fmt(g[('fixed_k0', 'keep', 1)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${fmt(g[('rate_k2', 'naive', 8)]['events_averted_pp']['mean'])}$ [${fmt(g[('rate_k2', 'naive', 8)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${fmt(g[('fixed_nl', 'untreated_ipw', 8)]['events_averted_pp']['mean'])}$ [${fmt(g[('fixed_nl', 'untreated_ipw', 8)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${abs(g[('rate_k0', 'naive', 8)]['calibration_in_large']['mean']):.3f}$ below the untreated risk")
    quoted(f"${g[('fixed_k0', 'naive', 2)]['observed_auroc']['mean']:.3f}$ in the rounds")
    gap = max(abs(g[(s["name"], "untreated_ipw_est", p["rounds"])]["events_averted_pp"]["mean"]
                  - g[(s["name"], "untreated_ipw", p["rounds"])]["events_averted_pp"]["mean"]) for s in p["settings"])
    assert gap <= 0.01, f"estimated and logged weights differ by {gap:.3f} points"
    quoted("by at most $0.01$ points in any setting")


def check_e1():
    d = load("merged_expE1.json")
    runs = rows_of(d["runs"])
    reps = d["parameters"]["reps"]
    for t in d["table"]:
        vals = [r["events_averted_pp"] for r in runs if (r["threshold_rule"], r["scenario"], r["rule"]) ==
                (t["threshold_rule"], t["scenario"], t["rule"])]
        assert len(vals) == reps
        close(t["events_averted_pp"]["mean"], mean(vals))
    by = {}
    for r in runs:
        by.setdefault((r["threshold_rule"], r["scenario"], r["rep"]), {})[r["rule"]] = r
    for cell in by.values():
        best = min(cell["never"]["events"], cell["always"]["events"])
        close(cell["best_of_two"]["events"], best)
        assert cell["never"]["retrained"] is False and cell["always"]["retrained"] is True
        for rule in ("exo_proposed", "outcome_proposed", "exo_monitor", "outcome_monitor"):
            src = cell["always"] if cell[rule]["retrained"] else cell["never"]
            close(cell[rule]["events"], src["events"])
    same_figure("fig6_decision.pdf")
    t = {(e["threshold_rule"], e["scenario"], e["rule"]): e for e in d["table"]}
    quoted(f"${t[('fixed', 'S2', 'exo_proposed')]['events_averted_pp']['mean']:.1f}$\nagainst ${t[('fixed', 'S2', 'never')]['events_averted_pp']['mean']:.1f}$")
    quoted(f"${100 * t[('fixed', 'S4', 'exo_proposed')]['retrain_fraction']:.0f}\\%$ of S4 replicates")
    loss_s1 = t[("fixed", "S1", "never")]["events_averted_pp"]["mean"] - t[("fixed", "S1", "exo_proposed")]["events_averted_pp"]["mean"]
    quoted(f"costs ${loss_s1:.1f}$ points")
    table_rules = ("never", "always", "exo_proposed", "outcome_proposed", "outcome_monitor", "refit_untreated")
    gap = max(abs(t[("rate", sc, a)]["events_averted_pp"]["mean"] - t[("rate", sc, b)]["events_averted_pp"]["mean"])
              for sc in ("S1", "S2", "S3", "S4") for a in table_rules for b in table_rules)
    assert gap <= 1.3, f"rate-held gap {gap:.2f} exceeds the quoted 1.3 points"
    quoted("no two of the rules in\nTable~\\ref{tab:decision} differ by more than\n$1.3$ points")


def check_ab():
    d = load("merged_expAB_robustness.json")
    assert {(x["d"], x["nonlinear_outcome"]) for x in d["settings"]} == {(10, 0.0), (10, 0.65), (20, 0.0), (20, 0.65)}
    for setting in d["settings"]:
        assert {r["scenario"] for r in setting["A"]} == {"S1", "S2", "S3", "S4"}
        s3 = next(r for r in setting["A"] if r["scenario"] == "S3")
        naive = s3["summary"]["naive2_Ygx"]["mean"]
        total = mean(r["naive2_est"]["dR"] for r in s3["runs"])
        assert 0.9 < naive / total < 1.1, "F1 survives"
        for scale in setting["n_scaling_eps_zero"]:
            close(scale["union3_raw_pi"]["mean"], mean(scale["runs"]))
            assert scale["union3_raw_pi"]["bias"] < -0.02, "F2 survives"


def check_g2():
    d = load("merged_expG2.json")
    assert len(d["rows"]) == 4
    for row in d["rows"]:
        assert len(row["runs"]) == d["parameters"]["seeds_per_row"]
        for run in row["runs"]:
            close(run["retro_truth"], run["x_endogenous_loss_truth"] + run["policy_on_current_x_truth"])
        for est, entry in row["summary"].items():
            if not isinstance(entry, dict):
                continue
            bias = mean(r[est + "_est"] - r[entry["target"] + "_truth"] for r in row["runs"])
            close(entry["bias_mean"], bias)
    s = {r["setting"]: r["summary"] for r in d["rows"]}
    quoted(f"${100 * s['correlated_order_common_trend']['retro']['relative_bias']:+.1f}\\%$")
    quoted(f"${100 * s['correlated_order_site_trend']['x_endogenous_mean']['relative_bias']:+.1f}\\%$ induced")


def check_e7():
    path = ROOT / "figures/merged_expE7.json"
    if not path.exists():
        print("  merged_expE7.json not present: skipped")
        return
    d = load("merged_expE7.json")
    A = d["A"]
    for sc, summ in A["summary"].items():
        rr = [r for r in A["runs"] if r["scenario"] == sc]
        for key in ("E3_rollout_pi", "naive2_est_Ygx", "union3_raw_pi"):
            close(summ[key]["mean"], mean(r[key] for r in rr))
        for cov in A["bootstrap_coverage"][sc].values():
            assert 0 <= cov["coverage"] <= 1 and cov["replicates"] <= len(rr)
    for r in d["B"]["summary"]:
        rr = [x for x in d["B"]["runs"] if x["eps"] == r["eps"]]
        close(r["E3_rollout_pi"]["mean"], mean(x["E3_rollout_pi"] for x in rr))
    for fig in ("fig1_attribution.pdf", "fig2_positivity.pdf", "fig4_triage.pdf", "fig5_utility.pdf"):
        same_figure(fig)
    # Two references, F1 (S3, S4)
    S3, S4, R3, R4 = A["summary"]["S3"], A["summary"]["S4"], A["references"]["S3"], A["references"]["S4"]
    for key in ("dR_observed", "naive2_est_Ygx", "naive2_est_X", "union3_raw_pi", "union3_clip_pi", "E3_rollout_pi",
                "E3_oracle_pi"):
        quoted(with_ci(S3[key]), f"F1 ({key})")
    quoted(f"{signed(R3['shapley_pi']['value'])} vs.\\ {signed(R3['pc_pi']['value'])}", "two references (S3)")
    quoted(f"{signed(R4['shapley_pi']['value'])} vs.\\ {signed(R4['pc_pi']['value'])}", "two references (S4)")
    quoted(f"$I(X, \\pi) = {R4['I_Xpi']['value']:+.4f}$")
    quoted(f"$I(\\pi, Y) = {R4['I_piY']['value']:+.4f}$")
    quoted(f"${S4['E3_rollout_X']['mean']:+.4f} / {S4['E3_rollout_Y']['mean']:+.4f} / {S4['E3_rollout_pi']['mean']:+.4f}$")
    quoted(f"${R4['pc_X']['value']:+.4f} / {R4['pc_Y']['value']:+.4f} / {R4['pc_pi']['value']:+.4f}$")
    quoted(f"single-run SD ${S3['naive2_est_Ygx']['sd']:.4f}$")
    quoted(f"effective sample size ${S3['ess']['mean']:.0f}$")
    # F2 (eps = 0, n scaling)
    b0 = next(r for r in d["B"]["summary"] if r["eps"] == 0.0)
    quoted(with_ci(b0["union3_raw_pi"]), "F2 (eps = 0)")
    quoted(f"${b0['ess']['mean']:,.0f}$".replace(",", "{,}"), "F2 (ESS)")
    ns = d["B"]["n_scaling"]["summary"]
    for r in (ns[0], ns[-1]):
        lo, hi = r["analyst_ci95_median"]
        quoted(f"[${lo:.4f}$, ${hi:.4f}$]", "F2 (analyst interval)")
        assert not lo <= r["truth_shapley_pi"] <= hi
    quoted(f"${100 * ns[-1]['union3_raw_pi']['bias'] / ns[-1]['truth_shapley_pi']:.0f}\\%$", "F2 (bias)")
    # F6 (utility, net benefit, harm, threshold discontinuity)
    D = {(r["theta"], r["p_alert"]): r for r in d["D"]["summary"]}
    quoted(with_ci(D[(2.5, 0.9)]["auroc_e1"], 3, sign=False), "F6 (AUROC)")
    quoted(with_ci(D[(2.5, 0.1)]["nb_std"], 3), "F6 (net benefit)")
    quoted(with_ci(D[(2.5, 0.9)]["nb_std"], 3), "F6 (net benefit)")
    quoted(f"${D[(2.5, 0.85)]['nb_standard_care']['mean']:+.3f}$ at\n$\\theta = 2.5$", "F6 (net benefit, standard care)")
    harm = D[(2.5, 0.85)]["net_averted_pp_by_harm"]
    quoted(f"${harm['0.0']['mean']:.2f}$ to ${harm['0.1']['mean']:.2f}$ points", "F6 (harm)")
    E = d["E"]
    quoted(with_ci(E["rd_local"], 3), "F6 (RD jump)")
    quoted(f"average policy effect of ${E['policy_effect_alerted']['mean']:.3f}$", "F6 (policy effect)")
    quoted(f"overstates by ${100 * (E['rd_relative_to_policy'] - 1):.0f}\\%$", "F6 (RD gap)")
    quoted(f"(${E['treatment_effect_alerted']['mean']:.3f}$)", "F6 (treatment effect)")


def check_prevalence_estimators():
    """Low-prevalence Experiment A and E1 (appendix, Table tab:prevalence-est)."""
    a, e1 = load("merged_expE7_prev.json"), load("merged_expE1_prev.json")
    A, op = a["A"], a["A"]["operating_point"]
    assert a["parameters"]["b0"] == e1["parameters"]["b0"] and a["parameters"]["parts"] == "A"
    quoted(f"untreated event rate of ${100 * op['event_rate_untreated']:.1f}\\%$ (alert rate ${100 * op['alert_rate']:.0f}\\%$)")
    S3, S4, C3 = A["summary"]["S3"], A["summary"]["S4"], A["bootstrap_coverage"]["S3"]
    quoted(f"books {with_ci(S3['naive2_est_Ygx'])} to $P(Y \\mid X)$, and its interval")
    quoted(f"zero, in ${100 * C3['naive2_Ygx_vs_policy_contrast_Y']['coverage']:.0f}\\%$ of replicates")
    quoted(f"policy term with ${100 * C3['E3_rollout_pi_vs_policy_contrast']['coverage']:.0f}\\%$ coverage")
    R4 = A["references"]["S4"]
    quoted(f"policy term (${R4['pc_pi']['value']:+.4f}$) offsets")
    quoted(f"(${R4['pc_Y']['value']:+.4f}$): the monitor books {with_ci(S4['naive2_est_Ygx'])}")
    t = {(x["scenario"], x["rule"]): x for x in e1["table"] if x["threshold_rule"] == "fixed"}
    assert t[("S3", "outcome_monitor")]["retrain_fraction"] == 1.0
    quoted(f"keeps ${t[('S3', 'outcome_monitor')]['events_averted_pp']['mean']:.2f}$ of\nthe "
           f"${t[('S3', 'never')]['events_averted_pp']['mean']:.2f}$ points")
    quoted(f"loses ${t[('S4', 'outcome_proposed')]['regret_vs_best_of_two_pp']['mean']:.2f}$ points in S4")
    o = e1["truth"]["S4"]["policy_contrast"]
    quoted(f"outcome share (${o['Y'] / o['dR']:.2f}$)")
    for sc, better in (("S1", False), ("S2", True), ("S3", False), ("S4", True)):
        reg = t[(sc, "refit_untreated")]["regret_vs_best_of_two_pp"]
        assert (reg["ci95"][1] < 0) if better else (reg["ci95"][0] <= 0 <= reg["ci95"][1] or reg["ci95"][1] < 0), sc


def check_e6_run(name, d):
    assert d["semi_synthetic"] is True and d["synthetic_fixture"] is False, f"{name}: E6 output must come from MIMIC-IV"
    assert d["cohort"]["admissions"] >= 5000, f"{name}: demo-sized cohort, not the full MIMIC-IV"
    A = d["A"]
    for sc, summ in A["summary"].items():
        rr = [r for r in A["runs"] if r["scenario"] == sc]
        assert len(rr) == d["parameters"]["reps_a"], f"{name}: {sc} replicate count"
        for key in ("proposed_pi", "monitor_Ygx", "proposed_exogenous"):
            close(summ[key]["mean"], mean(r[key] for r in rr))
        for r in rr:
            close(r["proposed_exogenous"], r["proposed_X"] + r["proposed_Y"], 1e-12)
            close(r["dR_observed"], r["proposed_exogenous"] + r["proposed_pi"], 1e-12)
    assert all(r["proposed_pi"] == 0 for r in A["runs"] if r["scenario"] == "drift_only"), "no policy term without deployment"
    keep = [r["events_averted_pp"] for r in d["C"]["runs"] if r["arm"] == "keep" and r["threshold_rule"] == "fixed"]
    assert max(keep) - min(keep) < 1e-9, f"{name}: keeping the score must not vary across replicates or rounds"


def check_e6():
    path = ROOT / "figures/merged_expE6.json"
    tagged = sorted(p.name for p in (ROOT / "figures").glob("merged_expE6_*.json"))
    if not path.exists():
        assert not tagged, "sensitivity runs present without the main E6 run"
        print("  merged_expE6.json not present (needs the credentialed run): skipped")
        return
    from e6_report import quotes
    d = load("merged_expE6.json")
    check_e6_run("merged_expE6.json", d)
    same_figure("fig7_mimic.pdf")
    for name in tagged:
        check_e6_run(name, load(name))
        same_figure(name.replace("merged_expE6", "fig7_mimic").replace(".json", ".pdf"))
    assert "\\todo{Results from \\texttt{merged\\_expE6.json}" not in TEX, "F7 TODO still open after the E6 run"
    assert "\\todo{Numbers and Figure~\\ref{fig:mimic}" not in TEX, "Appendix E6 TODO still open after the E6 run"
    for label, text in quotes(d).items():
        quoted(text, f"F7 ({label})")


def check_e6_open():
    """Open-data stand-in for E6 (figures/e6_open/); not used in the manuscript."""
    files = sorted((ROOT / "figures/e6_open").glob("merged_expE6_*.json"))
    if not files:
        print("  no open stand-in runs: skipped")
        return
    for path in files:
        d = json.loads(path.read_text())
        assert d["data_source"].startswith("PhysioNet/CinC Challenge 2019"), f"{path.name}: unexpected data source"
        check_e6_run(path.name, d)
        fig = path.parent / path.name.replace("merged_expE6", "fig_e6").replace(".json", ".pdf")
        assert fig.exists(), f"{fig.name} missing"


def check_tables():
    import make_tables
    import make_tables_e7
    generated = {}

    def capture(name, lines):
        generated[name] = lines
    for mod, prefix in ((make_tables, "% generated by sim2/make_tables.py; do not edit by hand\n"),
                        (make_tables_e7, "% generated by sim2/make_tables_e7.py; do not edit by hand\n")):
        original = mod.write
        mod.write = lambda name, lines, p=prefix: generated.__setitem__(name, p + "\n".join(lines) + "\n")
        try:
            with redirect_stdout(io.StringIO()):
                if mod is make_tables:
                    mod.table_e1(); mod.table_e3(); mod.table_e4(); mod.table_e5(); mod.table_prevalence()
                    mod.table_prevalence_estimators(); mod.table_e6(); mod.table_e6_sensitivity()
                else:
                    mod.tables_e7()
        finally:
            mod.write = original
    for name, text in generated.items():
        path = ROOT / "aaai/tables" / f"{name}.tex"
        assert path.exists() and path.read_text() == text, f"aaai/tables/{name}.tex is stale: run sim2/make_tables.py"
        assert f"\\input{{tables/{name}}}" in TEX, f"tables/{name} is not included in the manuscript"


def check_style():
    assert "\u2014" not in TEX and "---" not in re.sub(r"%.*", "", TEX), "no em dashes (team convention)"


if __name__ == "__main__":
    for name, fn in (("E3 (Experiment H)", check_h), ("E1 decision rules", check_e1), ("E4 robustness", check_ab),
                     ("E5 sites", check_g2), ("E7 intervals", check_e7),
                     ("low prevalence", check_prevalence_estimators), ("E6 MIMIC-IV", check_e6),
                     ("E6 open stand-in", check_e6_open),
                     ("generated tables", check_tables), ("style", check_style)):
        fn()
        print(f"ok  {name}")
    print("Saved E1-E7 evidence, mirrored files, tables and quoted numbers validated")
