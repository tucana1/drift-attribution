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
    for r in runs:
        if r["arm"] == "keep":
            assert r["round"] == 1 or r["events"] == next(
                x["events"] for x in runs if x["setting"] == r["setting"] and x["seed"] == r["seed"]
                and x["arm"] == "keep" and x["round"] == 1)
            break
    same_figure("fig3_retrain.pdf")
    g = {(e["setting"], e["arm"], e["round"]): e for e in d["summary"]}
    fmt = lambda v: f"{v:.2f}"
    quoted(f"${fmt(g[('fixed_k0', 'keep', 1)]['events_averted_pp']['mean'])}$ [${fmt(g[('fixed_k0', 'keep', 1)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${fmt(g[('rate_k2', 'naive', 8)]['events_averted_pp']['mean'])}$ [${fmt(g[('rate_k2', 'naive', 8)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${fmt(g[('fixed_nl', 'untreated_ipw', 8)]['events_averted_pp']['mean'])}$ [${fmt(g[('fixed_nl', 'untreated_ipw', 8)]['events_averted_pp']['ci95'][0])}$")
    quoted(f"${abs(g[('rate_k0', 'naive', 8)]['calibration_in_large']['mean']):.3f}$ below the untreated risk")
    quoted(f"${g[('fixed_k0', 'naive', 2)]['observed_auroc']['mean']:.3f}$ in the rounds")


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
    quoted(f"$-{abs(100 * s['correlated_order_common_trend']['retro']['relative_bias']):.1f}\\%$")
    quoted(f"$+{100 * s['correlated_order_site_trend']['x_endogenous_mean']['relative_bias']:.1f}\\%$ induced")


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


def check_e6():
    path = ROOT / "figures/merged_expE6.json"
    if not path.exists():
        print("  merged_expE6.json not present (needs the credentialed run): skipped")
        return
    d = load("merged_expE6.json")
    assert d["semi_synthetic"] is True and d["synthetic_fixture"] is False, "E6 output must come from MIMIC-IV"
    same_figure("fig7_mimic.pdf")


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
                     ("E5 sites", check_g2), ("E7 intervals", check_e7), ("E6 MIMIC-IV", check_e6),
                     ("generated tables", check_tables), ("style", check_style)):
        fn()
        print(f"ok  {name}")
    print("Saved E1-E7 evidence, mirrored files, tables and quoted numbers validated")
