"""Numbers for F7 from figures/merged_expE6.json, in the format the validator checks.

    python3 sim2/e6_report.py [path/to/merged_expE6.json]

Prints each quantity F7 quotes and a draft of the F7 result sentences whose
wording follows the sign of the policy term. check_e6 in
validate_josh_results.py requires every string returned by quotes() to appear
in the manuscript once the E6 run exists. The appendix numbers (cohort, event
rates, AUROC, sensitivity runs) are in the generated tables
aaai/tables/e6_mimic.tex and e6_sensitivity.tex (sim2/make_tables.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def brier_digits(d):
    """Four decimals as elsewhere in the paper, five when the deployment-only policy term is below 1e-3."""
    return 4 if abs(d["A"]["truth"]["deploy_only"]["pi"]) >= 1e-3 else 5


def value(v, digits, sign=True):
    return f"${v:+.{digits}f}$" if sign else f"${v:.{digits}f}$"


def with_ci(entry, digits, sign=True):
    lo, hi = entry["ci95"]
    if hi - lo < 0.5 * 10 ** -digits:       # no Monte Carlo spread (e.g. keeping the score)
        return value(entry["mean"], digits, sign)
    return f"{value(entry['mean'], digits, sign)} [{value(lo, digits, sign)}, {value(hi, digits, sign)}]"


def final_round(d, rule="fixed"):
    last = d["parameters"]["rounds"]
    return {e["arm"]: e for e in d["C"]["summary"] if e["threshold_rule"] == rule and e["round"] == last}


def quotes(d):
    """Formatted numbers F7 must contain (name -> LaTeX string)."""
    k = brier_digits(d)
    A = d["A"]
    dep, both = A["summary"]["deploy_only"], A["summary"]["drift_and_deploy"]
    r8 = final_round(d)
    return {
        "policy contrast, deployment only": value(A["truth"]["deploy_only"]["pi"], k),
        "randomised-arm policy term, deployment only": with_ci(dep["proposed_pi"], k),
        "monitor P(Y|X), deployment only": with_ci(dep["monitor_Ygx"], k),
        "union-graph policy term, deployment only": with_ci(dep["union_pi"], k),
        "observed AUROC change, deployment only": value(dep["auroc_change_observed"]["mean"], 3),
        "policy contrast, drift and deployment": value(A["truth"]["drift_and_deploy"]["pi"], k),
        "randomised-arm policy term, drift and deployment": with_ci(both["proposed_pi"], k),
        "monitor P(Y|X), drift and deployment": with_ci(both["monitor_Ygx"], k),
        "events averted at the last round, keep": with_ci(r8["keep"]["events_averted_pp"], 2, sign=False),
        "events averted at the last round, naive refit": with_ci(r8["naive"]["events_averted_pp"], 2, sign=False),
        "events averted at the last round, refit on A=0": with_ci(r8["untreated"]["events_averted_pp"], 2, sign=False),
        "events averted at the last round, weighted refit": with_ci(r8["untreated_ipw"]["events_averted_pp"], 2, sign=False),
    }


def draft(d):
    q = quotes(d)
    pi = d["A"]["truth"]["deploy_only"]["pi"]
    moves = "lowers" if pi < 0 else "raises"
    reading = "an apparent improvement" if pi < 0 else "a drop"
    rounds = d["parameters"]["rounds"]
    return (
        f"With deployment only, the alert policy {moves} the Brier score: the policy contrast is "
        f"{q['policy contrast, deployment only']} and the randomised arm estimates "
        f"{q['randomised-arm policy term, deployment only']}. The monitor books "
        f"{q['monitor P(Y|X), deployment only']} to $P(Y \\mid X)$, {reading} of the outcome mechanism, while observed "
        f"AUROC changes by {q['observed AUROC change, deployment only']}. The union-graph estimator with oracle policy "
        f"weights returns {q['union-graph policy term, deployment only']}. With real drift between periods the "
        f"policy contrast is {q['policy contrast, drift and deployment']}, the randomised arm estimates "
        f"{q['randomised-arm policy term, drift and deployment']}, and the monitor assigns "
        f"{q['monitor P(Y|X), drift and deployment']} to $P(Y \\mid X)$. Under the fixed threshold, events averted "
        f"at round {rounds} are {q['events averted at the last round, keep']} points when the score is kept, "
        f"{q['events averted at the last round, naive refit']} after naive refitting, "
        f"{q['events averted at the last round, refit on A=0']} after refitting on $A=0$ and "
        f"{q['events averted at the last round, weighted refit']} after the weighted refit."
    )


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "figures/merged_expE6.json"
    if not path.exists():
        print(f"{path} not found: run sim2/exp_E6_mimic.py first")
        return
    d = json.loads(path.read_text())
    c = d["cohort"]
    print(f"cohort: {c['admissions']:,} admissions, event rate {100 * c['event_rate_early']:.2f}% early, "
          f"{100 * c['event_rate_late']:.2f}% late; score AUROC {c['score_auroc_early_heldout']:.3f} early held out, "
          f"{c['score_auroc_late']:.3f} late")
    if d.get("synthetic_fixture"):
        print("WARNING: synthetic fixture, not MIMIC-IV; nothing here is a result")
    print("\nquoted numbers (the validator looks for each string in the manuscript):")
    for name, text in quotes(d).items():
        print(f"  {name}: {text}")
    print("\ndraft F7 results (replace the \\todo at the end of F7):\n")
    print(draft(d))


if __name__ == "__main__":
    main()
