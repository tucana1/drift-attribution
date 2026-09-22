"""Check saved E3-E5 evidence, numerical references, and mirrored figure files."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    root_path = ROOT / "figures" / name
    mirror_path = ROOT / "aaai/figures" / name
    assert root_path.read_bytes() == mirror_path.read_bytes(), name
    return json.loads(root_path.read_text())


def close(a, b, tol=1e-10):
    assert math.isfinite(a) and math.isfinite(b) and abs(a - b) < tol, (a, b)


def check_h():
    data = load("merged_expH.json")
    pars = data["parameters"]
    assert len(data["runs"]) == 2 * pars["seeds"] * pars["rounds"] * len(pars["arms"])
    assert len(pars["arms"]) == 5 and pars["kappa"] == [0.0, 2.0]
    for row in data["runs"]:
        close(row["events_averted_pp"], 100 * (row["standard_care_events"] - row["events"]))
    for kappa in pars["kappa"]:
        for seed in range(pars["seeds"]):
            references = {r["standard_care_events"] for r in data["runs"]
                          if r["kappa"] == kappa and r["seed"] == seed}
            assert len(references) == 1
    for row in data["summary"]:
        values = [r["events_averted_pp"] for r in data["runs"]
                  if (r["kappa"], r["arm"], r["round"]) ==
                     (row["kappa"], row["arm"], row["round"])]
        close(row["mean_events_averted_pp"], mean(values))
    figure = ROOT / "figures/fig3_retrain.pdf"
    assert figure.read_bytes() == (ROOT / "aaai/figures/fig3_retrain.pdf").read_bytes()


def check_ab():
    data = load("merged_expAB_robustness.json")
    assert {(x["d"], x["nonlinear_outcome"]) for x in data["settings"]} == {
        (10, 0.0), (10, 0.65), (20, 0.0), (20, 0.65)}
    for setting in data["settings"]:
        assert {r["scenario"] for r in setting["A"]} == {"S1", "S2", "S3", "S4"}
        assert [r["eps"] for r in setting["B"]] == [0, 1e-4, 1e-3, 1e-2, 5e-2]
        assert [r["n"] for r in setting["n_scaling_eps_zero"]] == [6000, 24000, 96000]
        s3 = next(r for r in setting["A"] if r["scenario"] == "S3")
        assert len(s3["runs"]) == data["parameters"]["seeds_per_row"]
        # F1 and F2 are checked against their own population targets.
        naive = s3["summary"]["naive2_Ygx"]["mean"]
        total = mean(r["naive2_est"]["dR"] for r in s3["runs"])
        assert 0.9 < naive / total < 1.1
        for scale in setting["n_scaling_eps_zero"]:
            close(scale["union3_raw_pi"]["mean"], mean(scale["runs"]))
            assert scale["union3_raw_pi"]["bias"] < -0.02


def check_g2():
    data = load("merged_expG2.json")
    assert len(data["rows"]) == 3
    assert all(len(row["runs"]) == data["parameters"]["seeds_per_row"] for row in data["rows"])
    for row in data["rows"]:
        for run in row["runs"]:
            close(run["retro_truth"], run["x_endogenous_loss_truth"] +
                  run["policy_on_current_x_truth"])
        for key in ("retro", "x_endogenous_mean", "x_exogenous_mean",
                    "x_endogenous_loss", "x_exogenous_loss"):
            bias = mean(r[key + "_est"] - r[key + "_truth"] for r in row["runs"])
            close(row["summary"][key]["bias_mean"], bias)


if __name__ == "__main__":
    check_h()
    check_ab()
    check_g2()
    print("Saved E3-E5 evidence and mirrored files validated")
