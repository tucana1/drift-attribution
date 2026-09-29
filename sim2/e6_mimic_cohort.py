"""E6 step 1: ward-deterioration cohort from a local MIMIC-IV installation.

Run this only on a machine where you hold credentialed PhysioNet access to
MIMIC-IV. Do not upload the data or the cohort file to online services
(including AI assistants); the PhysioNet credentialed data use agreement does
not allow it. The cohort file stays outside the repository (data/ is ignored).

    python3 sim2/e6_mimic_cohort.py --mimic-root /path/to/mimic-iv/3.1 --out data/e6_cohort.npz

Task (matches the clinical task paragraph). Adult ward inpatients. One
prediction time per admission, chosen at random (deterministic hash) among
creatinine draws taken on a ward, at least four hours after admission and
before discharge. Features are age, sex, emergency admission, hours since
admission, a prior ICU stay in the same admission, and the last value in the
preceding 24 hours of eleven routine blood tests (with missing indicators).
Outcome: ICU admission (icu/icustays) or in-hospital death within 12 hours of
the prediction time. Period: the patient's anchor_year_group.

Requirements: duckdb (see requirements-e6.txt). Tables read: hosp/patients,
hosp/admissions, hosp/transfers, hosp/labevents, hosp/d_labitems,
icu/icustays (csv or csv.gz).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Blood tests: label in hosp.d_labitems and the MIMIC-IV itemid used by default.
LABS = (
    ("creatinine", "Creatinine", 50912),
    ("urea_nitrogen", "Urea Nitrogen", 51006),
    ("sodium", "Sodium", 50983),
    ("potassium", "Potassium", 50971),
    ("chloride", "Chloride", 50902),
    ("bicarbonate", "Bicarbonate", 50882),
    ("glucose", "Glucose", 50931),
    ("anion_gap", "Anion Gap", 50868),
    ("wbc", "White Blood Cells", 51301),
    ("hemoglobin", "Hemoglobin", 51222),
    ("platelets", "Platelet Count", 51265),
)
EXCLUDED_UNITS = ("intensive care", "icu", "ccu", "emergency", "unknown", "discharge lounge", "pacu",
                  "labor", "obstetric", "newborn", "nursery", "psychiatr")
HORIZON_HOURS = 12
LOOKBACK_HOURS = 24


def table(root, module, name):
    for ext in (".csv.gz", ".csv"):
        path = Path(root) / module / f"{name}{ext}"
        if path.exists():
            return f"read_csv_auto('{path.as_posix()}', header=true, sample_size=-1)"
    raise FileNotFoundError(f"{module}/{name}.csv(.gz) not found under {root}")


def resolve_itemids(con, root):
    """Check the default itemids against d_labitems; fall back to the label (blood fluid)."""
    rows = con.execute(f"SELECT itemid, label, fluid FROM {table(root, 'hosp', 'd_labitems')}").fetchall()
    by_id = {int(r[0]): (r[1], r[2]) for r in rows}
    ids = {}
    for key, label, default in LABS:
        if default in by_id and by_id[default][0].strip().lower() == label.lower():
            ids[key] = default
            continue
        matches = [i for i, (lab, fluid) in by_id.items()
                   if lab and lab.strip().lower() == label.lower() and (fluid or "").lower() == "blood"]
        if not matches:
            raise ValueError(f"no blood itemid labelled {label!r} in d_labitems")
        ids[key] = min(matches)
        print(f"note: {label} resolved to itemid {ids[key]} (default {default} not found)", file=sys.stderr)
    return ids


def build(root, seed=20260929, min_hours_after_admission=4.0):
    import duckdb

    con = duckdb.connect()
    ids = resolve_itemids(con, root)
    id_list = ",".join(str(v) for v in ids.values())
    excluded = "|".join(EXCLUDED_UNITS)
    con.execute(f"CREATE TABLE icu AS SELECT subject_id, hadm_id, intime, first_careunit, last_careunit "
                f"FROM {table(root, 'icu', 'icustays')}")
    con.execute(f"""
        CREATE TABLE ward AS
        SELECT subject_id, hadm_id, intime, outtime, careunit
        FROM {table(root, 'hosp', 'transfers')}
        WHERE eventtype IN ('admit', 'transfer') AND hadm_id IS NOT NULL AND outtime IS NOT NULL
          AND careunit IS NOT NULL
          AND NOT regexp_matches(lower(careunit), '({excluded})')
          AND careunit NOT IN (SELECT first_careunit FROM icu WHERE first_careunit IS NOT NULL
                               UNION SELECT last_careunit FROM icu WHERE last_careunit IS NOT NULL)""")
    con.execute(f"""
        CREATE TABLE labs AS
        SELECT subject_id, itemid, charttime, valuenum
        FROM {table(root, 'hosp', 'labevents')}
        WHERE itemid IN ({id_list}) AND valuenum IS NOT NULL AND charttime IS NOT NULL""")
    con.execute(f"""
        CREATE TABLE adm AS
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.deathtime, a.admission_type,
               p.gender, p.anchor_age + (year(a.admittime) - p.anchor_year) AS age, p.anchor_year_group
        FROM {table(root, 'hosp', 'admissions')} a JOIN {table(root, 'hosp', 'patients')} p USING (subject_id)""")
    con.execute(f"""
        CREATE TABLE cand AS
        SELECT DISTINCT l.subject_id, w.hadm_id, l.charttime AS t
        FROM labs l
        JOIN ward w ON l.subject_id = w.subject_id AND l.charttime >= w.intime AND l.charttime < w.outtime
        JOIN adm a ON a.hadm_id = w.hadm_id
        WHERE l.itemid = {ids['creatinine']} AND a.age >= 18
          AND l.charttime >= a.admittime + INTERVAL {int(min_hours_after_admission * 60)} MINUTE
          AND l.charttime < a.dischtime""")
    con.execute(f"""
        CREATE TABLE pick AS
        SELECT subject_id, hadm_id, t FROM (
            SELECT *, row_number() OVER (PARTITION BY hadm_id
                                         ORDER BY hash(concat(CAST(hadm_id AS VARCHAR), '|', CAST(t AS VARCHAR), '|{seed}'))) AS rn
            FROM cand) WHERE rn = 1""")
    base = con.execute(f"""
        SELECT p.subject_id, p.hadm_id, CAST(p.t AS VARCHAR) AS t, a.age,
               CAST(a.gender = 'M' AS INTEGER) AS male,
               CAST(regexp_matches(upper(a.admission_type), 'EMER|URGENT') AS INTEGER) AS emergency,
               date_diff('minute', a.admittime, p.t) / 60.0 AS hours_since_admission,
               CAST(EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = p.hadm_id AND i.intime < p.t) AS INTEGER) AS prior_icu,
               CAST(EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = p.hadm_id AND i.intime > p.t
                                          AND i.intime <= p.t + INTERVAL {HORIZON_HOURS} HOUR)
                    OR (a.deathtime IS NOT NULL AND a.deathtime > p.t
                        AND a.deathtime <= p.t + INTERVAL {HORIZON_HOURS} HOUR) AS INTEGER) AS y,
               a.anchor_year_group AS period
        FROM pick p JOIN adm a ON a.hadm_id = p.hadm_id
        ORDER BY p.hadm_id""").fetchnumpy()
    last = con.execute(f"""
        SELECT p.hadm_id, l.itemid, arg_max(l.valuenum, l.charttime) AS v
        FROM pick p JOIN labs l ON l.subject_id = p.subject_id
             AND l.charttime > p.t - INTERVAL {LOOKBACK_HOURS} HOUR AND l.charttime <= p.t
        GROUP BY p.hadm_id, l.itemid""").fetchnumpy()
    hadm = np.asarray(base["hadm_id"])
    row = {h: i for i, h in enumerate(hadm.tolist())}
    col = {itemid: j for j, itemid in enumerate(ids.values())}
    labs = np.full((len(hadm), len(ids)), np.nan)
    for h, itemid, v in zip(last["hadm_id"], last["itemid"], last["v"]):
        labs[row[int(h)], col[int(itemid)]] = float(v)
    demo = np.column_stack([np.asarray(base[k], float) for k in
                            ("age", "male", "emergency", "hours_since_admission", "prior_icu")])
    names = ["age", "male", "emergency", "hours_since_admission", "prior_icu"] + [k for k, _, _ in LABS]
    return {"X_raw": np.column_stack([demo, labs]), "feature_names": np.array(names),
            "y": np.asarray(base["y"], int), "period": np.asarray(base["period"]).astype(str),
            "subject_id": np.asarray(base["subject_id"], np.int64), "hadm_id": hadm.astype(np.int64),
            "t": np.asarray(base["t"]).astype(str), "itemids": np.array(list(ids.values())),
            "synthetic_fixture": np.array((Path(root) / "FIXTURE_NOT_MIMIC").exists())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimic-root", required=True, help="directory containing hosp/ and icu/")
    parser.add_argument("--out", default="data/e6_cohort.npz")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    cohort = build(args.mimic_root, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cohort)
    periods, counts = np.unique(cohort["period"], return_counts=True)
    print(f"wrote {out}: {len(cohort['y'])} admissions, event rate {cohort['y'].mean():.4f}")
    for p, c in zip(periods, counts):
        print(f"  {p}: {c} admissions, event rate {cohort['y'][cohort['period'] == p].mean():.4f}")
    missing = np.isnan(cohort["X_raw"]).mean(0)
    for name, m in zip(cohort["feature_names"], missing):
        if m > 0:
            print(f"  missing {name}: {100 * m:.1f}%")


if __name__ == "__main__":
    main()
