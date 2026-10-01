"""E6 step 1: ward-deterioration cohort from a local MIMIC-IV installation.

Run this on the machine that holds credentialed MIMIC-IV. Do not upload the
tables, the cohort file or the error log to online services, including AI
assistants (PhysioNet credentialed data use agreement). The script prints
counts and rates only. If extraction fails, the error message, which can
quote a CSV row, goes to a log next to --out instead of the terminal;
--show-errors prints it, for the open demo or the synthetic fixture only.

    python3 sim2/e6_mimic_cohort.py --mimic-root /path/to/mimiciv/3.1 --out data/e6_cohort.npz

Task (matches the clinical task paragraph). Adult ward inpatients. One
prediction time per admission, chosen at random (deterministic hash) among
creatinine draws taken on a ward, at least four hours after admission and
before discharge. Features are age, sex, emergency admission, hours since
admission, a prior ICU stay in the same admission, and the last value in the
preceding 24 hours of eleven routine blood tests (with missing indicators).
Outcome: ICU admission (icu/icustays) or in-hospital death within 12 hours of
the prediction time. Period: the patient's anchor_year_group; the cohort also
stores year(admittime) - anchor_year, so the experiment can place admissions
by approximate admission year instead.

Tables are read as text and the columns used are converted with TRY_CAST, so
an unexpected value becomes NULL rather than an error. Intermediate tables
live in an on-disk DuckDB file next to --out (deleted at the end), with a
memory limit, so labevents does not have to fit in RAM.

After writing the cohort the script checks the acceptance criteria of
docs/NEXT_STEPS.md (size, event rate, period coverage, lab missingness) and
exits with status 3 if any fails.

Requirements: duckdb (requirements-e6.txt). Tables read: hosp/patients,
hosp/admissions, hosp/transfers, hosp/labevents, hosp/d_labitems,
icu/icustays (csv or csv.gz).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
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
EXPECTED_GROUPS = ("2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019")
SMALL_CELL = 10          # counts below this are printed as "<10"

# Columns used from each table and their types (present in MIMIC-IV v2.2 and v3.x).
TYPES = {
    "patients": {"subject_id": "BIGINT", "gender": "VARCHAR", "anchor_age": "INTEGER", "anchor_year": "INTEGER",
                 "anchor_year_group": "VARCHAR"},
    "admissions": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "admittime": "TIMESTAMP", "dischtime": "TIMESTAMP",
                   "deathtime": "TIMESTAMP", "admission_type": "VARCHAR"},
    "transfers": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "eventtype": "VARCHAR", "careunit": "VARCHAR",
                  "intime": "TIMESTAMP", "outtime": "TIMESTAMP"},
    "labevents": {"subject_id": "BIGINT", "itemid": "BIGINT", "charttime": "TIMESTAMP", "valuenum": "DOUBLE"},
    "d_labitems": {"itemid": "BIGINT", "label": "VARCHAR", "fluid": "VARCHAR"},
    "icustays": {"subject_id": "BIGINT", "hadm_id": "BIGINT", "first_careunit": "VARCHAR",
                 "last_careunit": "VARCHAR", "intime": "TIMESTAMP"},
}


def table(root, module, name):
    """SQL subquery over one MIMIC-IV table: all columns read as text, the used ones TRY_CAST."""
    for ext in (".csv.gz", ".csv"):
        path = Path(root) / module / f"{name}{ext}"
        if path.exists():
            cols = ", ".join(f'TRY_CAST("{c}" AS {t}) AS {c}' for c, t in TYPES[name].items())
            src = path.resolve().as_posix().replace("'", "''")
            return (f"(SELECT {cols} FROM read_csv('{src}', header=true, all_varchar=true, "
                    f"delim=',', quote='\"', escape='\"'))")
    raise FileNotFoundError(f"{module}/{name}.csv(.gz) not found under {root}")


class Steps:
    """Runs SQL and remembers which step is running, for the redacted error report."""

    def __init__(self, con):
        self.con, self.current = con, "connect"

    def __call__(self, step, sql):
        self.current = step
        return self.con.execute(sql)


def resolve_itemids(run, root):
    """Check the default itemids against d_labitems; fall back to the label (blood fluid)."""
    rows = run("d_labitems", f"SELECT itemid, label, fluid FROM {table(root, 'hosp', 'd_labitems')}").fetchall()
    by_id = {int(r[0]): (r[1], r[2]) for r in rows if r[0] is not None}
    ids = {}
    for key, label, default in LABS:
        if default in by_id and (by_id[default][0] or "").strip().lower() == label.lower():
            ids[key] = default
            continue
        matches = [i for i, (lab, fluid) in by_id.items()
                   if lab and lab.strip().lower() == label.lower() and (fluid or "").lower() == "blood"]
        if not matches:
            raise ValueError(f"no blood itemid labelled {label!r} in d_labitems")
        ids[key] = min(matches)
        print(f"note: {label} resolved to itemid {ids[key]} (default {default} not found)", file=sys.stderr)
    return ids


def build(root, run, seed=20260929, min_hours_after_admission=4.0):
    ids = resolve_itemids(run, root)
    id_list = ",".join(str(v) for v in ids.values())
    excluded = "|".join(EXCLUDED_UNITS)
    run("icustays", f"CREATE TABLE icu AS SELECT subject_id, hadm_id, intime, first_careunit, last_careunit "
                    f"FROM {table(root, 'icu', 'icustays')}")
    run("transfers", f"""
        CREATE TABLE ward AS
        SELECT subject_id, hadm_id, intime, outtime, careunit
        FROM {table(root, 'hosp', 'transfers')}
        WHERE lower(eventtype) IN ('admit', 'transfer') AND hadm_id IS NOT NULL AND intime IS NOT NULL
          AND outtime IS NOT NULL AND careunit IS NOT NULL
          AND NOT regexp_matches(lower(careunit), '({excluded})')
          AND careunit NOT IN (SELECT first_careunit FROM icu WHERE first_careunit IS NOT NULL
                               UNION SELECT last_careunit FROM icu WHERE last_careunit IS NOT NULL)""")
    run("labevents", f"""
        CREATE TABLE labs AS
        SELECT subject_id, itemid, charttime, valuenum
        FROM {table(root, 'hosp', 'labevents')}
        WHERE itemid IN ({id_list}) AND valuenum IS NOT NULL AND charttime IS NOT NULL
          AND subject_id IS NOT NULL""")
    run("admissions and patients", f"""
        CREATE TABLE adm AS
        SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.deathtime, a.admission_type,
               p.gender, p.anchor_age + (year(a.admittime) - p.anchor_year) AS age,
               year(a.admittime) - p.anchor_year AS years_from_anchor, p.anchor_year_group
        FROM {table(root, 'hosp', 'admissions')} a JOIN {table(root, 'hosp', 'patients')} p USING (subject_id)
        WHERE a.hadm_id IS NOT NULL AND a.admittime IS NOT NULL AND a.dischtime IS NOT NULL""")
    run("candidate times", f"""
        CREATE TABLE cand AS
        SELECT l.subject_id, w.hadm_id, l.charttime AS t, min(w.careunit) AS careunit
        FROM labs l
        JOIN ward w ON l.subject_id = w.subject_id AND l.charttime >= w.intime AND l.charttime < w.outtime
        JOIN adm a ON a.hadm_id = w.hadm_id
        WHERE l.itemid = {ids['creatinine']} AND a.age >= 18
          AND l.charttime >= a.admittime + INTERVAL {int(min_hours_after_admission * 60)} MINUTE
          AND l.charttime < a.dischtime
        GROUP BY l.subject_id, w.hadm_id, l.charttime""")
    run("prediction times", f"""
        CREATE TABLE pick AS
        SELECT subject_id, hadm_id, t, careunit FROM (
            SELECT *, row_number() OVER (PARTITION BY hadm_id
                                         ORDER BY hash(concat(CAST(hadm_id AS VARCHAR), '|', CAST(t AS VARCHAR), '|{seed}'))) AS rn
            FROM cand) WHERE rn = 1""")
    base = run("features and outcome", f"""
        SELECT p.subject_id, p.hadm_id, CAST(p.t AS VARCHAR) AS t, p.careunit, a.age,
               CAST(a.gender = 'M' AS INTEGER) AS male,
               CAST(regexp_matches(upper(coalesce(a.admission_type, '')), 'EMER|URGENT') AS INTEGER) AS emergency,
               date_diff('minute', a.admittime, p.t) / 60.0 AS hours_since_admission,
               CAST(EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = p.hadm_id AND i.intime < p.t) AS INTEGER) AS prior_icu,
               CAST(EXISTS (SELECT 1 FROM icu i WHERE i.hadm_id = p.hadm_id AND i.intime > p.t
                                          AND i.intime <= p.t + INTERVAL {HORIZON_HOURS} HOUR) AS INTEGER) AS y_icu,
               CAST(a.deathtime IS NOT NULL AND a.deathtime > p.t
                    AND a.deathtime <= p.t + INTERVAL {HORIZON_HOURS} HOUR AS INTEGER) AS y_death,
               a.anchor_year_group AS period, a.years_from_anchor
        FROM pick p JOIN adm a ON a.hadm_id = p.hadm_id
        ORDER BY p.hadm_id""").fetchnumpy()
    last = run("lookback labs", f"""
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
    demo_cols = ("age", "male", "emergency", "hours_since_admission", "prior_icu")
    demo = np.column_stack([np.asarray(base[k], float) for k in demo_cols])
    y_icu, y_death = np.asarray(base["y_icu"], int), np.asarray(base["y_death"], int)
    meta = {"seed": seed, "horizon_hours": HORIZON_HOURS, "lookback_hours": LOOKBACK_HOURS,
            "min_hours_after_admission": min_hours_after_admission, "excluded_unit_patterns": list(EXCLUDED_UNITS),
            "itemids": ids, "mimic_root_name": Path(root).resolve().name}
    return {"X_raw": np.column_stack([demo, labs]), "feature_names": np.array(list(demo_cols) + [k for k, _, _ in LABS]),
            "y": np.maximum(y_icu, y_death), "y_icu": y_icu, "y_death": y_death,
            "period": np.asarray(base["period"]).astype(str),
            "years_from_anchor": np.asarray(base["years_from_anchor"], float),
            "careunit": np.asarray(base["careunit"]).astype(str),
            "subject_id": np.asarray(base["subject_id"], np.int64), "hadm_id": hadm.astype(np.int64),
            "t": np.asarray(base["t"]).astype(str), "itemids": np.array(list(ids.values())),
            "meta_json": np.array(json.dumps(meta)),
            "synthetic_fixture": np.array((Path(root) / "FIXTURE_NOT_MIMIC").exists())}


# ----------------------------------------------------------------------------- reporting (aggregates only)
def count(n):
    return f"<{SMALL_CELL}" if 0 < n < SMALL_CELL else f"{n:,}"


def rate(events, n):
    return "suppressed" if n < SMALL_CELL or 0 < events < SMALL_CELL else f"{100 * events / n:.2f}%"


def describe(c):
    y, period = c["y"], c["period"]
    n = len(y)
    print(f"{count(n)} admissions, {count(len(np.unique(c['subject_id'])))} patients, "
          f"event rate {rate(int(y.sum()), n)}")
    print(f"  outcome: ICU admission within {HORIZON_HOURS} h {rate(int(c['y_icu'].sum()), n)}, "
          f"death within {HORIZON_HOURS} h {rate(int(c['y_death'].sum()), n)}")
    for p in np.unique(period):
        m = period == p
        print(f"  {p}: {count(int(m.sum()))} admissions, event rate {rate(int(y[m].sum()), int(m.sum()))}")
    names = list(c["feature_names"])
    x = c["X_raw"]
    j = {k: names.index(k) for k in ("age", "male", "emergency", "hours_since_admission", "prior_icu")}
    print(f"  median age {np.nanmedian(x[:, j['age']]):.0f}, male {100 * np.nanmean(x[:, j['male']]):.1f}%, "
          f"emergency {100 * np.nanmean(x[:, j['emergency']]):.1f}%, prior ICU stay {100 * np.nanmean(x[:, j['prior_icu']]):.1f}%, "
          f"median hours since admission {np.nanmedian(x[:, j['hours_since_admission']]):.1f}")
    years = c["years_from_anchor"]
    print(f"  admissions in the anchor year {100 * np.mean(years == 0):.1f}%, "
          f"more than 3 years from it {100 * np.mean(np.abs(years) > 3):.1f}%")
    for name, m in zip(names, np.isnan(x).mean(0)):
        if m > 0:
            print(f"  missing {name}: {100 * m:.1f}%")
    units, counts = np.unique(c["careunit"], return_counts=True)
    order = np.argsort(-counts)
    shown = [(units[i], counts[i]) for i in order if counts[i] >= SMALL_CELL][:15]
    print("  prediction times by ward unit: " + "; ".join(f"{u} {count(int(k))}" for u, k in shown)
          + (f"; {len(units) - len(shown)} other units" if len(units) > len(shown) else ""))


def checks(c, min_admissions=5000):
    """Acceptance criteria of docs/NEXT_STEPS.md, step 1.4."""
    y, period, names, x = c["y"], set(c["period"].tolist()), list(c["feature_names"]), c["X_raw"]
    out = [("admissions >= 5,000 (expect tens of thousands)", len(y) >= min_admissions, f"{len(y):,}")]
    r = float(y.mean()) if len(y) else float("nan")
    out.append(("event rate within 0.5% to 15% (expect about 1% to 10%)", 0.005 <= r <= 0.15, f"{100 * r:.2f}%"))
    missing = [g for g in EXPECTED_GROUPS if g not in period]
    out.append(("anchor_year_group 2008 - 2010 to 2017 - 2019 all present", not missing,
                "missing " + ", ".join(missing) if missing else "all present"))
    miss = {k: float(np.isnan(x[:, names.index(k)]).mean()) for k, _, _ in LABS}
    worst = max(miss, key=miss.get)
    out.append(("missingness of every lab below 10%", miss[worst] < 0.10, f"highest {worst} {100 * miss[worst]:.1f}%"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mimic-root", required=True, help="directory containing hosp/ and icu/")
    parser.add_argument("--out", default="data/e6_cohort.npz")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--memory-limit", default="4GB", help="DuckDB memory limit; it spills to disk beyond this")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--show-errors", action="store_true",
                        help="print the full error (can quote a data row); for the open demo or the fixture only")
    parser.add_argument("--no-checks", action="store_true", help="exit 0 even if an acceptance check fails")
    args = parser.parse_args()
    import duckdb

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix(".work.duckdb")
    log = out.with_suffix(".error.log")
    for f in (work, Path(f"{work}.wal"), log):
        f.unlink(missing_ok=True)
    con = duckdb.connect(str(work))
    run = Steps(con)
    try:
        run("configure", f"SET memory_limit = '{args.memory_limit}'")
        run("configure", f"SET threads = {int(args.threads)}")
        run("configure", "SET preserve_insertion_order = false")
        tmp = (out.parent / "duckdb_tmp").resolve().as_posix().replace("'", "''")
        run("configure", f"SET temp_directory = '{tmp}'")
        cohort = build(args.mimic_root, run, seed=args.seed)
    except Exception as exc:  # noqa: BLE001  report without quoting data
        if args.show_errors:
            raise
        log.write_text(traceback.format_exc())
        print(f"cohort extraction failed at step '{run.current}' ({type(exc).__name__}). The message is not shown "
              f"because it can quote a MIMIC-IV row; it is in {log}. Keep that file local.", file=sys.stderr)
        sys.exit(1)
    finally:
        con.close()
        for f in (work, Path(f"{work}.wal")):
            f.unlink(missing_ok=True)
        try:
            (out.parent / "duckdb_tmp").rmdir()
        except OSError:
            pass
    np.savez_compressed(out, **cohort)
    print(f"wrote {out}")
    describe(cohort)
    results = checks(cohort)
    for name, ok, value in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value}")
    if not all(ok for _, ok, _ in results) and not args.no_checks:
        print("acceptance checks failed: investigate before running sim2/exp_E6_mimic.py (docs/NEXT_STEPS.md, step 1.4)",
              file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
