"""E6 stand-in: semi-synthetic cohort from the open PhysioNet/CinC Challenge 2019 data.

Until credentialed MIMIC-IV access is available, this builds an E6 cohort from
openly licensed ICU data (CC BY 4.0; Reyna et al., Crit Care Med 2020): 40,336
ICU stays from two hospital systems (training sets A and B), hourly vitals and
labs. The cohort has the format written by e6_mimic_cohort.py, so
exp_E6_mimic.py runs on it with the two hospitals in place of the two periods:
the score is fitted at hospital A and the real shift is from A to B.

    python3 sim2/e6_sepsis_cohort.py --root data/challenge-2019 --out data/e6_sepsis_cohort.npz

The files (about 300 MB, one per stay) are fetched from PhysioNet's open S3
mirror on first use and checked against their S3 checksums.

Task. One prediction time per ICU stay, chosen at random (deterministic hash)
among hours at least 4 h into the stay and before sepsis onset. Features: age,
sex, ICU type (medical or surgical), hours in the ICU, hours in hospital before
ICU admission, and the last value in the preceding 24 h of heart rate,
systolic and mean arterial pressure, respiratory rate, temperature, oxygen
saturation, creatinine, urea nitrogen, potassium, chloride, bicarbonate,
glucose, white cells, haemoglobin, platelets and lactate. Outcome: sepsis
onset within 12 h, where onset is 6 h after the first positive SepsisLabel
(the Challenge labels hours from 6 h before Sepsis-3 onset). Stays labelled
positive from their first hour are excluded, since their onset time is not
known.

Differences from the MIMIC-IV task: ICU rather than ward patients, sepsis
onset rather than ICU admission or death, and a shift between hospitals rather
than between periods.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import numpy as np

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training/"
SOURCE = "PhysioNet/CinC Challenge 2019 (open, CC BY 4.0)"
VARS = (("hr", "HR"), ("sbp", "SBP"), ("map", "MAP"), ("resp", "Resp"), ("temp", "Temp"), ("o2sat", "O2Sat"),
        ("creatinine", "Creatinine"), ("urea_nitrogen", "BUN"), ("potassium", "Potassium"),
        ("chloride", "Chloride"), ("bicarbonate", "HCO3"), ("glucose", "Glucose"), ("wbc", "WBC"),
        ("hemoglobin", "Hgb"), ("platelets", "Platelets"), ("lactate", "Lactate"))
HORIZON_HOURS = 12
LOOKBACK_HOURS = 24
MIN_HOURS = 4


# ----------------------------------------------------------------------------- download
def get(url):
    """HTTPS GET through curl, which uses the system certificate store."""
    return subprocess.run(["curl", "--fail", "--silent", "--show-error", url], check=True, capture_output=True).stdout


def list_keys():
    """(key, md5, size) of every training file in the open S3 mirror."""
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={PREFIX}" + (f"&continuation-token={quote(token)}" if token else "")
        page = ET.fromstring(get(url))
        for c in page.findall("s3:Contents", ns):
            key = c.find("s3:Key", ns).text
            if key.endswith(".psv"):
                keys.append((key, c.find("s3:ETag", ns).text.strip('"'), int(c.find("s3:Size", ns).text)))
        if page.find("s3:IsTruncated", ns).text != "true":
            return keys
        token = page.find("s3:NextContinuationToken", ns).text


def intact(path, md5, size):
    return path.exists() and path.stat().st_size == size and hashlib.md5(path.read_bytes()).hexdigest() == md5


def fetch(root, parallel=16):
    """Download missing or damaged files with parallel curl; single-part S3 ETags are MD5 sums."""
    keys = list_keys()
    todo = [(k, m, n) for k, m, n in keys if not intact(Path(root) / k[len(PREFIX):], m, n)]
    if todo:
        with tempfile.NamedTemporaryFile("w", suffix=".curl", delete=False) as cfg:
            for key, _, _ in todo:
                dest = (Path(root) / key[len(PREFIX):]).resolve().as_posix().replace('"', '\\"')
                cfg.write(f'url = "{BUCKET}/{quote(key)}"\noutput = "{dest}"\n')
        try:
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--create-dirs", "--parallel",
                            "--parallel-max", str(parallel), "--retry", "3", "-K", cfg.name], check=True)
        finally:
            Path(cfg.name).unlink()
    bad = [k for k, m, n in todo if not intact(Path(root) / k[len(PREFIX):], m, n)]
    if bad:
        raise SystemExit(f"{len(bad)} files failed the checksum check, e.g. {bad[0]}; rerun to fetch them again")
    print(f"{len(keys):,} files checked against S3 checksums, {len(todo):,} downloaded", flush=True)
    return len(keys)


# ----------------------------------------------------------------------------- cohort
def build(root, seed=20260929):
    import duckdb
    con = duckdb.connect()
    files = (Path(root).resolve() / "training_set*" / "*.psv").as_posix().replace("'", "''")
    used = sorted({c for _, c in VARS} | {"Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS", "SepsisLabel"})
    casts = ", ".join(f'TRY_CAST("{c}" AS DOUBLE) AS "{c}"' for c in used)
    con.execute(f"""
        CREATE TABLE raw AS
        SELECT regexp_extract(filename, 'training_set([AB])', 1) AS site,
               CAST(regexp_extract(filename, 'p([0-9]+)[.]psv', 1) AS BIGINT) AS pid, {casts}
        FROM read_csv('{files}', delim='|', header=true, all_varchar=true, nullstr='NaN', filename=true,
                      union_by_name=true)""")
    con.execute("""
        CREATE TABLE stay AS
        SELECT pid, any_value(site) AS site, min("ICULOS") AS first_hour,
               min("ICULOS") FILTER (WHERE "SepsisLabel" = 1) AS first_positive,
               any_value("Age") AS age, any_value("Gender") AS gender, max("Unit1") AS unit1,
               max("Unit2") AS unit2, any_value("HospAdmTime") AS hosp_adm
        FROM raw GROUP BY pid""")
    con.execute(f"""
        CREATE TABLE cand AS
        SELECT r.pid, r."ICULOS" AS t FROM raw r JOIN stay s USING (pid)
        WHERE (s.first_positive IS NULL OR s.first_positive > s.first_hour)
          AND r."ICULOS" >= {MIN_HOURS} AND (s.first_positive IS NULL OR r."ICULOS" < s.first_positive + 6)""")
    con.execute(f"""
        CREATE TABLE pick AS
        SELECT pid, t FROM (
            SELECT *, row_number() OVER (PARTITION BY pid
                                         ORDER BY hash(concat(CAST(pid AS VARCHAR), '|', CAST(t AS VARCHAR), '|{seed}'))) AS rn
            FROM cand) WHERE rn = 1""")
    last = ", ".join(f'arg_max(r."{c}", r."ICULOS") FILTER (WHERE r."{c}" IS NOT NULL) AS {name}' for name, c in VARS)
    rows = con.execute(f"""
        WITH feats AS (
            SELECT p.pid, {last}
            FROM pick p JOIN raw r ON r.pid = p.pid AND r."ICULOS" > p.t - {LOOKBACK_HOURS} AND r."ICULOS" <= p.t
            GROUP BY p.pid)
        SELECT p.pid, s.site, p.t, s.age, s.gender, s.unit1, s.unit2, s.hosp_adm,
               CAST(s.first_positive IS NOT NULL AND s.first_positive + 6 > p.t
                    AND s.first_positive + 6 <= p.t + {HORIZON_HOURS} AS INTEGER) AS y,
               {', '.join(f'f.{name}' for name, _ in VARS)}
        FROM pick p JOIN stay s USING (pid) LEFT JOIN feats f USING (pid)
        ORDER BY p.pid""").fetchnumpy()
    n_stays = con.execute("SELECT count(*) FROM stay").fetchone()[0]
    n_excluded = con.execute("SELECT count(*) FROM stay WHERE first_positive = first_hour").fetchone()[0]
    f = lambda k: np.ma.filled(np.ma.asarray(rows[k], dtype=float), np.nan)     # DuckDB returns NULLs masked
    unit1, unit2 = f("unit1"), f("unit2")
    known_unit = np.isfinite(unit1) | np.isfinite(unit2)
    demo = {"age": f("age"), "male": f("gender"),
            "unit_micu": np.where(known_unit, (unit1 == 1).astype(float), np.nan),
            "unit_sicu": np.where(known_unit, (unit2 == 1).astype(float), np.nan),
            "hours_in_icu": f("t"), "hours_before_icu": np.clip(-f("hosp_adm"), 0, None)}
    names = list(demo) + [name for name, _ in VARS]
    X = np.column_stack([demo[k] for k in demo] + [f(name) for name, _ in VARS])
    site = np.asarray(rows["site"]).astype(str)
    careunit = np.where(unit1 == 1, "MICU", np.where(unit2 == 1, "SICU", "unknown"))
    meta = {"source": SOURCE, "seed": seed, "horizon_hours": HORIZON_HOURS, "lookback_hours": LOOKBACK_HOURS,
            "min_hours_in_icu": MIN_HOURS, "stays": int(n_stays), "excluded_positive_from_first_hour": int(n_excluded),
            "outcome": f"sepsis onset (Challenge 2019 labels, Sepsis-3) within {HORIZON_HOURS} h",
            "period_labels": {"A": "hospital system A", "B": "hospital system B"},
            "real_components": ["covariates", f"outcome under historical care (sepsis onset within {HORIZON_HOURS} h)",
                                "hospital system (A before deployment, B after)"],
            "scenario_labels": {"deploy_only": "deployment only\n(hospital A)",
                                "drift_and_deploy": "hospital shift\nand deployment"}}
    pid = np.asarray(rows["pid"], np.int64)
    return {"X_raw": X, "feature_names": np.array(names), "y": np.asarray(rows["y"], int), "period": site,
            "careunit": careunit, "subject_id": pid, "hadm_id": pid, "t": np.asarray(rows["t"]).astype(int).astype(str),
            "meta_json": np.array(json.dumps(meta)), "synthetic_fixture": np.array(False)}


def describe(c):
    y, site = c["y"], c["period"]
    meta = json.loads(str(c["meta_json"]))
    print(f"{len(y):,} ICU stays ({meta['stays']:,} in the data, {meta['excluded_positive_from_first_hour']:,} excluded "
          f"as positive from the first hour), event rate {100 * y.mean():.2f}%")
    for s in np.unique(site):
        m = site == s
        print(f"  hospital {s}: {m.sum():,} stays, event rate {100 * y[m].mean():.2f}%")
    for name, miss in zip(c["feature_names"], np.isnan(c["X_raw"]).mean(0)):
        if miss > 0:
            print(f"  missing {name}: {100 * miss:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/challenge-2019", help="local copy of the training sets")
    parser.add_argument("--out", default="data/e6_sepsis_cohort.npz")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--no-fetch", action="store_true", help="use the local files as they are")
    args = parser.parse_args()
    if not args.no_fetch:
        fetch(args.root)
    cohort = build(args.root, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cohort)
    print(f"wrote {out}")
    describe(cohort)


if __name__ == "__main__":
    sys.exit(main())
