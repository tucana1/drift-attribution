"""Synthetic stand-in for MIMIC-IV, for testing the E6 pipeline only.

Writes small gzip CSV tables with the MIMIC-IV column layout (hosp/patients,
hosp/admissions, hosp/transfers, hosp/labevents, hosp/d_labitems,
icu/icustays). Every value is simulated. Nothing produced from this fixture is
a result; it exists so that sim2/e6_mimic_cohort.py and sim2/exp_E6_mimic.py
can be exercised end to end without credentialed data.

    python3 sim2/e6_fixture.py --out /tmp/mimic_fixture --patients 6000
"""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

GROUPS = ("2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019")
# itemid, label, fluid, category, mean, sd, severity loading, unit
LABITEMS = (
    (50912, "Creatinine", "Blood", "Chemistry", 1.1, 0.5, 0.35, "mg/dL"),
    (51006, "Urea Nitrogen", "Blood", "Chemistry", 20.0, 9.0, 6.0, "mg/dL"),
    (50983, "Sodium", "Blood", "Chemistry", 139.0, 3.5, -1.2, "mEq/L"),
    (50971, "Potassium", "Blood", "Chemistry", 4.2, 0.5, 0.15, "mEq/L"),
    (50902, "Chloride", "Blood", "Chemistry", 102.0, 4.0, -0.8, "mEq/L"),
    (50882, "Bicarbonate", "Blood", "Chemistry", 25.0, 3.0, -1.6, "mEq/L"),
    (50931, "Glucose", "Blood", "Chemistry", 125.0, 40.0, 12.0, "mg/dL"),
    (50868, "Anion Gap", "Blood", "Chemistry", 13.0, 3.0, 1.4, "mEq/L"),
    (51301, "White Blood Cells", "Blood", "Hematology", 8.5, 3.5, 2.0, "K/uL"),
    (51222, "Hemoglobin", "Blood", "Hematology", 11.5, 2.0, -0.6, "g/dL"),
    (51265, "Platelet Count", "Blood", "Hematology", 230.0, 80.0, -20.0, "K/uL"),
    (50813, "Lactate", "Blood", "Blood Gas", 1.6, 0.8, 0.5, "mmol/L"),
)
WARDS = ("Medicine", "Med/Surg", "Surgery", "Cardiology", "Hematology/Oncology")
ICUS = ("Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)")
TS = "%Y-%m-%d %H:%M:%S"


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--patients", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    patients, admissions, transfers, labs, icustays = [], [], [], [], []
    lab_id = transfer_id = stay_id = 0
    for i in range(args.patients):
        subject, hadm = 10_000_000 + i, 20_000_000 + i
        g = int(rng.integers(0, len(GROUPS)))
        late = g >= 2
        anchor_year = 2150 + int(rng.integers(0, 40))
        age = int(np.clip(rng.normal(64, 16), 18, 91))
        male = rng.random() < 0.5
        admit = datetime(anchor_year, 1, 1) + timedelta(minutes=int(rng.integers(0, 525_000)))
        severity = rng.normal(0.25 * late, 1.0)
        emergency = rng.random() < 0.7
        ed_hours = rng.uniform(2, 8)
        los_hours = float(np.clip(rng.gamma(2.0, 48.0), 24, 600))
        ward_in = admit + timedelta(hours=ed_hours)
        disch = admit + timedelta(hours=los_hours)
        # deterioration hazard per hour on the ward (fake); later periods escalate earlier
        hazard = 0.004 * np.exp(0.9 * severity + 0.01 * (age - 64) + (0.35 if late else 0.0))
        t_event = ward_in + timedelta(hours=float(rng.exponential(1 / hazard)))
        deterioration = t_event < disch
        died = deterioration and rng.random() < 0.15
        ward = WARDS[int(rng.integers(0, len(WARDS)))]
        transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "ED", "Emergency Department",
                          admit.strftime(TS), ward_in.strftime(TS)])
        deathtime = ""
        if deterioration and not died:
            icu_out = t_event + timedelta(hours=float(rng.uniform(24, 96)))
            icu_unit = ICUS[int(rng.integers(0, len(ICUS)))]
            transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "admit", ward, ward_in.strftime(TS), t_event.strftime(TS)])
            transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "transfer", icu_unit, t_event.strftime(TS), icu_out.strftime(TS)])
            disch = max(disch, icu_out + timedelta(hours=24))
            transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "transfer", ward, icu_out.strftime(TS), disch.strftime(TS)])
            icustays.append([subject, hadm, 30_000_000 + (stay_id := stay_id + 1), icu_unit, icu_unit,
                             t_event.strftime(TS), icu_out.strftime(TS), round((icu_out - t_event).total_seconds() / 86400, 4)])
            ward_end = disch
        elif died:
            disch = t_event
            deathtime = t_event.strftime(TS)
            transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "admit", ward, ward_in.strftime(TS), disch.strftime(TS)])
            ward_end = disch
        else:
            transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "admit", ward, ward_in.strftime(TS), disch.strftime(TS)])
            ward_end = disch
        transfers.append([subject, hadm, (transfer_id := transfer_id + 1), "discharge", "", disch.strftime(TS), ""])
        # labs roughly every 12 h from ED arrival to discharge, severity drifting upward before deterioration
        t = admit + timedelta(hours=float(rng.uniform(0.5, 2)))
        specimen = 0
        while t < ward_end:
            specimen += 1
            ramp = 1.5 * max(0.0, 1 - (t_event - t).total_seconds() / 86400) if deterioration else 0.0
            for itemid, label, fluid, cat, mean, sd, load, unit in LABITEMS:
                if itemid == 50813 and rng.random() < 0.8:
                    continue          # lactate is rarely drawn on wards
                if rng.random() < 0.05:
                    continue
                value = max(0.05, mean + load * (severity + ramp) + sd * 0.6 * rng.normal())
                labs.append([(lab_id := lab_id + 1), subject, hadm if rng.random() > 0.02 else "", 40_000_000 + specimen,
                             itemid, "", t.strftime(TS), (t + timedelta(hours=1)).strftime(TS), f"{value:.2f}",
                             round(value, 2), unit, "", "", "", "ROUTINE", ""])
            t += timedelta(hours=float(rng.uniform(8, 16)))
        patients.append([subject, "M" if male else "F", age, anchor_year, GROUPS[g], ""])
        admissions.append([subject, hadm, admit.strftime(TS), disch.strftime(TS), deathtime,
                           "EW EMER." if emergency else "ELECTIVE", "", "EMERGENCY ROOM" if emergency else "PHYSICIAN REFERRAL",
                           "DIED" if died else "HOME", "Medicare", "ENGLISH", "", "WHITE",
                           admit.strftime(TS), ward_in.strftime(TS), int(died)])
    write(root / "hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"], patients)
    write(root / "hosp/admissions.csv.gz",
          ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id",
           "admission_location", "discharge_location", "insurance", "language", "marital_status", "race",
           "edregtime", "edouttime", "hospital_expire_flag"], admissions)
    write(root / "hosp/transfers.csv.gz", ["subject_id", "hadm_id", "transfer_id", "eventtype", "careunit", "intime", "outtime"], transfers)
    write(root / "hosp/labevents.csv.gz",
          ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime", "storetime",
           "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"], labs)
    write(root / "hosp/d_labitems.csv.gz", ["itemid", "label", "fluid", "category"],
          [[i, lab, fl, cat] for i, lab, fl, cat, *_ in LABITEMS])
    write(root / "icu/icustays.csv.gz",
          ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"], icustays)
    (root / "FIXTURE_NOT_MIMIC").write_text("Synthetic test fixture written by sim2/e6_fixture.py. Not MIMIC-IV data.\n")
    print(f"wrote synthetic MIMIC-IV-shaped fixture to {root} ({args.patients} patients, {len(labs)} lab rows)")


if __name__ == "__main__":
    main()
