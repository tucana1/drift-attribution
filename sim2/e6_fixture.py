"""Synthetic stand-in for MIMIC-IV, for testing the E6 pipeline only.

Writes small gzip CSV tables with the MIMIC-IV column layout (hosp/patients,
hosp/admissions, hosp/transfers, hosp/labevents, hosp/d_labitems,
icu/icustays). Every value is simulated. Nothing produced from this fixture is
a result; it exists so that sim2/e6_mimic_cohort.py and sim2/exp_E6_mimic.py
can be exercised end to end without credentialed data.

The layout follows MIMIC-IV v2.2/v3.x where it matters for the extraction:
real care-unit names (wards, ICUs, ED, observation, PACU, obstetric and
psychiatric units), ED visits without an admission, discharge rows without a
unit, repeat admissions years after the anchor year, a 2020 - 2022 anchor
group, MIMIC-IV admission types, lab rows without hadm_id or with a text value
and no valuenum, quoted comments containing commas, quotes and line breaks,
and a urine creatinine item that shares the blood item's label.

    python3 sim2/e6_fixture.py --out /tmp/mimic_fixture --patients 6000
"""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

GROUPS = ("2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022")
GROUP_P = (0.22, 0.22, 0.22, 0.22, 0.12)
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
    (51082, "Creatinine", "Urine", "Chemistry", 80.0, 40.0, 5.0, "mg/dL"),
)
WARDS = ("Medicine", "Med/Surg", "Surgery", "Cardiology", "Hematology/Oncology", "Medicine/Cardiology",
         "Neuro Stepdown", "Transplant", "Observation")
ICUS = ("Medical Intensive Care Unit (MICU)", "Surgical Intensive Care Unit (SICU)", "Coronary Care Unit (CCU)",
        "Cardiac Vascular Intensive Care Unit (CVICU)")
NON_WARD = ("PACU", "Labor & Delivery", "Psychiatry", "Emergency Department Observation", "Unknown", "Discharge Lounge")
ADMISSION_TYPES = (("EW EMER.", 0.45), ("URGENT", 0.12), ("DIRECT EMER.", 0.05), ("ELECTIVE", 0.12),
                   ("OBSERVATION ADMIT", 0.1), ("SURGICAL SAME DAY ADMISSION", 0.08), ("EU OBSERVATION", 0.04),
                   ("DIRECT OBSERVATION", 0.02), ("AMBULATORY OBSERVATION", 0.02))
COMMENTS = ("", "", "", "___", 'VERIFIED BY REPLICATE ANALYSIS, "HEMOLYZED"', "SPECIMEN, RECEIVED LATE\nRESULT CONFIRMED")
TS = "%Y-%m-%d %H:%M:%S"


def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


class Ids:
    def __init__(self):
        self.lab = self.transfer = self.stay = self.specimen = 0
        self.hadm = 20_000_000

    def next(self, name):
        setattr(self, name, getattr(self, name) + 1)
        return getattr(self, name)


def admission(rng, ids, subject, admit, severity, late, age, rows):
    """One hospital admission with transfers, labs and possibly an ICU stay or death."""
    transfers, labs, icustays, admissions = rows
    hadm = ids.next("hadm")
    kinds, probs = zip(*ADMISSION_TYPES)
    admission_type = kinds[int(rng.choice(len(kinds), p=np.array(probs) / sum(probs)))]
    emergency = "EMER" in admission_type or admission_type == "URGENT"
    surgical = "SURGICAL" in admission_type
    ed_hours = rng.uniform(2, 8) if emergency else rng.uniform(1, 3) if surgical else 0.0
    los_hours = float(np.clip(rng.gamma(2.0, 48.0), 24, 600))
    ward_in = admit + timedelta(hours=ed_hours)
    disch = admit + timedelta(hours=los_hours)
    # deterioration hazard per hour on the ward (fake); later periods escalate earlier
    hazard = 0.004 * np.exp(0.9 * severity + 0.01 * (age - 64) + (0.35 if late else 0.0))
    t_event = ward_in + timedelta(hours=float(rng.exponential(1 / hazard)))
    deterioration = t_event < disch
    died = deterioration and rng.random() < 0.15
    ward = WARDS[int(rng.integers(0, len(WARDS)))] if rng.random() > 0.05 else NON_WARD[int(rng.integers(0, len(NON_WARD)))]
    # as in MIMIC-IV: an ED row ('ED'), then 'admit' to the first inpatient unit, 'transfer' for later moves
    ward_event = "admit"
    if emergency:
        transfers.append([subject, hadm, ids.next("transfer"), "ED", "Emergency Department", admit.strftime(TS), ward_in.strftime(TS)])
    elif surgical:
        transfers.append([subject, hadm, ids.next("transfer"), "admit", "PACU", admit.strftime(TS), ward_in.strftime(TS)])
        ward_event = "transfer"
    deathtime = ""
    if deterioration and not died:
        icu_out = t_event + timedelta(hours=float(rng.uniform(24, 96)))
        icu_unit = ICUS[int(rng.integers(0, len(ICUS)))]
        transfers.append([subject, hadm, ids.next("transfer"), ward_event, ward, ward_in.strftime(TS), t_event.strftime(TS)])
        transfers.append([subject, hadm, ids.next("transfer"), "transfer", icu_unit, t_event.strftime(TS), icu_out.strftime(TS)])
        disch = max(disch, icu_out + timedelta(hours=24))
        transfers.append([subject, hadm, ids.next("transfer"), "transfer", ward, icu_out.strftime(TS), disch.strftime(TS)])
        icustays.append([subject, hadm, 30_000_000 + ids.next("stay"), icu_unit, icu_unit,
                         t_event.strftime(TS), icu_out.strftime(TS), round((icu_out - t_event).total_seconds() / 86400, 4)])
    else:
        if died:
            disch = t_event
            deathtime = t_event.strftime(TS)
        transfers.append([subject, hadm, ids.next("transfer"), ward_event, ward, ward_in.strftime(TS), disch.strftime(TS)])
    transfers.append([subject, hadm, ids.next("transfer"), "discharge", "", disch.strftime(TS), ""])
    # labs roughly every 12 h from arrival to discharge, severity drifting upward before deterioration
    t = admit + timedelta(hours=float(rng.uniform(0.5, 2)))
    while t < disch:
        specimen = 40_000_000 + ids.next("specimen")
        ramp = 1.5 * max(0.0, 1 - (t_event - t).total_seconds() / 86400) if deterioration else 0.0
        for itemid, label, fluid, cat, mean, sd, load, unit in LABITEMS:
            if itemid in (50813, 51082) and rng.random() < 0.8:
                continue          # lactate and urine creatinine are rarely drawn on wards
            if rng.random() < 0.05:
                continue
            value = max(0.05, mean + load * (severity + ramp) + sd * 0.6 * rng.normal())
            text, num = f"{value:.2f}", round(value, 2)
            if rng.random() < 0.003:
                text, num = ("___", "") if rng.random() < 0.5 else ("NEG", "")
            labs.append([ids.next("lab"), subject, hadm if rng.random() > 0.02 else "", specimen, itemid,
                         "P" + str(int(rng.integers(10_000, 99_999))), t.strftime(TS), (t + timedelta(hours=1)).strftime(TS),
                         text, num, unit, "", "", "abnormal" if rng.random() < 0.2 else "",
                         "ROUTINE" if rng.random() < 0.9 else "STAT", COMMENTS[int(rng.integers(0, len(COMMENTS)))]])
        t += timedelta(hours=float(rng.uniform(8, 16)))
    admissions.append([subject, hadm, admit.strftime(TS), disch.strftime(TS), deathtime, admission_type,
                       "P" + str(int(rng.integers(10_000, 99_999))),
                       "EMERGENCY ROOM" if emergency else "PHYSICIAN REFERRAL",
                       "DIED" if died else "HOME", "Medicare", "ENGLISH", "", "WHITE",
                       admit.strftime(TS) if emergency else "", ward_in.strftime(TS) if emergency else "", int(died)])
    return disch, died


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--patients", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    patients, admissions, transfers, labs, icustays = [], [], [], [], []
    rows = (transfers, labs, icustays, admissions)
    ids = Ids()
    for i in range(args.patients):
        subject = 10_000_000 + i
        g = int(rng.choice(len(GROUPS), p=GROUP_P))
        anchor_year = 2150 + int(rng.integers(0, 40))
        anchor_age = int(np.clip(rng.normal(64, 16), 18, 91))
        male = rng.random() < 0.5
        severity = rng.normal(0.25 * (g >= 2), 1.0)
        if rng.random() < 0.03:          # ED visit that did not lead to an admission
            t = datetime(anchor_year, 1, 1) + timedelta(minutes=int(rng.integers(0, 525_000)))
            transfers.append([subject, "", ids.next("transfer"), "ED", "Emergency Department", t.strftime(TS),
                              (t + timedelta(hours=5)).strftime(TS)])
        admit = datetime(anchor_year, 1, 1) + timedelta(minutes=int(rng.integers(0, 525_000)))
        years_later = 0
        died = False
        for _ in range(1 + int(rng.random() < 0.25)):     # a quarter of patients come back
            late = (g + years_later // 3) >= 2
            disch, died = admission(rng, ids, subject, admit, severity, late, anchor_age + years_later, rows)
            if died:
                break
            gap = int(rng.integers(1, 6))
            years_later += gap
            admit = disch + timedelta(days=365 * gap + int(rng.integers(0, 60)))
        patients.append([subject, "M" if male else "F", anchor_age, anchor_year, GROUPS[g], ""])
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
    print(f"wrote synthetic MIMIC-IV-shaped fixture to {root} ({args.patients} patients, {len(admissions)} admissions, "
          f"{len(labs)} lab rows)")


if __name__ == "__main__":
    main()
