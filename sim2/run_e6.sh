#!/usr/bin/env bash
# E6 end to end on a local MIMIC-IV copy. Run it yourself, on the machine that
# holds the data; every step prints and writes aggregates only.
#
#   sim2/run_e6.sh demo               open MIMIC-IV demo (no credentials, ~2 MB): extraction smoke test
#   sim2/run_e6.sh download USER      six credentialed tables (~3 GB) into $MIMIC_ROOT; asks for the password
#   sim2/run_e6.sh cohort             data/e6_cohort.npz, with the acceptance checks of NEXT_STEPS 1.4
#   sim2/run_e6.sh main               figures/merged_expE6.json and Figure 7
#   sim2/run_e6.sh sensitivity        tagged appendix runs (effect size, alert rate, periods, COVID era)
#   sim2/run_e6.sh report             figures, tables, F7 numbers, validator
#   sim2/run_e6.sh all USER           download, cohort, main, sensitivity, report
#
# Environment: MIMIC_ROOT (default ~/physionet/mimiciv/$MIMIC_VERSION), MIMIC_VERSION (3.1),
# WORKERS (4), SENS_REPS_A (100), SENS_REPS_C (30), MEMORY_LIMIT (4GB, DuckDB).
set -euo pipefail
cd "$(dirname "$0")/.."

MIMIC_VERSION=${MIMIC_VERSION:-3.1}
MIMIC_ROOT=${MIMIC_ROOT:-$HOME/physionet/mimiciv/$MIMIC_VERSION}
WORKERS=${WORKERS:-4}
SENS_REPS_A=${SENS_REPS_A:-100}
SENS_REPS_C=${SENS_REPS_C:-30}
MEMORY_LIMIT=${MEMORY_LIMIT:-4GB}
TABLES="hosp/patients hosp/admissions hosp/transfers hosp/labevents hosp/d_labitems icu/icustays"
PY=.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

say() { printf '\n== %s\n' "$*"; }

venv() {
  if [ ! -x "$PY" ]; then
    say "creating .venv"
    python3 -m venv .venv
  fi
  "$PY" -c "import duckdb, sklearn" 2>/dev/null || "$PY" -m pip install -q -r requirements-e6.txt
}

# fetch BASE_URL DEST [NETRC]: the six tables, resuming partial files, then SHA256 check when sums are available
fetch() {
  local base=$1 dest=$2 netrc=${3:-} auth=()
  [ -n "$netrc" ] && auth=(--netrc-file "$netrc")
  for f in $TABLES; do
    local out="$dest/$f.csv.gz"
    mkdir -p "$(dirname "$out")"
    if [ -s "$out" ]; then echo "have $f"; continue; fi
    echo "fetching $f"
    if ! curl --fail --location --progress-bar ${auth[@]+"${auth[@]}"} --continue-at - -o "$out.part" "$base/$f.csv.gz"; then
      echo "could not resume $f; fetching it again from the start"
      rm -f "$out.part"
      curl --fail --location --progress-bar ${auth[@]+"${auth[@]}"} -o "$out.part" "$base/$f.csv.gz"
    fi
    mv "$out.part" "$out"
  done
  if ! curl --fail --silent --location ${auth[@]+"${auth[@]}"} -o "$dest/SHA256SUMS.txt" "$base/SHA256SUMS.txt"; then
    echo "no SHA256SUMS.txt at $base: checksums not verified"
    return 0
  fi
  for f in $TABLES; do
    local want got
    want=$(awk -v f="$f.csv.gz" '$NF == f {print $1}' "$dest/SHA256SUMS.txt")
    got=$(shasum -a 256 "$dest/$f.csv.gz" | awk '{print $1}')
    if [ -z "$want" ]; then echo "not in SHA256SUMS.txt: $f"
    elif [ "$want" = "$got" ]; then echo "checksum ok: $f"
    else echo "checksum mismatch: $f (delete it and fetch again)" >&2; exit 1
    fi
  done
}

demo() {
  venv
  say "MIMIC-IV demo 2.2 (open access, ODbL): extraction smoke test"
  fetch https://physionet.org/files/mimic-iv-demo/2.2 data/mimic-iv-demo
  # The demo is open, so errors may be printed in full. Results at n = 100 patients mean nothing,
  # and the acceptance checks (size, period coverage) are expected to fail.
  "$PY" sim2/e6_mimic_cohort.py --mimic-root data/mimic-iv-demo --out data/e6_demo_cohort.npz \
        --show-errors --no-checks --memory-limit "$MEMORY_LIMIT"
}

download() {
  local user=${1:?usage: sim2/run_e6.sh download PHYSIONET_USERNAME}
  local parent; parent=$(mkdir -p "$MIMIC_ROOT" && cd "$MIMIC_ROOT/.." && pwd)
  local free_gb; free_gb=$(df -Pk "$parent" | awk 'NR==2 {printf "%d", $4 / 1048576}')
  say "MIMIC-IV $MIMIC_VERSION, six tables into $MIMIC_ROOT (${free_gb} GB free)"
  if [ "$free_gb" -lt 5 ]; then
    echo "need about 5 GB free (labevents.csv.gz is the large file, plus DuckDB work space); set MIMIC_ROOT elsewhere" >&2
    exit 1
  fi
  NETRC_DIR=$(mktemp -d)
  trap 'rm -rf "$NETRC_DIR"' EXIT
  local pass
  read -r -s -p "PhysioNet password for $user: " pass; echo
  pass=$(printf '%s' "$pass" | sed 's/\\/\\\\/g; s/"/\\"/g')     # quoted netrc field (curl >= 7.84)
  (umask 077; printf 'machine physionet.org login %s password "%s"\n' "$user" "$pass" > "$NETRC_DIR/netrc")
  unset pass
  fetch "https://physionet.org/files/mimiciv/$MIMIC_VERSION" "$MIMIC_ROOT" "$NETRC_DIR/netrc"
  rm -rf "$NETRC_DIR"
  trap - EXIT
}

cohort() {
  venv
  say "cohort from $MIMIC_ROOT"
  "$PY" sim2/e6_mimic_cohort.py --mimic-root "$MIMIC_ROOT" --out data/e6_cohort.npz --memory-limit "$MEMORY_LIMIT"
}

main_run() {
  venv
  say "main run (A: 200 replicates, C: 50; about 20 to 40 minutes with $WORKERS workers)"
  "$PY" sim2/exp_E6_mimic.py --cohort data/e6_cohort.npz --workers "$WORKERS"
}

sensitivity() {
  venv
  local common=(--cohort data/e6_cohort.npz --workers "$WORKERS" --reps-a "$SENS_REPS_A" --reps-c "$SENS_REPS_C")
  say "sensitivity runs ($SENS_REPS_A / $SENS_REPS_C replicates each)"
  "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag rrr25 --rrr 0.25
  "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag rrr75 --rrr 0.75
  "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag alert05 --alert-rate 0.05
  "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag alert20 --alert-rate 0.20
  "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag admyear --period-by admission_year
  if "$PY" -c "import numpy as np, sys; sys.exit(0 if '2020 - 2022' in set(np.load('data/e6_cohort.npz')['period'].tolist()) else 1)"; then
    "$PY" sim2/exp_E6_mimic.py "${common[@]}" --tag covid --late "2020 - 2022"
  else
    echo "no 2020 - 2022 anchor group (MIMIC-IV v2.2): COVID-era run skipped"
  fi
}

report() {
  venv
  say "figures, tables, F7 numbers"
  "$PY" sim2/make_figures2.py
  "$PY" sim2/make_tables.py
  "$PY" sim2/e6_report.py
  say "validator (fails on the open F7 TODO until the manuscript is updated with the numbers above)"
  "$PY" sim2/validate_josh_results.py || true
  say "commit only aggregates"
  echo "git add figures/merged_expE6*.json figures/fig7_mimic*.pdf aaai/figures/merged_expE6*.json aaai/figures/fig7_mimic*.pdf aaai/tables/e6_*.tex"
  echo "data/ (cohort, logs, demo) is ignored by git and stays on this machine."
}

case "${1:-}" in
  demo) demo ;;
  download) download "${2:-}" ;;
  cohort) cohort ;;
  main) main_run ;;
  sensitivity) sensitivity ;;
  report) report ;;
  all) download "${2:-}"; cohort; main_run; sensitivity; report ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
