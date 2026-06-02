#!/usr/bin/env bash
# Run low-risk stronger-baseline stress tests for the Phase 55 spot-size claim.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/matsci-gnn-pinn}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
PYTHONUTF8="${PYTHONUTF8:-1}"
PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
N_ESTIMATORS="${N_ESTIMATORS:-200}"
RANDOM_STATE="${RANDOM_STATE:-7}"

export PYTHONUTF8 PYTHONIOENCODING

cd "$REPO_ROOT"
mkdir -p outputs/baselines outputs/reports logs

run_stress_baseline() {
  local dataset_label="$1"
  local base_run_id="$2"
  local table="$3"
  local split_manifest="$4"
  local strategy="$5"
  local tag="$6"
  shift 6
  echo "[phase58] ${dataset_label} ${strategy}/${tag}"
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.eval.field_baseline \
    --table "$table" \
    --target temperature_C \
    --strategy "$strategy" \
    --split-manifest "$split_manifest" \
    --fit-split train \
    --n-estimators "$N_ESTIMATORS" \
    --random-state "$RANDOM_STATE" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    "$@" \
    --output "outputs/baselines/${base_run_id}_${strategy}_${tag}_regions_q90.json"
}

artifact_index_py="outputs/reports/_phase58_spot_size_artifact_index.py"
cat > "$artifact_index_py" <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("outputs/reports/phase55_spot_size_route_seed_check_summary.json").read_text())
for dataset in summary["datasets"]:
    metrics_path = Path(dataset["rows"]["broad_process_v1"][0]["path"])
    metrics = json.loads(metrics_path.read_text())
    config = metrics["config"]
    print("\t".join([dataset["label"], dataset["base_run_id"], config["table"], config["split_manifest"]]))
PY
"$CONDA_BIN" run -n "$CONDA_ENV" python "$artifact_index_py" > outputs/reports/phase58_spot_size_artifact_index.tsv
rm -f "$artifact_index_py"

while IFS=$'\t' read -r dataset_label base_run_id table split_manifest; do
  if [[ -z "${dataset_label}${base_run_id}${table}${split_manifest}" ]]; then
    continue
  fi
  if [[ -z "$dataset_label" || -z "$base_run_id" || -z "$table" || -z "$split_manifest" ]]; then
    echo "[phase58] incomplete artifact index row: ${dataset_label:-<empty>} ${base_run_id:-<empty>} ${table:-<empty>} ${split_manifest:-<empty>}" >&2
    exit 2
  fi
  run_stress_baseline "$dataset_label" "$base_run_id" "$table" "$split_manifest" random_forest coords \
    --feature-column x --feature-column y --feature-column t
  run_stress_baseline "$dataset_label" "$base_run_id" "$table" "$split_manifest" random_forest process \
    --feature-column x --feature-column y --feature-column t \
    --feature-column laser_power_W --feature-column scan_speed_mm_s --feature-column spot_size_um
  run_stress_baseline "$dataset_label" "$base_run_id" "$table" "$split_manifest" hist_gradient_boosting coords \
    --feature-column x --feature-column y --feature-column t
  run_stress_baseline "$dataset_label" "$base_run_id" "$table" "$split_manifest" hist_gradient_boosting process \
    --feature-column x --feature-column y --feature-column t \
    --feature-column laser_power_W --feature-column scan_speed_mm_s --feature-column spot_size_um
done < outputs/reports/phase58_spot_size_artifact_index.tsv

"$CONDA_BIN" run -n "$CONDA_ENV" python scripts/server/summarize_phase58_stronger_baseline_stress.py \
  --root . \
  --json-output outputs/reports/phase58_stronger_baseline_stress_summary.json \
  --markdown-output outputs/reports/phase58_stronger_baseline_stress_summary.md \
  --require-complete \
  --require-pass
