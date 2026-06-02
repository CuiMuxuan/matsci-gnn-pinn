#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/matsci-gnn-pinn}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-500}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
LAYERS="${LAYERS:-4}"
LR="${LR:-1e-3}"
SEEDS="${SEEDS:-1 2}"
SUMMARY_SEEDS="${SUMMARY_SEEDS:-7 1 2}"
DATASET_LIMITS="${DATASET_LIMITS:-12 21}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-spot_size}"
BASE_PROFILE_TAG="${BASE_PROFILE_TAG:-broad_process_profile}"
PROCESS_CONDITIONING_PROFILE="${PROCESS_CONDITIONING_PROFILE:-broad_process_v1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_NO_PROCESS="${RUN_NO_PROCESS:-1}"
RUN_BROAD_PROCESS="${RUN_BROAD_PROCESS:-1}"
REQUIRE_PASS="${REQUIRE_PASS:-0}"

cd "$REPO_ROOT"
mkdir -p logs outputs/runs outputs/reports

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

base_run_id() {
  local dataset_limit="$1"
  echo "ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${SPLIT_STRATEGY}_${BASE_PROFILE_TAG}_smoke_a100_sxm4_40gb_v1"
}

check_base_artifacts() {
  local dataset_limit="$1"
  local run_id
  run_id="$(base_run_id "$dataset_limit")"
  local table="data/interim/ambench/2022_single_track/AMB2022-03/${run_id}.csv"
  local split="outputs/data_splits/${run_id}_split.json"
  if [[ ! -f "$table" || ! -f "$split" ]]; then
    echo "Missing Phase 30 base table/split for broad${dataset_limit} ${SPLIT_STRATEGY}: ${run_id}" >&2
    echo "Run scripts/server/run_phase30_broad_process_selector_smoke_a100.sh for this split before Phase 55." >&2
    exit 2
  fi
}

run_macro_pinn() {
  local dataset_limit="$1"
  local seed="$2"
  local tag="$3"
  shift 3
  local run_id
  run_id="$(base_run_id "$dataset_limit")"
  local table="data/interim/ambench/2022_single_track/AMB2022-03/${run_id}.csv"
  local split="outputs/data_splits/${run_id}_split.json"
  local out_dir="outputs/runs/${run_id}_seed${seed}_macro_pinn_minmax_${tag}_v1"

  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/metrics.json" ]]; then
    echo "=== skip existing ${out_dir} ==="
    return
  fi

  echo "=== broad${dataset_limit} ${SPLIT_STRATEGY} seed=${seed} tag=${tag} ==="
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split" \
    --output-dir "$out_dir" \
    --steps "$STEPS" \
    --hidden-dim "$HIDDEN_DIM" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --seed "$seed" \
    --device "$DEVICE" \
    --input-normalization minmax \
    --spacetime-encoding raw \
    "$@" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

for dataset_limit in $DATASET_LIMITS; do
  check_base_artifacts "$dataset_limit"
  for seed in $SEEDS; do
    if [[ "$RUN_NO_PROCESS" == "1" ]]; then
      run_macro_pinn "$dataset_limit" "$seed" no_process
    fi
    if [[ "$RUN_BROAD_PROCESS" == "1" ]]; then
      run_macro_pinn "$dataset_limit" "$seed" "$BASE_PROFILE_TAG" \
        --input-conditioning-mode concat \
        --input-feature-normalization same \
        --input-conditioning-profile "$PROCESS_CONDITIONING_PROFILE" \
        --input-feature-column laser_power_W \
        --input-feature-column scan_speed_mm_s \
        --input-feature-column spot_size_um
    fi
  done
done

summary_args=()
for dataset_limit in $DATASET_LIMITS; do
  summary_args+=(--dataset-limit "$dataset_limit")
done
for seed in $SUMMARY_SEEDS; do
  summary_args+=(--seed "$seed")
done
if [[ "$REQUIRE_PASS" == "1" ]]; then
  summary_args+=(--require-pass)
fi

"$CONDA_BIN" run -n "$CONDA_ENV" python -X utf8 scripts/server/summarize_phase55_spot_size_seed_check.py \
  "${summary_args[@]}" \
  --dataset-order "$DATASET_ORDER" \
  --split "$SPLIT_STRATEGY" \
  --json-output outputs/reports/phase55_spot_size_route_seed_check_summary.json \
  --markdown-output outputs/reports/phase55_spot_size_route_seed_check_summary.md \
  --require-complete
