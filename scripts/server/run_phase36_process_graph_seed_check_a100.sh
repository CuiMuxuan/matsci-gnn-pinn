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
PROFILE_SPLITS="${PROFILE_SPLITS:-spot_size laser_power}"
DATASET_LIMIT="${DATASET_LIMIT:-12}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
PROCESS_GRAPH_TAG="${PROCESS_GRAPH_TAG:-pg_rbf_global}"
PROCESS_GRAPH_FEATURE_COUNT="${PROCESS_GRAPH_FEATURE_COUNT:-4}"
PROCESS_GRAPH_LENGTH_SCALE="${PROCESS_GRAPH_LENGTH_SCALE:-1.0}"
PROCESS_GRAPH_FIT_SCOPE="${PROCESS_GRAPH_FIT_SCOPE:-global}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$REPO_ROOT"
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

base_run_id_for_split() {
  local split="$1"
  echo "ambench_multiline_process_temperature_broad${DATASET_LIMIT}_${DATASET_ORDER}_${split}_broad_process_profile_smoke_a100_sxm4_40gb_v1"
}

run_macro_pinn() {
  local split="$1"
  local seed="$2"
  local tag="$3"
  shift 3

  local base_run_id
  base_run_id="$(base_run_id_for_split "$split")"
  local table="data/interim/ambench/2022_single_track/AMB2022-03/${base_run_id}.csv"
  local split_manifest="outputs/data_splits/${base_run_id}_split.json"
  local out_dir="outputs/runs/${base_run_id}_seed${seed}_macro_pinn_minmax_${tag}_v1"

  if [[ ! -f "$table" ]]; then
    echo "Missing table for ${split}: ${table}" >&2
    exit 2
  fi
  if [[ ! -f "$split_manifest" ]]; then
    echo "Missing split manifest for ${split}: ${split_manifest}" >&2
    exit 2
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/metrics.json" ]]; then
    echo "=== skip existing ${out_dir} ==="
    return
  fi

  echo "=== ${out_dir} ==="
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$table" \
    --target temperature_C \
    --split-manifest "$split_manifest" \
    --output-dir "$out_dir" \
    --steps "$STEPS" \
    --hidden-dim "$HIDDEN_DIM" \
    --layers "$LAYERS" \
    --lr "$LR" \
    --seed "$seed" \
    --device "$DEVICE" \
    --input-normalization minmax \
    --spacetime-encoding raw \
    --input-conditioning-mode concat \
    --input-feature-normalization same \
    --input-conditioning-profile broad_process_v1 \
    --input-feature-column laser_power_W \
    --input-feature-column scan_speed_mm_s \
    --input-feature-column spot_size_um \
    "$@" \
    --hot-quantile 0.9 \
    --gradient-quantile 0.9 \
    --log-every 100
}

for split_strategy in $PROFILE_SPLITS; do
  case "$split_strategy" in
    line|laser_power|scan_speed|spot_size|process)
      ;;
    *)
      echo "Unsupported PROFILE_SPLITS entry: ${split_strategy}" >&2
      exit 2
      ;;
  esac
  for seed in $SEEDS; do
    run_macro_pinn "$split_strategy" "$seed" broad_process_profile
    run_macro_pinn "$split_strategy" "$seed" "$PROCESS_GRAPH_TAG" \
      --process-graph-feature-mode rbf \
      --process-graph-feature-column laser_power_W \
      --process-graph-feature-column scan_speed_mm_s \
      --process-graph-feature-column spot_size_um \
      --process-graph-feature-count "$PROCESS_GRAPH_FEATURE_COUNT" \
      --process-graph-length-scale "$PROCESS_GRAPH_LENGTH_SCALE" \
      --process-graph-fit-scope "$PROCESS_GRAPH_FIT_SCOPE"
  done
done
