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
REGION_WEIGHTED_TAGS="${REGION_WEIGHTED_TAGS:-rw15 rw125}"
DATA_LOSS_WEIGHTING="${DATA_LOSS_WEIGHTING:-hot_gradient}"
DATA_LOSS_HOT_QUANTILE="${DATA_LOSS_HOT_QUANTILE:-0.9}"
DATA_LOSS_GRADIENT_QUANTILE="${DATA_LOSS_GRADIENT_QUANTILE:-0.9}"
BASE_RUN_ID="${BASE_RUN_ID:-ambench_multiline_process_temperature_broad12_process_round_robin_spot_size_broad_process_profile_smoke_a100_sxm4_40gb_v1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

cd "$REPO_ROOT"
mkdir -p logs outputs/runs

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

TABLE="data/interim/ambench/2022_single_track/AMB2022-03/${BASE_RUN_ID}.csv"
SPLIT="outputs/data_splits/${BASE_RUN_ID}_split.json"

region_weight_for_tag() {
  case "$1" in
    rw2) echo "2.0" ;;
    rw15) echo "1.5" ;;
    rw135) echo "1.35" ;;
    rw125) echo "1.25" ;;
    *)
      echo "Unsupported REGION_WEIGHTED_TAGS entry: $1" >&2
      exit 2
      ;;
  esac
}

run_macro_pinn() {
  local seed="$1"
  local tag="$2"
  shift 2
  local out_dir="outputs/runs/${BASE_RUN_ID}_seed${seed}_macro_pinn_minmax_${tag}_v1"

  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/metrics.json" ]]; then
    echo "=== skip existing ${out_dir} ==="
    return
  fi

  echo "=== ${out_dir} ==="
  "$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.train.macro_pinn \
    --table "$TABLE" \
    --target temperature_C \
    --split-manifest "$SPLIT" \
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

for seed in $SEEDS; do
  run_macro_pinn "$seed" broad_process_profile
  for tag in $REGION_WEIGHTED_TAGS; do
    weight="$(region_weight_for_tag "$tag")"
    run_macro_pinn "$seed" "$tag" \
      --data-loss-weighting "$DATA_LOSS_WEIGHTING" \
      --data-loss-hot-quantile "$DATA_LOSS_HOT_QUANTILE" \
      --data-loss-gradient-quantile "$DATA_LOSS_GRADIENT_QUANTILE" \
      --data-loss-region-weight "$weight"
  done
done
