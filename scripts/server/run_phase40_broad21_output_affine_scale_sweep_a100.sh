#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/matsci-gnn-pinn}"
OUTPUT_AFFINE_SCALES="${OUTPUT_AFFINE_SCALES:-0.1 0.25}"
OUTPUT_AFFINE_TAG_PREFIX="${OUTPUT_AFFINE_TAG_PREFIX:-oa}"
PHASE40_LOG_PATH="${PHASE40_LOG_PATH:-logs/phase40_broad21_laser_power_output_affine_scale_sweep_a100_v1.log}"
PHASE40_SELF_LOG="${PHASE40_SELF_LOG:-1}"

PROFILE_SPLITS="${PROFILE_SPLITS:-laser_power}"
DATASET_LIMIT="${DATASET_LIMIT:-21}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
STEPS="${STEPS:-500}"
N_ESTIMATORS="${N_ESTIMATORS:-80}"

cd "$REPO_ROOT"

if [[ "$PHASE40_SELF_LOG" == "1" ]]; then
  mkdir -p "$(dirname "$PHASE40_LOG_PATH")"
  exec > "$PHASE40_LOG_PATH" 2>&1
fi

for scale in $OUTPUT_AFFINE_SCALES; do
  tag="${OUTPUT_AFFINE_TAG_PREFIX}${scale//./}"
  echo "=== Phase 40 broad21 output-affine scale=${scale} tag=${tag} ==="
  PROFILE_SPLITS="$PROFILE_SPLITS" \
  DATASET_LIMIT="$DATASET_LIMIT" \
  DATASET_ORDER="$DATASET_ORDER" \
  STEPS="$STEPS" \
  N_ESTIMATORS="$N_ESTIMATORS" \
  OUTPUT_AFFINE_SCALE="$scale" \
  OUTPUT_AFFINE_TAG="$tag" \
    bash scripts/server/run_phase39_broad_output_affine_a100.sh
done
