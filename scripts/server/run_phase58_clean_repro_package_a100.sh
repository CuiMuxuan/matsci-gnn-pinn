#!/usr/bin/env bash
# Rebuild Phase 55/56/57 reports from a clean GitHub checkout.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/CuiMuxuan/matsci-gnn-pinn.git}"
BRANCH="${BRANCH:-main}"
SOURCE_REPO="${SOURCE_REPO:-/root/matsci-gnn-pinn}"
CLEAN_REPO="${CLEAN_REPO:-/root/matsci-gnn-pinn-phase58-clean}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATASET_LIMITS="${DATASET_LIMITS:-12 21}"
DATASET_ORDER="${DATASET_ORDER:-process_round_robin}"
SEEDS="${SEEDS:-7 1 2}"
SPLIT="${SPLIT:-spot_size}"
PROFILE_TAG="${PROFILE_TAG:-broad_process_profile}"
PYTHONUTF8="${PYTHONUTF8:-1}"
PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

export PYTHONUTF8 PYTHONIOENCODING

if [ ! -d "${SOURCE_REPO}/outputs" ]; then
  echo "Missing source artifact tree: ${SOURCE_REPO}/outputs" >&2
  exit 2
fi

if [ ! -d "${CLEAN_REPO}/.git" ]; then
  rm -rf "${CLEAN_REPO}"
  git clone --branch "${BRANCH}" "${REPO_URL}" "${CLEAN_REPO}"
else
  git -C "${CLEAN_REPO}" fetch origin "${BRANCH}"
  git -C "${CLEAN_REPO}" checkout "${BRANCH}"
  git -C "${CLEAN_REPO}" reset --hard "origin/${BRANCH}"
  git -C "${CLEAN_REPO}" clean -fdx
fi

cd "${CLEAN_REPO}"
mkdir -p outputs/reports outputs/baselines outputs/runs docs/results/phase58_clean_repro

python_bin=(python3)
if [ -x "${CONDA_BIN}" ]; then
  python_bin=("${CONDA_BIN}" run -n "${CONDA_ENV}" python)
fi

for rel in \
  "outputs/reports/phase54_broad12_claim_boundary_input_summary.json" \
  "outputs/reports/phase54_broad21_claim_boundary_input_summary.json"; do
  if [ ! -f "${SOURCE_REPO}/${rel}" ]; then
    echo "Missing required source report: ${SOURCE_REPO}/${rel}" >&2
    exit 3
  fi
  mkdir -p "$(dirname "${rel}")"
  cp "${SOURCE_REPO}/${rel}" "${rel}"
done

copy_required_artifact() {
  local rel="$1"
  if [ ! -f "${SOURCE_REPO}/${rel}" ]; then
    echo "Missing required source artifact: ${SOURCE_REPO}/${rel}" >&2
    exit 4
  fi
  mkdir -p "$(dirname "${rel}")"
  cp "${SOURCE_REPO}/${rel}" "${rel}"
}

for dataset_limit in ${DATASET_LIMITS}; do
  base_run_id="ambench_multiline_process_temperature_broad${dataset_limit}_${DATASET_ORDER}_${SPLIT}_${PROFILE_TAG}_smoke_a100_sxm4_40gb_v1"
  for tag in mean_constant knn_coords knn_process extra_trees_coords extra_trees_process; do
    copy_required_artifact "outputs/baselines/${base_run_id}_${tag}_regions_q90.json"
  done
  for seed in ${SEEDS}; do
    for run_tag in no_process "${PROFILE_TAG}"; do
      if [ "${seed}" = "7" ]; then
        copy_required_artifact "outputs/runs/${base_run_id}_macro_pinn_minmax_${run_tag}_v1/metrics.json"
      else
        copy_required_artifact "outputs/runs/${base_run_id}_seed${seed}_macro_pinn_minmax_${run_tag}_v1/metrics.json"
      fi
    done
  done
done

"${python_bin[@]}" scripts/server/summarize_phase54_process_route_claim_boundary.py \
  --input outputs/reports/phase54_broad12_claim_boundary_input_summary.json \
  --input outputs/reports/phase54_broad21_claim_boundary_input_summary.json \
  --json-output outputs/reports/phase54_process_route_claim_boundary_summary.json \
  --markdown-output outputs/reports/phase54_process_route_claim_boundary_summary.md \
  --require-comparable \
  > docs/results/phase58_clean_repro/phase58_phase54_rebuild.log

"${python_bin[@]}" scripts/server/summarize_phase55_spot_size_seed_check.py \
  $(for dataset_limit in ${DATASET_LIMITS}; do printf -- "--dataset-limit %s " "${dataset_limit}"; done) \
  --dataset-order "${DATASET_ORDER}" \
  --split "${SPLIT}" \
  $(for seed in ${SEEDS}; do printf -- "--seed %s " "${seed}"; done) \
  --profile-tag "${PROFILE_TAG}" \
  --json-output outputs/reports/phase55_spot_size_route_seed_check_summary.json \
  --markdown-output outputs/reports/phase55_spot_size_route_seed_check_summary.md \
  --require-complete \
  --require-pass \
  > docs/results/phase58_clean_repro/phase58_phase55_rebuild.log

"${python_bin[@]}" scripts/server/build_phase56_manuscript_package.py \
  --root . \
  --output-dir docs/results/phase56_manuscript_package \
  --manifest-output outputs/reports/phase56_manuscript_package_manifest.json \
  > docs/results/phase58_clean_repro/phase58_phase56_rebuild.log

"${python_bin[@]}" scripts/server/build_phase57_claim_governance.py \
  --root . \
  --output-dir docs/results/phase57_claim_governance \
  > docs/results/phase58_clean_repro/phase58_phase57_rebuild.log

"${python_bin[@]}" - <<'PY'
import json
from pathlib import Path

root = Path(".")
phase55 = json.loads((root / "outputs/reports/phase55_spot_size_route_seed_check_summary.json").read_text())
phase57 = json.loads((root / "docs/results/phase57_claim_governance/phase57_claim_governance_manifest.json").read_text())
summary = {
    "phase": 58,
    "objective": "clean_checkout_rebuild_phase55_56_57",
    "commit": __import__("subprocess").check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
    "source_repo": str(Path(__import__("os").environ.get("SOURCE_REPO", "/root/matsci-gnn-pinn"))),
    "transfer_gate": phase55.get("transfer_gate"),
    "ledger_counts": phase57.get("ledger_counts"),
    "outputs": {
        "phase54_summary": "outputs/reports/phase54_process_route_claim_boundary_summary.json",
        "phase55_summary": "outputs/reports/phase55_spot_size_route_seed_check_summary.json",
        "phase56_manifest": "outputs/reports/phase56_manuscript_package_manifest.json",
        "phase57_manifest": "docs/results/phase57_claim_governance/phase57_claim_governance_manifest.json",
    },
}
out = root / "docs/results/phase58_clean_repro/phase58_clean_repro_manifest.json"
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
