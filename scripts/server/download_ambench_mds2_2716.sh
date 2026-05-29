#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONDA_BIN="${CONDA_BIN:-/home/vipuser/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-gnnpinn}"
DATA_ROOT="${DATA_ROOT:-data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716}"
OUTPUT="${OUTPUT:-outputs/data_audits/ambench_mds2_2716_download_report.json}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT")"

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

"$CONDA_BIN" run -n "$CONDA_ENV" python -m gnnpinn.data.ambench_downloads \
  --root "$DATA_ROOT" \
  --download \
  --verify-sha256 \
  --output "$OUTPUT"
