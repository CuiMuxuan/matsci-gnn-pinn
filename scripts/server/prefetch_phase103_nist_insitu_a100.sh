#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${REPO_ROOT:-}" ]]; then
  cd "$REPO_ROOT"
else
  cd "$(dirname "$0")/../.."
fi

DATA_ROOT="${DATA_ROOT:-data/raw/nist_ammt/mds2_2044}"
LOG_DIR="${LOG_DIR:-logs}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-5}"
DOWNLOAD_TIMEOUT_SECONDS="${DOWNLOAD_TIMEOUT_SECONDS:-900}"
EXPECTED_BYTES="${EXPECTED_BYTES:-9170420366}"

mkdir -p "$DATA_ROOT" "$LOG_DIR"

output_path="$DATA_ROOT/In-situ Meas Data.zip"
url="https://data.nist.gov/od/ds/85196AB9232E7202E053245706813DFA2044/In-situ%20Meas%20Data.zip"

if [[ -f "$output_path" ]]; then
  actual_bytes="$(wc -c < "$output_path")"
  if [[ "$actual_bytes" == "$EXPECTED_BYTES" ]]; then
    echo "already_complete $output_path $actual_bytes"
    exit 0
  fi
  echo "resume_incomplete $output_path $actual_bytes expected $EXPECTED_BYTES"
fi

wget -c \
  --tries "$DOWNLOAD_RETRIES" \
  --timeout "$DOWNLOAD_TIMEOUT_SECONDS" \
  -O "$output_path" \
  "$url" \
  > "$LOG_DIR/phase103_nist_insitu_prefetch.log" 2>&1

actual_bytes="$(wc -c < "$output_path")"
echo "download_finished $output_path $actual_bytes expected $EXPECTED_BYTES"
