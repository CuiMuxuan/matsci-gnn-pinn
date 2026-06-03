#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
WATCH_LOG="${WATCH_LOG:-$LOG_DIR/phase103_nist_ammt_triage_watch.log}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_CHECKS="${MAX_CHECKS:-0}"
INTAKE_MANIFEST="${INTAKE_MANIFEST:-logs/phase103_nist_ammt_intake_a100_manifest.json}"

mkdir -p "$LOG_DIR"

checks=0
echo "phase103 triage watcher started $(date)" | tee -a "$WATCH_LOG"

while true; do
  checks=$((checks + 1))
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t phase103_nist_large 2>/dev/null; then
    echo "$(date) phase103_nist_large still active; check=$checks" | tee -a "$WATCH_LOG"
  elif [[ -s "$INTAKE_MANIFEST" ]]; then
    echo "$(date) intake manifest ready; launching post-download triage" | tee -a "$WATCH_LOG"
    bash scripts/server/run_phase103_nist_ammt_post_download_triage_a100.sh \
      >> "$WATCH_LOG" 2>&1
    echo "$(date) post-download triage finished" | tee -a "$WATCH_LOG"
    exit 0
  else
    echo "$(date) runner inactive but intake manifest missing or empty: $INTAKE_MANIFEST" | tee -a "$WATCH_LOG"
    exit 2
  fi

  if [[ "$MAX_CHECKS" != "0" && "$checks" -ge "$MAX_CHECKS" ]]; then
    echo "$(date) max checks reached without triage launch" | tee -a "$WATCH_LOG"
    exit 3
  fi
  sleep "$SLEEP_SECONDS"
done
