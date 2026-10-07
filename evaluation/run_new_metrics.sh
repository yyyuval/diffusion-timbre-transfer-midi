#!/usr/bin/env bash
# Run the additive metrics from docs/METRICS_HANDOFF.md.
# Assumes inference_metrics.csv already exists and FAD wav folders are prepared
# (evaluation/prepare_fad_sets.py + evaluation/compute_clap_fad.py as before).
#
# Usage:
#   bash evaluation/run_new_metrics.sh
#   TRACK_LIST=evaluation/track_list_20.txt bash evaluation/run_new_metrics.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
TRACK_LIST="${TRACK_LIST:-evaluation/track_list_20.txt}"
SOURCE_ENSEMBLE="${SOURCE_ENSEMBLE:-flute_bassoon}"
NOTE_THRESHOLD="${NOTE_THRESHOLD:-0.5}"

echo "=== 1) Transcription ceiling (real input vs GT MIDI) ==="
"$PYTHON" evaluation/compute_transcription_ceiling.py \
  --track-list "$TRACK_LIST" \
  --ensemble "$SOURCE_ENSEMBLE" \
  --note-threshold "$NOTE_THRESHOLD"

echo "=== 2) Content metrics on generated outputs ==="
"$PYTHON" evaluation/compute_content_metrics.py \
  --source "$SOURCE_ENSEMBLE" \
  --note-threshold "$NOTE_THRESHOLD"

echo "=== 3) EnCodec FAD (requires fad/ wav folders) ==="
"$PYTHON" evaluation/compute_encodec_fad.py

echo "=== 4) Timbre classifier train (real stems) — optional / long ==="
if [[ "${TRAIN_CLASSIFIER:-0}" == "1" ]]; then
  "$PYTHON" evaluation/train_timbre_classifier.py
else
  echo "skip classifier (set TRAIN_CLASSIFIER=1 to train)"
fi

echo "Done. CSVs under Outputs/WAV_files/metrics/"
