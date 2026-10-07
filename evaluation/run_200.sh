#!/usr/bin/env bash
# 200-clip handoff sweep: generation, then metrics, then the summary.
#
#   bash evaluation/run_200.sh tracks                         # write the two track lists
#   GPU=0 bash evaluation/run_200.sh verify [pair] [handoff]  # ALWAYS FIRST
#   bash evaluation/run_200.sh split 0 3 5 6                  # print the per-GPU nohup commands
#   bash evaluation/run_200.sh launch 0 3 5 6                 # start them (detached, one per GPU)
#   bash evaluation/run_200.sh status                         # clips written per experiment
#   GPU=0 nohup bash evaluation/run_200.sh metrics > /dev/null 2>&1 &
#
# Env: PYTHON, OUT_ROOT, BATCH_SIZE, N, HANDOFFS ("5 25 50 75 100"), BASELINE=1
# (adds the sigma_max=5 pairs at handoff 5), NOTE_THRESHOLD, CONTENT_WORKERS,
# MONO_CONTENT=1 (mir_eval on mono too), ENCODEC_FAD=1, TRAIN_CLASSIFIER=1.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV_PY=/home/shared_workspace/diffusion-timbre-transfer/venv_yuval/bin/python
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$VENV_PY" ]]; then PYTHON="$VENV_PY"; else PYTHON=python; fi
fi
OUT_ROOT="${OUT_ROOT:-Outputs/WAV_files/metrics_200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N="${N:-200}"
HANDOFFS="${HANDOFFS:-5 25 50 75 100}"
NOTE_THRESHOLD="${NOTE_THRESHOLD:-0.5}"
CONTENT_WORKERS="${CONTENT_WORKERS:-6}"
LOG_DIR="$OUT_ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"/{audio,csv,logs,content,ceiling,fad,summary}

RUNNER=(evaluation/run_bridge_batched.py --repo-path "$ROOT" --out-root "$OUT_ROOT" --batch-size "$BATCH_SIZE"
        --n "$N" --handoffs $HANDOFFS
        --track-list-poly "evaluation/track_list_${N}_poly.txt"
        --track-list-mono "evaluation/track_list_${N}_mono.txt")
[[ "${BASELINE:-0}" == "1" ]] && RUNNER+=(--baseline)

log_to() {
  echo "logging to $1"
  exec > >(tee -a "$1") 2>&1
}

cmd="${1:-}"
shift || true

case "$cmd" in
  tracks)
    "$PYTHON" evaluation/make_track_lists.py --n "$N"
    ;;

  verify)
    log_to "$LOG_DIR/verify_$STAMP.log"
    export CUDA_VISIBLE_DEVICES="${GPU:?set GPU=<physical index>}"
    "$PYTHON" -u "${RUNNER[@]}" --verify \
      --verify-pair "${1:-original_midi_s100}" --verify-handoff "${2:-25}"
    ;;

  split)
    "$PYTHON" "${RUNNER[@]}" --print-split "$@"
    ;;

  launch)
    [[ $# -ge 1 ]] || { echo "usage: run_200.sh launch <gpu> [<gpu> ...]"; exit 1; }
    "$PYTHON" "${RUNNER[@]}" --list-ckpts
    "$PYTHON" "${RUNNER[@]}" --launch-gpus "$@"
    ;;

  generate)
    log_to "$LOG_DIR/gen_gpu${GPU:-x}_$STAMP.log"
    export CUDA_VISIBLE_DEVICES="${GPU:?set GPU=<physical index>}"
    "$PYTHON" -u "${RUNNER[@]}" "$@"
    ;;

  status)
    for f in "$OUT_ROOT"/csv/*.csv; do
      [[ -e "$f" ]] || { echo "no CSVs yet"; break; }
      printf "%-45s %5d\n" "$(basename "$f" .csv)" "$(($(wc -l < "$f") - 1))"
    done
    ps aux | grep -c "[r]un_bridge_batched.py" | xargs echo "running workers:"
    ;;

  metrics)
    log_to "$LOG_DIR/metrics_$STAMP.log"
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
    if pgrep -f "run_bridge_batched.py --jobs" > /dev/null; then
      echo "WARNING: generation workers are still running; metrics will cover only what exists."
    fi

    echo "=== 1) merge per-experiment CSVs ==="
    "$PYTHON" "${RUNNER[@]}" --merge
    MERGED="$OUT_ROOT/bridge_metrics_all.csv"

    echo "=== 2) transcription ceiling: real inputs vs GT MIDI ==="
    [[ -s "$OUT_ROOT/ceiling/ceiling_poly.csv" ]] || "$PYTHON" evaluation/compute_transcription_ceiling.py \
      --track-list "evaluation/track_list_${N}_poly.txt" --ensemble flute_bassoon \
      --note-threshold "$NOTE_THRESHOLD" --out-csv "$OUT_ROOT/ceiling/ceiling_poly.csv"
    [[ -s "$OUT_ROOT/ceiling/ceiling_mono.csv" ]] || "$PYTHON" evaluation/compute_transcription_ceiling.py \
      --track-list "evaluation/track_list_${N}_mono.txt" --ensemble cello \
      --note-threshold "$NOTE_THRESHOLD" --out-csv "$OUT_ROOT/ceiling/ceiling_mono.csv"

    echo "=== 3) mir_eval content metrics (${CONTENT_WORKERS} parallel CPU workers) ==="
    CASES_FOR_CONTENT="poly"
    [[ "${MONO_CONTENT:-0}" == "1" ]] && CASES_FOR_CONTENT="poly mono"
    EXPS=$("$PYTHON" -c "
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cases = sys.argv[2].split()
print('\n'.join(sorted({r['experiment'] for r in rows if r['case'] in cases})))
" "$MERGED" "$CASES_FOR_CONTENT")
    printf '%s\n' $EXPS | xargs -P "$CONTENT_WORKERS" -I{} \
      "$PYTHON" evaluation/compute_content_metrics.py --metrics-csv "$MERGED" \
        --experiments {} --out-csv "$OUT_ROOT/content/{}.csv" --note-threshold "$NOTE_THRESHOLD"

    echo "=== 4) CLAP-FAD per experiment ==="
    "$PYTHON" evaluation/compute_clap_fad_200.py --out-root "$OUT_ROOT" \
      --exclude-track-lists "evaluation/track_list_${N}_poly.txt" "evaluation/track_list_${N}_mono.txt"

    if [[ "${ENCODEC_FAD:-0}" == "1" ]]; then
      echo "=== 4b) EnCodec-FAD (opt-in) ==="
      for c in poly mono; do
        [[ -d "$OUT_ROOT/fad/$c/generated" ]] || continue
        GEN=()
        for d in "$OUT_ROOT/fad/$c/generated"/*/; do GEN+=("$(basename "$d")=$d"); done
        "$PYTHON" evaluation/compute_encodec_fad.py --real-dir "$OUT_ROOT/fad/$c/reference" \
          --gen-dirs "${GEN[@]}" --out-csv "$OUT_ROOT/fad/encodec_fad_200_$c.csv"
      done
    fi
    if [[ "${TRAIN_CLASSIFIER:-0}" == "1" ]]; then
      echo "=== 4c) timbre classifier (opt-in) ==="
      "$PYTHON" evaluation/train_timbre_classifier.py
    fi

    echo "=== 5) summary ==="
    "$PYTHON" evaluation/summarize_200.py --out-root "$OUT_ROOT"
    ;;

  *)
    sed -n '2,15p' "$0"
    exit 1
    ;;
esac
