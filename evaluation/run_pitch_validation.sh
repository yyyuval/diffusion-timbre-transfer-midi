#!/usr/bin/env bash
# Shared-validation pitch evaluation. No FAD or classifier stages.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"
OUT_ROOT="${OUT_ROOT:-Outputs/WAV_files/pitch_validation_200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
JOBS=(no_midi:5 original_midi:5 no_midi_s100:100 original_midi_s100:100
      mono_c2b_no_midi:5 mono_c2b_original_midi:5
      mono_c2b_no_midi_s100:100 mono_c2b_original_midi_s100:100)
RUNNER=(evaluation/run_bridge_batched.py --out-root "$OUT_ROOT" --batch-size "$BATCH_SIZE"
        --n 200 --track-list-poly evaluation/track_list_200_poly.txt
        --track-list-mono evaluation/track_list_200_mono.txt --jobs "${JOBS[@]}")
mkdir -p "$OUT_ROOT/logs"
case "${1:-}" in
  tracks)
    "$PYTHON" evaluation/select_validation_tracks.py --splits "${SPLITS_CSV:-evaluation/splits.csv}"
    ;;
  check)
    "$PYTHON" "${RUNNER[@]}" --list-ckpts
    ;;
  verify)
    export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to an available physical GPU index}"
    "$PYTHON" -u "${RUNNER[@]}" --verify --verify-pair original_midi --verify-handoff 5
    ;;
  smoke)
    export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to an available physical GPU index}"
    "$PYTHON" -u evaluation/run_bridge_batched.py --out-root "${OUT_ROOT}_smoke" \
      --batch-size 1 --n 1 --track-list-poly evaluation/track_list_200_poly.txt \
      --track-list-mono evaluation/track_list_200_mono.txt --jobs "${JOBS[@]}"
    ;;
  generate)
    export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to an available physical GPU index}"
    "$PYTHON" -u "${RUNNER[@]}"
    ;;
  metrics)
    "$PYTHON" "${RUNNER[@]}" --merge
    "$PYTHON" -c 'import csv, sys; rows=list(csv.DictReader(open(sys.argv[1]))); expected=sys.argv[2:]; counts={e:len({r["sample_id"] for r in rows if r["experiment"]==e}) for e in expected}; print(counts); assert all(n==200 for n in counts.values()), "Generation incomplete: need 200 distinct outputs per condition"' "$OUT_ROOT/bridge_metrics_all.csv" "${JOBS[@]%:*}"
    mkdir -p "$OUT_ROOT/content" "$OUT_ROOT/ceiling"
    for case in poly mono; do
      source=flute_bassoon
      [[ "$case" == mono ]] && source=cello
      "$PYTHON" -u evaluation/compute_transcription_ceiling.py --track-list "evaluation/track_list_200_$case.txt" --ensemble "$source" --out-csv "$OUT_ROOT/ceiling/$case.csv"
    done
    for job in "${JOBS[@]}"; do
      experiment="${job%:*}"
      "$PYTHON" -u evaluation/compute_content_metrics.py --metrics-csv "$OUT_ROOT/bridge_metrics_all.csv" --experiments "$experiment" --out-csv "$OUT_ROOT/content/$experiment.csv"
    done
    ;;
  *) echo "Usage: bash evaluation/run_pitch_validation.sh {tracks|check|verify|smoke|generate|metrics}"; exit 1 ;;
esac
