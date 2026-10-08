#!/usr/bin/env bash
# Generate matched-sigma validation outputs, then compute all timbre metrics.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:?Set GPU to a free physical GPU index}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
OUT="${OUT_ROOT:-Outputs/WAV_files/clap_validation_1000}"
LISTS=evaluation/clap_1000
JOBS=(no_midi:5 original_midi:5 no_midi_s100:100 original_midi_s100:100
      mono_c2b_no_midi:5 mono_c2b_original_midi:5
      mono_c2b_no_midi_s100:100 mono_c2b_original_midi_s100:100)
mkdir -p "$OUT/logs" "$OUT/classifier"
"$PYTHON" evaluation/make_eval_pools.py --splits evaluation/splits.csv --out-dir tools/pools --cases poly mono
if [[ ! -f "$LISTS/frozen.ok" ]]; then
  "$PYTHON" evaluation/make_track_lists.py --pool-dir tools/pools --n-content 200 --n-fad 1000 --out-dir "$LISTS" --cases poly mono
  for case in poly mono; do
    for arch in midi nomidi; do
      file="$LISTS/track_list_${case}_${arch}.txt"
      [[ $(wc -l < "$file") -eq 1000 ]] || { echo "Need exactly 1000 entries in $file"; exit 1; }
    done
  done
  touch "$LISTS/frozen.ok"
fi
ARGS=(--out-root "$OUT" --track-list-dir "$LISTS" --n 1000 --batch-size "${BATCH_SIZE:-4}" --no-dpd --jobs "${JOBS[@]}")
echo "=== 1 Checkpoints and generation ==="
"$PYTHON" evaluation/run_bridge_batched.py "${ARGS[@]}" --list-ckpts
"$PYTHON" -u evaluation/run_bridge_batched.py "${ARGS[@]}"
"$PYTHON" evaluation/run_bridge_batched.py --out-root "$OUT" --merge
"$PYTHON" -c '
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for job in sys.argv[2:]:
 exp=job.split(":")[0]
 ids=[r["sample_id"] for r in rows if r["experiment"]==exp]
 if len(ids)!=1000 or len(set(ids))!=1000: raise SystemExit("Incomplete or duplicate outputs: "+exp)
print("Verified 1000 outputs in every condition")
' "$OUT/bridge_metrics_all.csv" "${JOBS[@]}"
test -s tools/pools/reference_poly.txt
test -s tools/pools/reference_mono.txt
echo "=== 2 CLAP FAD with clean target references ==="
"$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$OUT" --track-list-dir "$LISTS" --reference-from 'tools/pools/reference_{case}.txt' --device cuda:0
echo "=== 3 EnCodec FAD ==="
"$PYTHON" -u evaluation/compute_encodec_fad.py --out-root "$OUT" --out-csv "$OUT/fad/encodec_fad_results.csv" --device cuda:0
EXCLUDES=("$LISTS"/track_list_*.txt "$OUT"/fad/poly/reference_tracks.txt "$OUT"/fad/mono/reference_tracks.txt)
echo "=== 4 Instrument classifier on separate real tracks ==="
# Reuse only the checkpoint dedicated to this run and its exclusion manifest.
CKPT="$OUT/classifier/timbre_classifier.pt"
if [[ ! -f "$CKPT" ]]; then
 "$PYTHON" -u evaluation/train_timbre_classifier.py --ckpt "$CKPT" --exclude-track-lists "${EXCLUDES[@]}" --device cuda:0
fi
echo "Real-audio validation results:"
cat "${CKPT%.pt}.json"
echo "=== 5 Classifier scores on generated audio ==="
for case in poly mono; do
 TARGET=(violin cello)
 [[ "$case" == mono ]] && TARGET=(bassoon)
 for dir in "$OUT/fad/$case/generated/"*/; do
  "$PYTHON" -u evaluation/train_timbre_classifier.py --eval-only --ckpt "$CKPT" --eval-wav-dir "$dir" --target-instruments "${TARGET[@]}" --device cuda:0
 done
done
echo "=== ALL TIMBRE STAGES COMPLETED ==="
echo "Classifier results require review of real-audio validation before interpretation."
