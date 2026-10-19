#!/usr/bin/env bash
# The roll-swap test, start to finish, in one detachable run. Sampling only --
# no training.
#
#   GPUS="5 6 7" setsid nohup bash evaluation/run_roll_swap.sh > Outputs/roll_swap.log 2>&1 < /dev/null &
#
# merged_generated_midi_s100 has the worst EnCodec-FAD of any MIDI model, 6.13
# against 3.69 with real rolls. Is that the roll or the weights? Each sigma_max
# = 100 merged bridge runs again with the OTHER kind of roll:
#   swap_realtrained_genroll_s100   real-trained models + generated roll
#   swap_gentrained_realroll_s100   generated-trained models + real roll
# on the same 1,000 set-B inputs as the originals (handoff 100), so their
# EnCodec-FAD is directly comparable with the book's.
#
# Phases (PHASES="..." to run a subset; every phase is resumable)
#   1  generation: both swaps, each split across all GPUs (--shard). While it
#      runs, on the CPU: compare_rolls.py, and the originals' transcriptions
#      for the false-note diagnosis, cached so phase 2 only adds the new clips.
#   2  in parallel: content metrics for each swap (200 paired clips), EnCodec
#      embeddings + FAD, false-note diagnosis of all four conditions
#   3  REPORT.md -> Outputs/WAV_files/merged_midi_test/roll_swap/
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
PHASES="${PHASES:-1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RETRIES="${RETRIES:-3}"
# basic-pitch runs on the CPU; cap each CPU job so it does not starve the others
CPU_THREADS="${CPU_THREADS:-8}"
CPU=(env OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS")
B_ROOT="Outputs/WAV_files/merged_midi_test"
LISTS="evaluation/merged_test_lists"
OUT="$B_ROOT/roll_swap"
LOGDIR="$OUT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
N_FAD=1000
SWAPS=(swap_realtrained_genroll_s100 swap_gentrained_realroll_s100)
ORIGINALS=(merged_real_midi_s100 merged_generated_midi_s100)
mkdir -p "$LOGDIR"

banner() { printf '\n%s\n== %s  (%s)\n%s\n' "======================================================================" "$*" "$(date '+%H:%M')" "======================================================================"; }
have()   { for p in $PHASES; do [ "$p" = "$1" ] && return 0; done; return 1; }
die()    { echo "FAILED: $*" >&2; exit 1; }

: "${GPUS:?set GPUS, e.g. GPUS=\"5 6 7\"}"
read -ra GPU_ARR <<< "$GPUS"
N_SHARDS="${#GPU_ARR[@]}"
echo "run_roll_swap  $STAMP   GPUs ${GPU_ARR[*]}   ${SWAPS[*]}"

for f in "$LISTS/frozen.ok" "$LISTS/track_list_poly_merged.txt" "$LISTS/content_poly.txt" \
         "$B_ROOT/fad/poly/reference_encodec.npz" "$B_ROOT/fad/encodec_fad_results.csv" \
         midi_cache_wood_fps75_gt_merged midi_cache_wood_fps75_bp_merged \
         midi_cache_string_fps75_gt_merged midi_cache_string_fps75_bp_merged; do
  [ -e "$f" ] || die "missing $f -- the merged-MIDI test must have run first"
done
for exp in "${ORIGINALS[@]}"; do
  [ -s "$B_ROOT/content/$exp.csv" ] || die "missing $B_ROOT/content/$exp.csv"
  [ -s "$B_ROOT/fad/poly/generated/${exp}_encodec.npz" ] || die "missing the cached EnCodec embeddings of $exp"
done

count_rows() {   # experiment -> clips in bridge_metrics_all.csv
  "$PYTHON" - "$B_ROOT/bridge_metrics_all.csv" "$1" <<'PY'
import csv, io, os, sys
path, exp = sys.argv[1], sys.argv[2]
rows = csv.DictReader(io.open(path, encoding="utf-8")) if os.path.isfile(path) else []
print(len({r["sample_id"] for r in rows if r["experiment"] == exp}))
PY
}

# ---------------------------------------------------------------- 1 generation
if have 1; then
  banner "1  generation: ${#SWAPS[@]} swaps x $N_FAD clips, each split over $N_SHARDS GPUs"

  # CPU side jobs, while the GPUs generate
  "${CPU[@]}" "$PYTHON" -u evaluation/compare_rolls.py --wood-tracks "$LISTS/track_list_poly_merged.txt" \
    --out-dir "$OUT" > "$LOGDIR/compare_rolls_$STAMP.log" 2>&1 &
  ROLLS_PID=$!
  ( "$PYTHON" -u evaluation/report_roll_swap.py --prepare --original-only \
      && "${CPU[@]}" "$PYTHON" -u evaluation/diagnose_false_notes.py --root "$B_ROOT" \
           --bridge-csv "$OUT/bridge_rows_original.csv" --out-csv "$OUT/false_notes_original.csv" \
  ) > "$LOGDIR/notes_original_$STAMP.log" 2>&1 &
  NOTES_PID=$!

  jobs_spec=()
  for s in "${SWAPS[@]}"; do jobs_spec+=("$s:100"); done
  attempt=1
  while :; do
    pids=()
    for i in "${!GPU_ARR[@]}"; do
      gpu="${GPU_ARR[$i]}"
      echo "  [$(date '+%H:%M')] GPU $gpu <- shard $i/$N_SHARDS of ${jobs_spec[*]}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u evaluation/run_bridge_batched.py \
        --out-root "$B_ROOT" --batch-size "$BATCH_SIZE" --n "$N_FAD" --no-dpd \
        --track-list-dir "$LISTS" --shard "$i/$N_SHARDS" --jobs "${jobs_spec[@]}" \
        >> "$LOGDIR/gen_gpu${gpu}_$STAMP.log" 2>&1 \
        || echo "  [$(date '+%H:%M')] GPU $gpu exited non-zero -- see $LOGDIR/gen_gpu${gpu}_$STAMP.log" &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    "$PYTHON" evaluation/run_bridge_batched.py --out-root "$B_ROOT" --merge > "$LOGDIR/merge_$STAMP.log" 2>&1 \
      || die "merging failed (clips are safe in $B_ROOT/csv/): $(tail -n 1 "$LOGDIR/merge_$STAMP.log")"
    missing=""
    for s in "${SWAPS[@]}"; do
      n="$(count_rows "$s")"
      echo "  $s: $n/$N_FAD clips"
      [ "$n" -ge "$N_FAD" ] || missing+=" $s"
    done
    [ -z "$missing" ] && break
    attempt=$((attempt + 1))
    [ "$attempt" -le "$RETRIES" ] || die "generation incomplete after $RETRIES attempts:$missing"
    echo "  retrying:$missing"
  done

  wait "$ROLLS_PID" || die "compare_rolls.py -- see $LOGDIR/compare_rolls_$STAMP.log"
  wait "$NOTES_PID" || die "false-note cache for the originals -- see $LOGDIR/notes_original_$STAMP.log"
  echo "  roll comparison and the originals' transcriptions done"
fi

# ------------------------------------------------------------------ 2 metrics
if have 2; then
  banner "2  content metrics, EnCodec-FAD and the false-note diagnosis, in parallel"
  "$PYTHON" -u evaluation/report_roll_swap.py --prepare > "$LOGDIR/prepare_$STAMP.log" 2>&1 \
    || die "prepare -- see $LOGDIR/prepare_$STAMP.log"
  mkdir -p "$B_ROOT/content"
  pids=()
  for i in "${!SWAPS[@]}"; do
    exp="${SWAPS[$i]}"
    gpu="${GPU_ARR[$((i % N_SHARDS))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${CPU[@]}" "$PYTHON" -u evaluation/compute_content_metrics.py \
      --metrics-csv "$B_ROOT/bridge_metrics_all.csv" --experiments "$exp" \
      --track-list "$LISTS/content_poly.txt" --out-csv "$B_ROOT/content/$exp.csv" --device cuda \
      > "$LOGDIR/content_${exp}_$STAMP.log" 2>&1 &
    pids+=($!)
  done
  gen_dirs=()
  for exp in "${ORIGINALS[@]}" "${SWAPS[@]}"; do gen_dirs+=("$exp=$B_ROOT/fad/poly/generated/$exp"); done
  CUDA_VISIBLE_DEVICES="${GPU_ARR[$((N_SHARDS - 1))]}" "$PYTHON" -u evaluation/compute_encodec_fad.py \
    --real-dir "$B_ROOT/fad/poly/reference" --gen-dirs "${gen_dirs[@]}" \
    --out-csv "$OUT/encodec_fad.csv" --device cuda:0 > "$LOGDIR/encodec_$STAMP.log" 2>&1 &
  pids+=($!)
  "${CPU[@]}" "$PYTHON" -u evaluation/diagnose_false_notes.py --root "$B_ROOT" \
    --bridge-csv "$OUT/bridge_rows_all.csv" --out-csv "$OUT/false_notes.csv" \
    > "$LOGDIR/notes_all_$STAMP.log" 2>&1 &
  pids+=($!)
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" = 0 ] || die "a phase-2 job failed -- see $LOGDIR/*_$STAMP.log"
  for exp in "${SWAPS[@]}"; do
    echo "  content $exp: $(($(wc -l < "$B_ROOT/content/$exp.csv") - 1)) clips"
  done
fi

# ------------------------------------------------------------------- 3 report
if have 3; then
  banner "3  report"
  "$PYTHON" -u evaluation/report_roll_swap.py --report || die "report"
fi

banner "done -- $OUT/REPORT.md"
