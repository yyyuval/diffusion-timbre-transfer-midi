#!/usr/bin/env bash
# The sigma_handoff sweep: every sigma_max=100 model sampled at handoff 5, 20
# and 50 (100 already exists), as in the paper's Table III. Detachable and
# resumable.
#
#   GPUS="3 4 0" setsid nohup bash evaluation/run_handoff_sweep.sh > Outputs/handoff_sweep.log 2>&1 < /dev/null &
#
# The handoff is where the source model's reverse diffusion hands over to the
# target model: low keeps more of the input (melody), high gives the target
# model more room (timbre). It changes no weights, so this is sampling only --
# no training.
#
# Each model runs on the SAME test clips as the summary table, in the same
# result folder, so its handoff=100 row is the existing one, not a new run:
#   set A, content  Outputs/WAV_files/pitch_validation_200   (200 paired clips)
#   set A, timbre   Outputs/WAV_files/clap_validation_1000   (1,000 per model)
#   set B, both     Outputs/WAV_files/merged_midi_test       (200 paired / 1,000)
# DPD and JD only for the monophonic models -- they keep one note per moment,
# which is meaningless with two instruments.
#
# Generation is a queue: each GPU takes the next unclaimed task until none are
# left, so a fast GPU is never idle while a slow one still has a backlog.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
PHASES="${PHASES:-1 2 3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RETRIES="${RETRIES:-3}"
HANDOFFS="5,20,50"
A_CONTENT="Outputs/WAV_files/pitch_validation_200"
A_TIMBRE="Outputs/WAV_files/clap_validation_1000"
B_ROOT="Outputs/WAV_files/merged_midi_test"
CLASSIFIER="$A_TIMBRE/classifier/timbre_classifier.pt"
LOGDIR="Outputs/handoff_sweep_logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

POLY_A=(no_midi_s100 original_midi_s100)
MONO_A=(mono_c2b_no_midi_s100 mono_c2b_original_midi_s100)
POLY_B=(original_midi_s100 merged_real_midi_s100 merged_generated_midi_s100)

banner() { printf '\n%s\n== %s  (%s)\n%s\n' "======================================================================" "$*" "$(date '+%H:%M')" "======================================================================"; }
have()   { for p in $PHASES; do [ "$p" = "$1" ] && return 0; done; return 1; }
die()    { echo "FAILED: $*" >&2; exit 1; }

: "${GPUS:?set GPUS, e.g. GPUS=\"3 4 0\"}"
read -ra GPU_ARR <<< "$GPUS"
FIRST_GPU="${GPU_ARR[0]}"
echo "run_handoff_sweep  $STAMP   GPUs ${GPU_ARR[*]}   handoffs $HANDOFFS (+100 existing)"

for f in evaluation/track_list_200_poly.txt evaluation/track_list_200_mono.txt \
         evaluation/clap_1000/frozen.ok evaluation/merged_test_lists/frozen.ok "$CLASSIFIER"; do
  [ -e "$f" ] || die "missing $f -- the earlier evaluations must have run"
done

# One task per (folder, model): the runner loads the model pair once and does
# all three handoffs. Bigger tasks first, so the queue ends evenly.
TASKS=()
add() { TASKS+=("$*"); }   # root|n|extra flags|job
for m in "${POLY_A[@]}" "${MONO_A[@]}"; do
  add "$A_TIMBRE|1000|--no-dpd --track-list-dir evaluation/clap_1000|$m:$HANDOFFS"
done
for m in "${POLY_B[@]}"; do
  add "$B_ROOT|1000|--no-dpd --track-list-dir evaluation/merged_test_lists|$m:$HANDOFFS"
done
for m in "${POLY_A[@]}"; do
  add "$A_CONTENT|200|--no-dpd --track-list-poly evaluation/track_list_200_poly.txt --track-list-mono evaluation/track_list_200_mono.txt|$m:$HANDOFFS"
done
for m in "${MONO_A[@]}"; do   # with DPD and JD
  add "$A_CONTENT|200|--track-list-poly evaluation/track_list_200_poly.txt --track-list-mono evaluation/track_list_200_mono.txt|$m:$HANDOFFS"
done

expected_rows() {   # root n experiments... -> prints what is still missing
  "$PYTHON" - "$@" <<'PY'
import csv, io, os, sys
root, want = sys.argv[1], int(sys.argv[2])
path = os.path.join(root, "bridge_metrics_all.csv")
rows = list(csv.DictReader(io.open(path, encoding="utf-8"))) if os.path.isfile(path) else []
bad = []
for exp in sys.argv[3:]:
    n = len({r["sample_id"] for r in rows if r["experiment"] == exp})
    if n < want:
        bad.append("%s/%s %d/%d" % (os.path.basename(root), exp, n, want))
print("; ".join(bad))
PY
}

sweep_exps() { local m h; for m in "$@"; do for h in ${HANDOFFS//,/ }; do echo "${m}_h${h}"; done; done; }

# ---------------------------------------------------------------- 1 generation
if have 1; then
  banner "1  generation: ${#TASKS[@]} tasks over ${#GPU_ARR[@]} GPUs"
  attempt=1
  while :; do
    claims="$LOGDIR/claims_${STAMP}_$attempt"
    rm -rf "$claims"; mkdir -p "$claims"
    pids=()
    for gpu in "${GPU_ARR[@]}"; do
      (
        for i in "${!TASKS[@]}"; do
          mkdir "$claims/$i" 2>/dev/null || continue   # atomic claim
          IFS='|' read -r root n flags job <<< "${TASKS[$i]}"
          echo "  [$(date '+%H:%M')] GPU $gpu <- $job  ($(basename "$root"), n=$n)"
          # shellcheck disable=SC2086
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u evaluation/run_bridge_batched.py \
            --out-root "$root" --batch-size "$BATCH_SIZE" --n "$n" $flags --jobs "$job" \
            >> "$LOGDIR/gen_gpu${gpu}_$STAMP.log" 2>&1 \
            || echo "  [$(date '+%H:%M')] GPU $gpu: $job exited non-zero -- see $LOGDIR/gen_gpu${gpu}_$STAMP.log"
        done
      ) &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    for root in "$A_CONTENT" "$A_TIMBRE" "$B_ROOT"; do
      # Not silenced: a merge that cannot write -- bridge_metrics_all.csv owned
      # by another user -- leaves the old file in place, and every clip just
      # generated then looks missing.
      "$PYTHON" evaluation/run_bridge_batched.py --out-root "$root" --merge > "$LOGDIR/merge_$STAMP.log" 2>&1 \
        || die "merging $root failed (clips are safe in $root/csv/): $(tail -n 1 "$LOGDIR/merge_$STAMP.log")"
    done
    missing="$(expected_rows "$A_TIMBRE" 1000 $(sweep_exps "${POLY_A[@]}" "${MONO_A[@]}"))"
    missing+=" $(expected_rows "$B_ROOT" 1000 $(sweep_exps "${POLY_B[@]}"))"
    missing+=" $(expected_rows "$A_CONTENT" 200 $(sweep_exps "${POLY_A[@]}" "${MONO_A[@]}"))"
    missing="$(echo "$missing" | xargs)"
    [ -z "$missing" ] && { echo "  every sweep condition is complete"; break; }
    echo "  attempt $attempt incomplete: $missing"
    attempt=$((attempt + 1))
    [ "$attempt" -le "$RETRIES" ] || die "generation incomplete after $RETRIES attempts"
  done
fi

# --------------------------------------------------------------- 2 content
content() {   # root list experiments...
  local root="$1" list="$2" exp out; shift 2
  mkdir -p "$root/content"
  for exp in "$@"; do
    out="$root/content/$exp.csv"
    [ -s "$out" ] && continue
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON" -u evaluation/compute_content_metrics.py \
      --metrics-csv "$root/bridge_metrics_all.csv" --experiments "$exp" --track-list "$list" \
      --out-csv "$out" --device cuda > "$LOGDIR/content_${exp}_$STAMP.log" 2>&1 \
      || die "content metrics for $exp -- see $LOGDIR/content_${exp}_$STAMP.log"
    echo "  $(basename "$root")/$exp: $(($(wc -l < "$out") - 1)) clips"
  done
}
if have 2; then
  banner "2  content metrics"
  content "$A_CONTENT" evaluation/track_list_200_poly.txt $(sweep_exps "${POLY_A[@]}")
  content "$A_CONTENT" evaluation/track_list_200_mono.txt $(sweep_exps "${MONO_A[@]}")
  content "$B_ROOT" evaluation/merged_test_lists/content_poly.txt $(sweep_exps "${POLY_B[@]}")
fi

# ---------------------------------------------------------------- 3 timbre
timbre() {   # root cases reference-pattern list-dir
  local root="$1" cases="$2" ref="$3" lists="$4" case dir target
  export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
  # shellcheck disable=SC2086
  "$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$root" --cases $cases \
    --track-list-dir "$lists" --reference-from "$ref" --device cuda:0 \
    > "$LOGDIR/clap_$(basename "$root")_$STAMP.log" 2>&1 || die "CLAP-FAD in $root"
  # shellcheck disable=SC2086
  "$PYTHON" -u evaluation/compute_encodec_fad.py --out-root "$root" --cases $cases \
    --out-csv "$root/fad/encodec_fad_results.csv" --device cuda:0 \
    > "$LOGDIR/encodec_$(basename "$root")_$STAMP.log" 2>&1 || die "EnCodec-FAD in $root"
  for case in $cases; do
    target="violin cello"; [ "$case" = mono ] && target="bassoon"
    for dir in "$root/fad/$case/generated/"*_h*/; do
      [ -d "$dir" ] || continue
      [ -s "$dir/timbre_classifier_scores.csv" ] && continue
      # shellcheck disable=SC2086
      "$PYTHON" -u evaluation/train_timbre_classifier.py --eval-only --ckpt "$CLASSIFIER" \
        --eval-wav-dir "$dir" --target-instruments $target --device cuda:0 \
        > "$LOGDIR/classifier_$(basename "$dir")_$STAMP.log" 2>&1 || die "classifier on $dir"
      echo "  $(basename "$root")/$(basename "$dir"): classified"
    done
  done
  unset CUDA_VISIBLE_DEVICES
}
if have 3; then
  banner "3  FAD and the instrument classifier"
  timbre "$A_TIMBRE" "poly mono" "tools/pools/reference_{case}.txt" evaluation/clap_1000
  timbre "$B_ROOT" "poly" "tools/pools_merged/reference_{case}.txt" evaluation/merged_test_lists
fi

banner "done -- now: venv_yuval/bin/python evaluation/collect_handoff_sweep.py"
