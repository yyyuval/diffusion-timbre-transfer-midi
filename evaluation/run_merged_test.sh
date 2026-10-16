#!/usr/bin/env bash
# The merged-MIDI test, start to finish, in one detachable run.
#
#   setsid nohup bash evaluation/run_merged_test.sh > Outputs/merged_test.log 2>&1 < /dev/null &
#
#   GPUS     GPU indices to use. Default: the N_GPUS with the least memory in
#            use at launch. Inference needs ~6-8 GB, not a whole card.
#   N_GPUS   how many to pick when GPUS is unset (default 3).
#   PHASES   which phases to run (default all), e.g. PHASES="6 7".
#
# Phases
#   1  recover the merged models' train/validation split. They landed on a
#      different one from the stacked models -- a different parameter count
#      draws the RNG differently -- so the old test lists are ~80% their
#      training data. CPU only.
#   2  pools and track lists: clips held out by the stacked AND the merged
#      models, 200 shared for the paired content metrics, 1000 per
#      architecture for FAD. Frozen after the first run.
#   3  checkpoints resolve
#   4  generation: the 200 shared clips with DPD/JD, then up to 1000 without.
#      Resumable; retried up to RETRIES times.
#   5  content metrics on the 200 shared clips
#   6  timbre: CLAP-FAD (also builds the reference), EnCodec-FAD, classifier
#   7  REPORT.md, results.csv and the listening folder
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
N_GPUS="${N_GPUS:-3}"
PHASES="${PHASES:-1 2 3 4 5 6 7}"
OUT="${OUT_ROOT:-Outputs/WAV_files/merged_midi_test}"
POOLS="${POOLS:-tools/pools_merged}"
LISTS="${LISTS:-evaluation/merged_test_lists}"
SPLITS_MERGED="${SPLITS_MERGED:-evaluation/splits_merged.csv}"
CLASSIFIER="${CLASSIFIER:-Outputs/WAV_files/clap_validation_1000/classifier/timbre_classifier.pt}"
N_CONTENT=200
N_FAD="${N_FAD:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RETRIES="${RETRIES:-3}"

JOBS=(original_midi:5 original_midi_s100:100
      merged_real_midi:5 merged_real_midi_s100:100
      merged_generated_midi:5 merged_generated_midi_s100:100)
EXPS=("${JOBS[@]%:*}")

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

banner() { printf '\n%s\n== %s\n%s\n' "======================================================================" "$*" "======================================================================"; }
have()   { for p in $PHASES; do [ "$p" = "$1" ] && return 0; done; return 1; }
die()    { echo "FAILED: $*" >&2; exit 1; }

if [ -z "${GPUS:-}" ]; then
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | sort -t, -k2 -n | head -n "$N_GPUS" | cut -d, -f1 | tr -d ' ' | tr '\n' ' ')"
fi
read -ra GPU_ARR <<< "$GPUS"
FIRST_GPU="${GPU_ARR[0]}"

echo "run_merged_test  $STAMP"
echo "  GPUs    ${GPU_ARR[*]}"
echo "  out     $OUT"
echo "  phases  $PHASES"

# ------------------------------------------------------------------ 1 splits
if have 1; then
  banner "1  the merged models' train/validation split"
  if [ -s "$SPLITS_MERGED" ]; then
    echo "  $SPLITS_MERGED exists -- kept (delete it to recover again)"
  else
    CUDA_VISIBLE_DEVICES="" "$PYTHON" evaluation/dump_splits.py \
      --only flute_bassoon_merged violin_cello_merged flute_bassoon_merged_bp violin_cello_merged_bp \
      --out "$SPLITS_MERGED" 2>&1 | tee "$LOGDIR/splits_$STAMP.log" | grep -v "^Seed set"
    grep -q "DIFFERENT SPLITS" "$LOGDIR/splits_$STAMP.log" && \
      die "the real- and generated-MIDI runs did not share a split; their pools must be separated"
    [ -s "$SPLITS_MERGED" ] || die "dump_splits wrote nothing"
  fi
fi

# ---------------------------------------------------------- 2 pools, lists
if have 2; then
  banner "2  pools and track lists"
  # To a log, not through `head`: under pipefail, head closing the pipe early
  # would fail the whole step.
  "$PYTHON" evaluation/make_eval_pools.py --splits evaluation/splits.csv "$SPLITS_MERGED" \
    --architectures midi merged --cases poly --out-dir "$POOLS" \
    > "$LOGDIR/pools_$STAMP.log" 2>&1 || { cat "$LOGDIR/pools_$STAMP.log"; die "make_eval_pools"; }
  sed -n '1,8p' "$LOGDIR/pools_$STAMP.log"
  if [ ! -f "$LISTS/frozen.ok" ]; then
    "$PYTHON" evaluation/make_track_lists.py --pool-dir "$POOLS" --architectures midi merged \
      --cases poly --n-content "$N_CONTENT" --n-fad "$N_FAD" --out-dir "$LISTS" || die "make_track_lists"
    for arch in midi merged; do
      n=$(grep -cve '^\s*$' "$LISTS/track_list_poly_$arch.txt")
      echo "  track_list_poly_$arch.txt: $n"
      [ "$n" -ge "$N_CONTENT" ] || die "fewer than $N_CONTENT clips for $arch"
    done
    head -n "$N_CONTENT" "$LISTS/track_list_poly_merged.txt" > "$LISTS/content_poly.txt"
    cmp -s "$LISTS/content_poly.txt" <(head -n "$N_CONTENT" "$LISTS/track_list_poly_midi.txt") \
      || die "the first $N_CONTENT clips differ between the two lists -- the content test would not be paired"
    touch "$LISTS/frozen.ok"
    echo "  frozen: $LISTS"
  else
    echo "  $LISTS/frozen.ok exists -- lists kept"
  fi
fi

RUNNER=(evaluation/run_bridge_batched.py --out-root "$OUT" --track-list-dir "$LISTS"
        --batch-size "$BATCH_SIZE")

# ------------------------------------------------------------- 3 checkpoints
if have 3; then
  banner "3  checkpoints"
  "$PYTHON" "${RUNNER[@]}" --jobs "${JOBS[@]}" --list-ckpts || die "a checkpoint did not resolve"
fi

# -------------------------------------------------------------- 4 generation
generate() {   # $1 = n, then extra runner flags
  local n="$1"; shift
  local attempt=1 pids i gpu subset missing
  while [ "$attempt" -le "$RETRIES" ]; do
    pids=(); i=0
    for gpu in "${GPU_ARR[@]}"; do
      subset=()
      for ((j=i; j<${#JOBS[@]}; j+=${#GPU_ARR[@]})); do subset+=("${JOBS[$j]}"); done
      i=$((i+1))
      [ "${#subset[@]}" -eq 0 ] && continue
      echo "    GPU $gpu <- ${subset[*]}  (n=$n $*)"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "${RUNNER[@]}" --n "$n" "$@" --jobs "${subset[@]}" \
        > "$LOGDIR/gen_n${n}_gpu${gpu}_$STAMP.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    "$PYTHON" "${RUNNER[@]}" --merge >/dev/null 2>&1
    missing=$("$PYTHON" - "$OUT/bridge_metrics_all.csv" "$n" "${EXPS[@]}" <<'PY'
import csv, io, os, sys
path, want = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(io.open(path, encoding="utf-8"))) if os.path.isfile(path) else []
bad = []
for exp in sys.argv[3:]:
    n = len({r["sample_id"] for r in rows if r["experiment"] == exp})
    if n < want:
        bad.append("%s %d/%d" % (exp, n, want))
print("; ".join(bad))
PY
)
    [ -z "$missing" ] && { echo "  all conditions have $n clips"; return 0; }
    echo "  attempt $attempt incomplete: $missing"
    attempt=$((attempt+1))
  done
  die "generation (n=$n) incomplete after $RETRIES attempts: $missing -- see $LOGDIR"
}

if have 4; then
  banner "4  generation"
  echo "  the $N_CONTENT shared clips, with DPD and JD"
  generate "$N_CONTENT"
  echo "  the rest, up to $N_FAD per condition, for FAD and the classifier"
  generate "$N_FAD" --no-dpd
fi

# ----------------------------------------------------------------- 5 content
if have 5; then
  banner "5  content metrics on the $N_CONTENT shared clips"
  mkdir -p "$OUT/content"
  for exp in "${EXPS[@]}"; do
    out="$OUT/content/$exp.csv"
    [ -s "$out" ] && { echo "  $exp: exists, kept"; continue; }
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON" -u evaluation/compute_content_metrics.py \
      --metrics-csv "$OUT/bridge_metrics_all.csv" --experiments "$exp" \
      --track-list "$LISTS/content_poly.txt" --out-csv "$out" --device cuda \
      > "$LOGDIR/content_${exp}_$STAMP.log" 2>&1 || die "content metrics for $exp -- see $LOGDIR"
    echo "  $exp: $(($(wc -l < "$out") - 1)) clips"
  done
fi

# ------------------------------------------------------------------ 6 timbre
if have 6; then
  banner "6  timbre: FAD and the instrument classifier"
  export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
  "$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$OUT" --cases poly \
    --track-list-dir "$LISTS" --reference-from "$POOLS/reference_{case}.txt" --device cuda:0 \
    > "$LOGDIR/clap_$STAMP.log" 2>&1 || die "CLAP-FAD -- see $LOGDIR/clap_$STAMP.log"
  tail -n 12 "$LOGDIR/clap_$STAMP.log"
  "$PYTHON" -u evaluation/compute_encodec_fad.py --out-root "$OUT" --cases poly \
    --out-csv "$OUT/fad/encodec_fad_results.csv" --device cuda:0 \
    > "$LOGDIR/encodec_$STAMP.log" 2>&1 || die "EnCodec-FAD -- see $LOGDIR/encodec_$STAMP.log"
  tail -n 10 "$LOGDIR/encodec_$STAMP.log"
  [ -f "$CLASSIFIER" ] || die "no classifier at $CLASSIFIER"
  for exp in "${EXPS[@]}"; do
    dir="$OUT/fad/poly/generated/$exp/"
    [ -d "$dir" ] || { echo "  no $dir -- skipped"; continue; }
    "$PYTHON" -u evaluation/train_timbre_classifier.py --eval-only --ckpt "$CLASSIFIER" \
      --eval-wav-dir "$dir" --target-instruments violin cello --device cuda:0 \
      > "$LOGDIR/classifier_${exp}_$STAMP.log" 2>&1 || die "classifier on $exp"
    echo "  $exp: $(grep -o 'accuracy: [0-9.]*' "$LOGDIR/classifier_${exp}_$STAMP.log" | tail -1)"
  done
  unset CUDA_VISIBLE_DEVICES
fi

# ------------------------------------------------------------------ 7 report
if have 7; then
  banner "7  report"
  "$PYTHON" evaluation/report_merged_midi.py --out-root "$OUT" || die "report"
fi

banner "done  $(date '+%Y-%m-%d %H:%M:%S')"
