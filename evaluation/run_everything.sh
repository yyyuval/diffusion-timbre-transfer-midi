#!/usr/bin/env bash
# Everything outstanding, in one detachable run.
#
#   bash evaluation/run_everything.sh            # all phases
#   PHASES="1 2 3" bash evaluation/run_everything.sh
#
#   GPUS   space-separated physical GPU indices. More than one splits the
#          8,000 generations across them -- the experiments write to separate
#          CSVs and separate wav names by design, so this is safe.
#   PHASES which phases to run (default all).
#
# Phases
#   1  preflight: pools, track lists, and the checks that must pass before
#      anything long starts
#   2  transcription ceiling measured on the TARGET instruments (cheap, ~30 min)
#   3  contamination: drop the tracks the target model trained on, restate the
#      existing pitch results (seconds, no GPU)
#   4  generation for the timbre metrics -- 8 conditions x 1000 clips, the long
#      pole. Resumable: re-running picks up where it stopped.
#   5  CLAP-FAD, EnCodec-FAD, instrument classifier
#   6  CLAP reference sensitivity: clean 365-clip reference vs padded to 1000
#   7  summary and the sanity checks worth reading first
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
GPUS="${GPUS:-0}"
PHASES="${PHASES:-1 2 3 4 5 6 7}"
SPLITS="${SPLITS:-evaluation/splits.csv}"
OUT_PITCH="${OUT_PITCH:-Outputs/WAV_files/pitch_validation_200}"
OUT_TIMBRE="${OUT_TIMBRE:-Outputs/WAV_files/clap_validation_1000}"
LISTS="${LISTS:-evaluation/clap_1000}"
POOLS="${POOLS:-tools/pools}"
N_CEILING="${N_CEILING:-200}"
N_FAD="${N_FAD:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RETRIES="${RETRIES:-3}"

JOBS=(no_midi:5 original_midi:5 no_midi_s100:100 original_midi_s100:100
      mono_c2b_no_midi:5 mono_c2b_original_midi:5
      mono_c2b_no_midi_s100:100 mono_c2b_original_midi_s100:100)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUT_TIMBRE/logs"
mkdir -p "$LOGDIR" "$POOLS"

banner() { printf '\n%s\n== %s\n%s\n' "======================================================================" "$*" "======================================================================"; }
have()   { for p in $PHASES; do [ "$p" = "$1" ] && return 0; done; return 1; }
die()    { echo "FAILED: $*" >&2; exit 1; }

echo "run_everything  $STAMP"
echo "  python   $PYTHON"
echo "  GPUs     $GPUS"
echo "  phases   $PHASES"
echo "  pitch    $OUT_PITCH"
echo "  timbre   $OUT_TIMBRE"
"$PYTHON" -c "import torch; print('  torch    %s, %d GPU(s) visible' % (torch.__version__, torch.cuda.device_count()))" \
  || die "the interpreter at $PYTHON cannot import torch. Set PYTHON= to the one that can."

# ---------------------------------------------------------------- 1 preflight
if have 1; then
  banner "1  preflight"
  [ -f "$SPLITS" ] || die "no $SPLITS"

  "$PYTHON" evaluation/make_eval_pools.py --splits "$SPLITS" --out-dir "$POOLS" --cases poly mono \
    || die "make_eval_pools"

  for case in poly mono; do
    for kind in pool shared reference; do
      f="$POOLS/${kind}_${case}.txt"
      [ -s "$f" ] || [ -s "$POOLS/${kind}_${case}_midi.txt" ] || echo "  note: $f missing or empty"
    done
  done

  if [ ! -f "$LISTS/frozen.ok" ]; then
    "$PYTHON" evaluation/make_track_lists.py --pool-dir "$POOLS" --n-content 200 \
      --n-fad "$N_FAD" --out-dir "$LISTS" --cases poly mono || die "make_track_lists"
    short=0
    for case in poly mono; do
      for arch in midi nomidi; do
        f="$LISTS/track_list_${case}_${arch}.txt"
        n=$(grep -cve '^\s*$' "$f" 2>/dev/null || echo 0)
        echo "  $f: $n"
        [ "$n" -ge "$N_FAD" ] || { echo "  SHORT -- wanted $N_FAD"; short=1; }
      done
    done
    [ "$short" -eq 0 ] || die "a track list came up short. The pools hold 1715-2211 tracks, so this
means make_accept rejected clips for missing stems or length. Inspect the list,
then either lower N_FAD or widen the accept rule -- do NOT start phase 4."
    touch "$LISTS/frozen.ok"
    echo "  lists frozen"
  else
    echo "  $LISTS/frozen.ok exists -- lists left alone"
  fi

  "$PYTHON" evaluation/run_bridge_batched.py --out-root "$OUT_TIMBRE" \
    --track-list-dir "$LISTS" --jobs "${JOBS[@]}" --list-ckpts || die "a checkpoint is missing"
fi

# ------------------------------------------------- 2 ceiling, target-side
if have 2; then
  banner "2  transcription ceiling on the TARGET instruments"
  # The ceiling already measured is the SOURCE side -- basic-pitch on
  # flute+bassoon and on cello. But the audio being scored is violin+cello and
  # bassoon, and basic-pitch's error depends on the timbre in front of it. The
  # same tracks cannot be used: a woodwind_track* has no violin stem on disk,
  # and CocoChorales never rendered the same piece for another ensemble. Held
  # out target material of the right instruments is the honest proxy, and
  # reference_{case}.txt is exactly that.
  mkdir -p "$OUT_PITCH/ceiling"
  export CUDA_VISIBLE_DEVICES="${GPUS%% *}"
  set -- "poly:violin_cello:$POOLS/reference_poly.txt" "mono:bassoon:$POOLS/reference_mono.txt"
  for spec in "$@"; do
    case_name="${spec%%:*}"; rest="${spec#*:}"; ens="${rest%%:*}"; list="${rest#*:}"
    [ -s "$list" ] || { echo "  $list missing -- run phase 1"; continue; }
    out="$OUT_PITCH/ceiling/${case_name}_target.csv"
    echo "  $case_name: $ens over $(grep -cve '^\s*$' "$list") held-out tracks, capped at $N_CEILING"
    "$PYTHON" -u evaluation/compute_transcription_ceiling.py --track-list "$list" \
      --ensemble "$ens" --limit "$N_CEILING" --device cuda:0 --out-csv "$out" \
      2>&1 | tee "$LOGDIR/ceiling_${case_name}_${STAMP}.log"
  done
  echo
  echo "  source-side (existing) vs target-side (new):"
  "$PYTHON" - "$OUT_PITCH/ceiling" <<'PY'
import sys, csv, io, glob, os
d = sys.argv[1]
for path in sorted(glob.glob(os.path.join(d, "*.csv"))):
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    if not rows:
        continue
    cols = [c for c in rows[0] if c.lower() in ("f1", "note_f1", "frame_f1", "precision", "recall")]
    def mean(c):
        vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        return sum(vals) / len(vals) if vals else float("nan")
    print("    %-28s n=%-4d %s" % (os.path.basename(path), len(rows),
          "  ".join("%s %.3f" % (c, mean(c)) for c in cols)))
PY
fi

# --------------------------------------------------------- 3 contamination
if have 3; then
  banner "3  contamination: tracks the target model trained on"
  CSVS=()
  for f in "$OUT_PITCH/content"/*.csv "$OUT_PITCH/bridge_metrics_all.csv"; do
    [ -f "$f" ] && case "$f" in *.clean.csv) ;; *) CSVS+=("$f") ;; esac
  done
  if [ "${#CSVS[@]}" -eq 0 ]; then
    echo "  no metric CSVs under $OUT_PITCH -- listing the tracks only"
    "$PYTHON" evaluation/filter_contaminated.py --splits "$SPLITS" --out-dir "$POOLS" --list-only
  else
    "$PYTHON" evaluation/filter_contaminated.py --splits "$SPLITS" --out-dir "$POOLS" \
      --csv "${CSVS[@]}" || die "filter_contaminated"
    clean="$OUT_PITCH/bridge_metrics_all.clean.csv"
    if [ -f "$clean" ]; then
      echo
      echo "  restating the mono comparison on the clean rows:"
      for pair in "mono_c2b_no_midi mono_c2b_original_midi" \
                  "mono_c2b_no_midi_s100 mono_c2b_original_midi_s100"; do
        set -- $pair
        "$PYTHON" evaluation/paired_stats.py --metrics_csv "$clean" --a "$1" --b "$2" \
          --metrics DPD JD 2>&1 | sed 's/^/    /'
      done
    fi
  fi
  echo
  echo "  poly is unaffected: flute and bassoon live only in woodwind_track*,"
  echo "  violin and cello only in string_track*. The sole exposure is the"
  echo "  random_track* ensemble, which can hold both a cello and a bassoon."
fi

# ------------------------------------------------------------ 4 generation
run_gpu_subset() {
  local gpu="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u evaluation/run_bridge_batched.py \
    --out-root "$OUT_TIMBRE" --track-list-dir "$LISTS" --n "$N_FAD" \
    --batch-size "$BATCH_SIZE" --no-dpd --jobs "$@" \
    > "$LOGDIR/gen_gpu${gpu}_${STAMP}.log" 2>&1
}

if have 4; then
  banner "4  generation  (8 conditions x $N_FAD clips)"
  read -ra GPU_ARR <<< "$GPUS"
  attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    echo "  attempt $attempt of $RETRIES  ($(date '+%H:%M:%S'))"
    pids=(); i=0
    for gpu in "${GPU_ARR[@]}"; do
      subset=()
      for ((j=i; j<${#JOBS[@]}; j+=${#GPU_ARR[@]})); do subset+=("${JOBS[$j]}"); done
      i=$((i+1))
      [ "${#subset[@]}" -eq 0 ] && continue
      echo "    GPU $gpu <- ${subset[*]}"
      run_gpu_subset "$gpu" "${subset[@]}" &
      pids+=($!)
    done
    ok=1
    for pid in "${pids[@]}"; do wait "$pid" || ok=0; done
    "$PYTHON" evaluation/run_bridge_batched.py --out-root "$OUT_TIMBRE" --merge >/dev/null 2>&1
    missing=$("$PYTHON" - "$OUT_TIMBRE/bridge_metrics_all.csv" "$N_FAD" "${JOBS[@]}" <<'PY'
import sys, csv, io, os
path, want = sys.argv[1], int(sys.argv[2])
jobs = [j.split(":")[0] for j in sys.argv[3:]]
rows = list(csv.DictReader(io.open(path, encoding="utf-8"))) if os.path.isfile(path) else []
bad = []
for exp in jobs:
    n = len({r["sample_id"] for r in rows if r["experiment"] == exp})
    if n < want:
        bad.append("%s %d/%d" % (exp, n, want))
print("; ".join(bad))
PY
)
    if [ -z "$missing" ]; then
      echo "  all conditions complete"
      break
    fi
    echo "  still incomplete: $missing"
    [ "$ok" -eq 1 ] && echo "  (every worker exited 0 -- check the logs in $LOGDIR)"
    attempt=$((attempt+1))
    [ "$attempt" -le "$RETRIES" ] && echo "  resuming in 60s" && sleep 60
  done
  [ -z "${missing:-}" ] || die "generation did not finish after $RETRIES attempts: $missing"
fi

# -------------------------------------------------------- 5 timbre metrics
if have 5; then
  banner "5  CLAP-FAD, EnCodec-FAD, instrument classifier"
  export CUDA_VISIBLE_DEVICES="${GPUS%% *}"
  export PYTHONUNBUFFERED=1
  mkdir -p "$OUT_TIMBRE/classifier"

  echo "-- CLAP-FAD (512-D) against the clean held-out reference"
  "$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$OUT_TIMBRE" \
    --track-list-dir "$LISTS" --reference-from "$POOLS/reference_{case}.txt" --device cuda:0 \
    2>&1 | tee "$LOGDIR/clap_clean_${STAMP}.log"

  echo "-- EnCodec-FAD (128-D) -- the paper's metric, and the one to lead with"
  "$PYTHON" -u evaluation/compute_encodec_fad.py --out-root "$OUT_TIMBRE" \
    --out-csv "$OUT_TIMBRE/fad/encodec_fad_results.csv" --device cuda:0 \
    2>&1 | tee "$LOGDIR/encodec_fad_${STAMP}.log"

  echo "-- instrument classifier, trained with every evaluation track excluded"
  CKPT="$OUT_TIMBRE/classifier/timbre_classifier.pt"
  EXCLUDES=("$LISTS"/track_list_*.txt "$OUT_TIMBRE"/fad/poly/reference_tracks.txt \
            "$OUT_TIMBRE"/fad/mono/reference_tracks.txt)
  if [ ! -f "$CKPT" ]; then
    "$PYTHON" -u evaluation/train_timbre_classifier.py --ckpt "$CKPT" \
      --exclude-track-lists "${EXCLUDES[@]}" --device cuda:0 \
      2>&1 | tee "$LOGDIR/classifier_train_${STAMP}.log"
  fi
  for case in poly mono; do
    TARGET=(violin cello); [ "$case" = mono ] && TARGET=(bassoon)
    for dir in "$OUT_TIMBRE/fad/$case/generated/"*/; do
      [ -d "$dir" ] || continue
      "$PYTHON" -u evaluation/train_timbre_classifier.py --eval-only --ckpt "$CKPT" \
        --eval-wav-dir "$dir" --target-instruments "${TARGET[@]}" --device cuda:0
    done
  done 2>&1 | tee "$LOGDIR/classifier_eval_${STAMP}.log"
fi

# ------------------------------------------- 6 CLAP reference sensitivity
if have 6; then
  banner "6  CLAP reference sensitivity"
  # CLAP embeddings are 512-D and the clean reference is 365 clips for poly,
  # 400 for mono. With n < D the reference covariance is singular: the distance
  # is still computable and still biased the same way for every condition, so
  # the RANKING holds, but the absolute value does not mean much. Padding to
  # 1000 from the wider target family buys a full-rank covariance at the cost
  # of letting in target material that is no longer strictly held out. Running
  # both says how much that choice is worth. If the two orderings agree, report
  # the clean one and cite this as the check.
  export CUDA_VISIBLE_DEVICES="${GPUS%% *}"
  "$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$OUT_TIMBRE" \
    --track-list-dir "$LISTS" --reference-from "$POOLS/reference_{case}.txt" \
    --pad-reference-to "$N_FAD" --out-csv "$OUT_TIMBRE/fad/clap_fad_padded.csv" --device cuda:0 \
    2>&1 | tee "$LOGDIR/clap_padded_${STAMP}.log"
fi

# ---------------------------------------------------------------- 7 summary
if have 7; then
  banner "7  summary"
  "$PYTHON" evaluation/summarize_200.py --out-root "$OUT_TIMBRE" \
    --ceiling-poly "$OUT_PITCH/ceiling/poly_target.csv" \
    --ceiling-mono "$OUT_PITCH/ceiling/mono_target.csv" 2>&1 | tail -40

  echo
  echo "-- read these before anything else --"
  CJSON="$OUT_TIMBRE/classifier/timbre_classifier.json"
  if [ -f "$CJSON" ]; then
    "$PYTHON" - "$CJSON" <<'PY'
import sys, json
m = json.load(open(sys.argv[1]))
f1 = m.get("val_macro_f1")
print("  classifier macro-F1 on held-out REAL stems: %s" % f1)
if isinstance(f1, (int, float)) and f1 < 0.90:
    print("  *** below 0.90. A classifier that cannot identify real instruments")
    print("  *** says nothing about generated ones. Do not report stage-5 numbers")
    print("  *** until this is understood.")
PY
  else
    echo "  no classifier json at $CJSON"
  fi
  for f in "$OUT_TIMBRE/fad/clap_fad_200.csv" "$OUT_TIMBRE/fad/clap_fad_padded.csv" \
           "$OUT_TIMBRE/fad/encodec_fad_results.csv"; do
    [ -f "$f" ] && { echo "  $f"; head -20 "$f" | sed 's/^/    /'; }
  done
  echo
  echo "  logs: $LOGDIR"
fi

banner "done  $(date '+%Y-%m-%d %H:%M:%S')"
