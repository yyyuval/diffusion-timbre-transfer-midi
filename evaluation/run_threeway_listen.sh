#!/usr/bin/env bash
# The three-way listening test (evaluation/make_threeway_listen.py), start to
# finish, in one detachable run. Sampling only -- no training.
#
#   GPUS="5 6 7" setsid nohup bash evaluation/run_threeway_listen.sh > Outputs/threeway_listen.log 2>&1 < /dev/null &
#
#   default     choose 15 held-out tracks, generate the 45 clips, build the page   (~5 min)
#   FULL_S5=1   first also run swap_realtrained_genroll (real-trained sigma_max=5
#               models + generated roll) on all 1,000 set-B inputs, with its
#               EnCodec-FAD and note F1, so the page can show its FAD   (+~35 min)
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
FULL_S5="${FULL_S5:-0}"
OUT="Outputs/threeway_listen"
GEN="$OUT/gen"
B_ROOT="Outputs/WAV_files/merged_midi_test"
LISTS="evaluation/merged_test_lists"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="Outputs/threeway_listen_logs"
mkdir -p "$LOGDIR"

banner() { printf '\n%s\n== %s  (%s)\n%s\n' "======================================================================" "$*" "$(date '+%H:%M')" "======================================================================"; }
die()    { echo "FAILED: $*" >&2; exit 1; }

: "${GPUS:?set GPUS, e.g. GPUS=\"5 6 7\"}"
read -ra G <<< "$GPUS"
[ "${#G[@]}" -ge 3 ] || die "needs three GPUs"
echo "run_threeway_listen  $STAMP   GPUs ${G[*]}   FULL_S5=$FULL_S5"

# ------------------------------------------- optional: sigma 5 swap on all of set B
if [ "$FULL_S5" = 1 ]; then
  banner "sigma_max 5 swap on 1,000 set-B inputs, split over 3 GPUs"
  for attempt in 1 2 3; do
    pids=()
    for i in 0 1 2; do
      CUDA_VISIBLE_DEVICES="${G[$i]}" "$PYTHON" -u evaluation/run_bridge_batched.py \
        --out-root "$B_ROOT" --batch-size 4 --n 1000 --no-dpd --track-list-dir "$LISTS" \
        --shard "$i/3" --jobs swap_realtrained_genroll:5 >> "$LOGDIR/s5_gpu${G[$i]}_$STAMP.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    "$PYTHON" evaluation/run_bridge_batched.py --out-root "$B_ROOT" --merge > /dev/null 2>&1 || die "merge in $B_ROOT"
    "$PYTHON" -u evaluation/make_threeway_listen.py --link-s5 && break
    [ "$attempt" = 3 ] && die "sigma 5 swap incomplete after 3 attempts"
  done
  CUDA_VISIBLE_DEVICES="${G[0]}" "$PYTHON" -u evaluation/compute_encodec_fad.py \
    --real-dir "$B_ROOT/fad/poly/reference" \
    --gen-dirs "swap_realtrained_genroll=$B_ROOT/fad/poly/generated/swap_realtrained_genroll" \
    --out-csv "$B_ROOT/roll_swap/encodec_fad_s5.csv" --device cuda:0 > "$LOGDIR/s5_fad_$STAMP.log" 2>&1 \
    || die "EnCodec-FAD -- see $LOGDIR/s5_fad_$STAMP.log"
  CUDA_VISIBLE_DEVICES="${G[1]}" OMP_NUM_THREADS=8 "$PYTHON" -u evaluation/compute_content_metrics.py \
    --metrics-csv "$B_ROOT/bridge_metrics_all.csv" --experiments swap_realtrained_genroll \
    --track-list "$LISTS/content_poly.txt" --out-csv "$B_ROOT/content/swap_realtrained_genroll.csv" \
    --device cuda > "$LOGDIR/s5_content_$STAMP.log" 2>&1 || die "content metrics -- see $LOGDIR/s5_content_$STAMP.log"
  "$PYTHON" - <<'PY'
import csv, io
B = "Outputs/WAV_files/merged_midi_test"
f1 = lambda e: (lambda r: sum(float(x["transcription_f1"]) for x in r) / len(r))(
    list(csv.DictReader(io.open("%s/content/%s.csv" % (B, e), encoding="utf-8"))))
fad = [r for r in csv.DictReader(io.open(B + "/roll_swap/encodec_fad_s5.csv", encoding="utf-8"))
       if r["experiment"] == "swap_realtrained_genroll"][0]
print("sigma 5, real-trained + generated roll: note F1 %.3f, EnCodec-FAD %.2f (n=%s)"
      % (f1("swap_realtrained_genroll"), float(fad["fad"]), fad["n_generated"]))
print("  for reference, set B sigma 5: merged real %.3f / 4.53, merged generated %.3f / 3.82"
      % (f1("merged_real_midi"), f1("merged_generated_midi")))
PY
fi

# ------------------------------------------------------------- 1 the 15 tracks
banner "1  choose 15 held-out tracks"
"$PYTHON" -u evaluation/make_threeway_listen.py --select || die "track selection"

# ------------------------------------------------------------- 2 the 45 clips
banner "2  generate the 45 clips"
gen() {   # gpu list jobs...
  local gpu="$1" list="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u evaluation/run_bridge_batched.py --out-root "$GEN" \
    --batch-size 5 --no-dpd --track-list-poly "$list" --jobs "$@" >> "$LOGDIR/gen_gpu${gpu}_$STAMP.log" 2>&1
}
( gen "${G[0]}" "$OUT/tracks_all.txt" no_midi:5 ) &
p0=$!
( gen "${G[1]}" "$OUT/tracks_block1.txt" original_midi:5 original_midi_s100:100 \
  && gen "${G[1]}" "$OUT/tracks_block2.txt" merged_real_midi:5 merged_real_midi_s100:100 ) &
p1=$!
( gen "${G[2]}" "$OUT/tracks_block3.txt" swap_realtrained_genroll:5 swap_realtrained_genroll_s100:100 ) &
p2=$!
fail=0
for p in $p0 $p1 $p2; do wait "$p" || fail=1; done
[ "$fail" = 0 ] || die "generation -- see $LOGDIR/gen_gpu*_$STAMP.log"
"$PYTHON" evaluation/run_bridge_batched.py --out-root "$GEN" --merge > /dev/null 2>&1 || die "merge"

# -------------------------------------------------------------- 3 the page
banner "3  build the page"
CUDA_VISIBLE_DEVICES="${G[0]}" "$PYTHON" -u evaluation/make_threeway_listen.py --build || die "build"

banner "done -- download $OUT/threeway_listen.zip"
