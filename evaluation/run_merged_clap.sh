#!/usr/bin/env bash
# CLAP-FAD with a full-rank reference, run AFTER run_merged_test.sh finishes.
#
#   GPU=3 bash evaluation/run_merged_clap.sh
#
# The shared reference -- real violin + cello that BOTH target models held out
# -- is ~280 clips, under CLAP's 512 dimensions, so its covariance is singular.
# Each target model alone held out ~1,800. So each architecture is scored here
# against a reference of 1,000 clips its OWN target model never saw.
#
# The two references are different samples of the same population, so they
# add a little sampling noise to a comparison between architectures. That
# noise is measured rather than assumed: the CLAP-FAD between the two
# references themselves is the floor, written to fad/clap_reference_floor.csv
# and shown in REPORT.md. A difference between conditions well above it is
# real.
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-venv_yuval/bin/python}"
OUT="${OUT_ROOT:-Outputs/WAV_files/merged_midi_test}"
POOLS="${POOLS:-tools/pools_merged}"
LISTS="${LISTS:-evaluation/merged_test_lists}"
SPLITS_MERGED="${SPLITS_MERGED:-evaluation/splits_merged.csv}"
: "${GPU:?set GPU to one free GPU index, e.g. GPU=3}"
export CUDA_VISIBLE_DEVICES="$GPU"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
die() { echo "FAILED: $*" >&2; exit 1; }

[ -s "$OUT/bridge_metrics_all.csv" ] || die "no $OUT/bridge_metrics_all.csv -- run run_merged_test.sh first"

echo "== per-architecture reference lists"
"$PYTHON" evaluation/make_eval_pools.py --splits evaluation/splits.csv "$SPLITS_MERGED" \
  --architectures midi merged --cases poly --out-dir "$POOLS" > "$LOGDIR/pools_clap_$STAMP.log" 2>&1 \
  || { cat "$LOGDIR/pools_clap_$STAMP.log"; die "make_eval_pools"; }
for arch in midi merged; do
  echo "  reference_poly_$arch.txt: $(grep -c . "$POOLS/reference_poly_$arch.txt") candidates"
done

run_clap() {   # name, arch, csv, experiments...
  local name="$1" arch="$2" csv="$3"; shift 3
  echo "== CLAP-FAD: $* against $name"
  "$PYTHON" -u evaluation/compute_clap_fad_200.py --out-root "$OUT" --cases poly \
    --experiments "$@" --reference-name "$name" \
    --reference-from "$POOLS/reference_{case}_$arch.txt" --n-ref 1000 \
    --track-list-dir "$LISTS" --out-csv "$OUT/fad/$csv" --device cuda:0 \
    > "$LOGDIR/clap_${name}_$STAMP.log" 2>&1 || die "CLAP-FAD $name -- see $LOGDIR/clap_${name}_$STAMP.log"
  grep -E "sanity|CLAP-FAD" "$LOGDIR/clap_${name}_$STAMP.log" | sed 's/^/  /'
}
run_clap reference_stacked midi clap_fad_stacked_ref.csv original_midi original_midi_s100
run_clap reference_merged merged clap_fad_merged_ref.csv \
  merged_real_midi merged_real_midi_s100 merged_generated_midi merged_generated_midi_s100

echo "== the floor: one reference against the other"
"$PYTHON" - "$OUT/fad/poly" "$OUT/fad/clap_reference_floor.csv" <<'PY'
import csv, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from evaluation.compute_clap_fad import frechet_distance
d, out = Path(sys.argv[1]), sys.argv[2]
a = np.load(d / "reference_stacked_clap.npz", allow_pickle=False)
b = np.load(d / "reference_merged_clap.npz", allow_pickle=False)
shared = len(set(a["names"]) & set(b["names"]))
floor = frechet_distance(a["emb"], b["emb"])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_stacked_reference", "n_merged_reference", "clips_in_both", "clap_fad"])
    w.writerow([a["emb"].shape[0], b["emb"].shape[0], shared, floor])
print("  stacked reference %d clips, merged reference %d, %d in both: CLAP-FAD %.4f"
      % (a["emb"].shape[0], b["emb"].shape[0], shared, floor))
PY

echo "== report"
"$PYTHON" evaluation/report_merged_midi.py --out-root "$OUT" | tail -n 12
