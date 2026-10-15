#!/usr/bin/env bash
# The eight merged-MIDI runs, on a fixed allocation of GPUs, one run per GPU.
#
#   mkdir -p train_logs && setsid nohup bash train_merged_midi.sh > train_logs/merged_queue.log 2>&1 < /dev/null &
#
# GPUS              GPU indices to use. Default: the N_GPUS with the least
#                   memory in use at launch.
# N_GPUS            how many to pick when GPUS is unset (default 3 -- our share).
# SKIP_TRANSCRIBE=1 do not start basic-pitch; use whatever is in the caches.
#
# What it does:
#   1. Starts basic-pitch on the CPU for both ensembles, 4 processes each, in
#      the background. That fills the generated-MIDI caches runs C and D read.
#      CPU on purpose: it keeps the GPU allocation for training.
#   2. Runs the eight trainings as a queue, at most one per GPU: A and B first
#      (real merged MIDI, caches already built), then C and D. A C/D run waits
#      until its generated cache holds as many rolls as the real one -- that is
#      exactly the set of tracks training will ask for. If transcription has
#      ended short of that, the run is reported as blocked rather than started
#      on a cache that would serve all-zero rolls.
#   3. Before each launch, checks that the GPU still has room. Another lab
#      member taking it shows up as a wait in this log, not as an OOM at
#      epoch 40.
#
# A run that exits 0 gets train_logs/<tag>.done and is skipped on re-run, so if
# anything fails, fix it and launch the same command again.
set -uo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
PY="${PY:-venv_yuval/bin/python}"
N_GPUS="${N_GPUS:-3}"
POLL="${POLL:-60}"
MAX_USED_MIB="${MAX_USED_MIB:-16000}"   # a run needs ~12 GB of the 32
DATASET="${DATASET:-/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/cocochorales_tiny_v1_zipped/main_dataset}"
LOGS="train_logs"
mkdir -p "$LOGS"

#        exp_tag                                  | exp                  | cache                                | sigma_max
JOBS=(
  "flute_bassoon_gtmerged_ms_85ep|flute_bassoon_latent|midi_cache_wood_fps75_gt_merged|5"
  "violin_cello_gtmerged_ms_85ep|violin_cello_latent|midi_cache_string_fps75_gt_merged|5"
  "flute_bassoon_gtmerged_ms_85ep_sigma100|flute_bassoon_latent|midi_cache_wood_fps75_gt_merged|100"
  "violin_cello_gtmerged_ms_85ep_sigma100|violin_cello_latent|midi_cache_string_fps75_gt_merged|100"
  "flute_bassoon_bpmerged_ms_85ep|flute_bassoon_latent|midi_cache_wood_fps75_bp_merged|5"
  "violin_cello_bpmerged_ms_85ep|violin_cello_latent|midi_cache_string_fps75_bp_merged|5"
  "flute_bassoon_bpmerged_ms_85ep_sigma100|flute_bassoon_latent|midi_cache_wood_fps75_bp_merged|100"
  "violin_cello_bpmerged_ms_85ep_sigma100|violin_cello_latent|midi_cache_string_fps75_bp_merged|100"
)

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }

gpu_used_mib() {   # memory in use on one GPU, or 0 when nvidia-smi is absent
  command -v nvidia-smi >/dev/null || { echo 0; return; }
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

rolls_in() {
  if [ -d "$1" ]; then find "$1" -name '*_pianoroll.npy' | wc -l | tr -d ' '; else echo 0; fi
}

# ---------------------------------------------------------------- the GPUs
if [ -z "${GPUS:-}" ]; then
  command -v nvidia-smi >/dev/null || { echo "no nvidia-smi: set GPUS"; exit 1; }
  GPUS="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
          | sort -t, -k2 -n | head -n "$N_GPUS" | cut -d, -f1 | tr -d ' ' | tr '\n' ' ')"
fi
read -ra GPU_LIST <<< "$GPUS"
say "GPUs: ${GPU_LIST[*]}"
for g in "${GPU_LIST[@]}"; do say "  GPU $g: $(gpu_used_mib "$g") MiB in use"; done

# ---------------------------------------------------------- transcription
BP_PIDS=()
if [ "${SKIP_TRANSCRIBE:-0}" != "1" ]; then
  for spec in "wood:flute bassoon" "string:violin cello"; do
    ens="${spec%%:*}"; inst="${spec#*:}"
    real="midi_cache_${ens}_fps75_gt_merged"; gen="midi_cache_${ens}_fps75_bp_merged"
    if [ "$(rolls_in "$gen")" -ge "$(rolls_in "$real")" ] && [ "$(rolls_in "$real")" -gt 0 ]; then
      say "basic-pitch: $gen already complete"
      continue
    fi
    for i in 0 1 2 3; do
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES="" "$PY" midi_preprocessing/create_mixture_bp_midi_cache.py \
        --dataset_root "$DATASET" --instrument $inst --shard "$i/4" \
        --cache_root "$gen" --midi_out_root "midi_bp_merged/$ens" \
        > "$LOGS/bp_${ens}_$i.log" 2>&1 &
      BP_PIDS+=($!)
    done
    say "basic-pitch: started 4 CPU processes for $inst -> $gen (logs $LOGS/bp_${ens}_*.log)"
  done
fi

transcribing() {
  local p
  for p in "${BP_PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && return 0; done
  return 1
}

# A cache is ready when it holds a roll for every track the real merged cache
# does -- both are built over get_paired_stem_filenames, so the sets match.
cache_state() {   # prints ready | wait | blocked
  local cache="$1" real="${1/_bp_merged/_gt_merged}" have want
  have="$(rolls_in "$cache")"; want="$(rolls_in "$real")"
  if [ "$want" -gt 0 ] && [ "$have" -ge "$want" ]; then echo ready
  elif [ "$cache" != "$real" ] && transcribing; then echo wait
  else echo blocked
  fi
}

# --------------------------------------------------------------- the queue
declare -A SLOT_PID=() SLOT_TAG=()
next=0
failed=()
blocked=()

launch() {
  local g="$1" tag="$2" exp="$3" cache="$4" sigma="$5"
  local extra=()
  [ "$sigma" != "5" ] && extra+=("sigma_max=$sigma")
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$g" "$PY" train.py \
    exp="$exp" use_midi=true midi_merged=true midi_bins=128 \
    midi_cache_root="$ROOT/$cache" ${extra[@]+"${extra[@]}"} \
    trainer.max_epochs=85 datamodule.batch_size=16 datamodule.num_workers=8 \
    exp_tag="$tag" > "$LOGS/$tag.log" 2>&1 &
  SLOT_PID[$g]=$!
  SLOT_TAG[$g]="$tag"
  say "START  $tag on GPU $g (pid ${SLOT_PID[$g]}, log $LOGS/$tag.log)"
}

while :; do
  # reap the runs that ended
  for g in "${GPU_LIST[@]}"; do
    pid="${SLOT_PID[$g]:-}"
    [ -z "$pid" ] && continue
    kill -0 "$pid" 2>/dev/null && continue
    wait "$pid"; rc=$?
    tag="${SLOT_TAG[$g]}"
    if [ "$rc" -eq 0 ]; then
      touch "$LOGS/$tag.done"; say "DONE   $tag (GPU $g free)"
    else
      failed+=("$tag"); say "FAILED $tag, exit $rc -- see $LOGS/$tag.log"
    fi
    SLOT_PID[$g]=""; SLOT_TAG[$g]=""
  done

  # fill the free GPUs, in queue order
  for g in "${GPU_LIST[@]}"; do
    [ -n "${SLOT_PID[$g]:-}" ] && continue
    while [ "$next" -lt "${#JOBS[@]}" ]; do
      IFS='|' read -r tag exp cache sigma <<< "${JOBS[$next]}"
      if [ -f "$LOGS/$tag.done" ]; then
        say "skip   $tag (finished earlier)"; next=$((next + 1)); continue
      fi
      state="$(cache_state "$cache")"
      if [ "$state" = blocked ]; then
        blocked+=("$tag")
        say "BLOCKED $tag: $cache has $(rolls_in "$cache") rolls, the real cache $(rolls_in "${cache/_bp_merged/_gt_merged}") -- check $LOGS/bp_*.log, then re-run this script"
        next=$((next + 1)); continue
      fi
      break
    done
    [ "$next" -ge "${#JOBS[@]}" ] && break
    [ "$state" = wait ] && break        # queue order: nothing behind it jumps ahead
    used="$(gpu_used_mib "$g")"
    if [ "$used" -gt "$MAX_USED_MIB" ]; then
      say "GPU $g has $used MiB in use by others -- waiting for it"
      continue
    fi
    IFS='|' read -r tag exp cache sigma <<< "${JOBS[$next]}"
    launch "$g" "$tag" "$exp" "$cache" "$sigma"
    next=$((next + 1))
  done

  running=0
  for g in "${GPU_LIST[@]}"; do [ -n "${SLOT_PID[$g]:-}" ] && running=$((running + 1)); done
  if [ "$next" -ge "${#JOBS[@]}" ] && [ "$running" -eq 0 ]; then break; fi
  sleep "$POLL"
done

say "queue finished"
[ "${#failed[@]}" -gt 0 ] && say "failed : ${failed[*]}"
[ "${#blocked[@]}" -gt 0 ] && say "blocked: ${blocked[*]}"
transcribing && say "basic-pitch is still running -- see $LOGS/bp_*.log"
exit 0
