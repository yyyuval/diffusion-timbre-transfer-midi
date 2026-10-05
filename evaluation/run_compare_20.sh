#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
TRACK_LIST="${TRACK_LIST:-evaluation/track_list_20.txt}"

while read -r TRACK_NAME; do
    [ -z "$TRACK_NAME" ] && continue

    echo "============================================================"
    echo "TRACK: $TRACK_NAME | EXPERIMENT: no_midi"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="$GPU" TRACK_NAME="$TRACK_NAME" EXPERIMENT_NAME=no_midi \
        python Run_timbre_transfer.py

    echo "============================================================"
    echo "TRACK: $TRACK_NAME | EXPERIMENT: original_midi"
    echo "============================================================"
    CUDA_VISIBLE_DEVICES="$GPU" TRACK_NAME="$TRACK_NAME" EXPERIMENT_NAME=original_midi \
        python Run_timbre_transfer.py

done < "$TRACK_LIST"
