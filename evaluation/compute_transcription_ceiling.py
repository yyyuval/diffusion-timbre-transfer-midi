"""Transcription ceiling: real input audio vs its own ground-truth MIDI.

This measures basic-pitch's error on this material. Every model score should
be read against it. Run per instrument (or per ensemble mix) as needed.

  python evaluation/compute_transcription_ceiling.py \\
      --track-list evaluation/track_list_20.txt \\
      --ensemble flute_bassoon

  python evaluation/compute_transcription_ceiling.py \\
      --track-list evaluation/track_list_20.txt \\
      --ensemble flute --per-stem
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.content_metrics import (  # noqa: E402
    ENSEMBLE_STEMS,
    midi_paths_from_audio_paths,
    resolve_stem_paths,
    score_audio_against_midi,
    sum_stems,
)

DEFAULT_DATASET_ROOT = Path(
    "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
    "cocochorales_tiny_v1_zipped/main_dataset"
)
DEFAULT_OUT_CSV = Path("Outputs/WAV_files/metrics/transcription_ceiling.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--track-list", type=Path, required=True)
    p.add_argument(
        "--ensemble",
        required=True,
        help="Ensemble or single instrument key from ENSEMBLE_STEMS",
    )
    p.add_argument(
        "--per-stem",
        action="store_true",
        help="Score each stem alone instead of the summed mixture",
    )
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--note-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.ensemble not in ENSEMBLE_STEMS:
        raise SystemExit(
            "Unknown ensemble %r. Known: %s"
            % (args.ensemble, sorted(ENSEMBLE_STEMS))
        )

    tracks = [
        line.strip()
        for line in args.track_list.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if args.limit is not None:
        tracks = tracks[: args.limit]

    from audio_diffusion_pytorch import PitchTracker

    tracker = PitchTracker()
    device = torch.device(args.device)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    fieldnames = [
        "timestamp",
        "sample_id",
        "ensemble",
        "stem",
        "note_threshold",
        "transcription_precision",
        "transcription_recall",
        "transcription_f1",
        "multipitch_precision",
        "multipitch_recall",
        "multipitch_f1",
        "multipitch_accuracy",
        "n_ref_notes",
        "n_est_notes",
    ]

    with args.out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for track in tracks:
            try:
                stem_paths = resolve_stem_paths(args.dataset_root, track, args.ensemble)
            except FileNotFoundError as e:
                print("skip %s: %s" % (track, e))
                continue

            jobs = []
            if args.per_stem:
                for path in stem_paths:
                    jobs.append((path.stem, [path]))
            else:
                jobs.append(("mix:" + args.ensemble, stem_paths))

            for stem_label, paths in jobs:
                wav = sum_stems(paths, target_sr=24000, clip_length=409600)
                midi_paths = midi_paths_from_audio_paths(paths)
                scores = score_audio_against_midi(
                    wav,
                    midi_paths,
                    tracker,
                    note_threshold=args.note_threshold,
                    device=device,
                )
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sample_id": track,
                    "ensemble": args.ensemble,
                    "stem": stem_label,
                    "note_threshold": args.note_threshold,
                    **{k: scores[k] for k in fieldnames if k in scores},
                }
                writer.writerow(row)
                f.flush()
                print(
                    "%s  %s  F1_tr=%.3f  F1_mp=%.3f"
                    % (track, stem_label, scores["transcription_f1"], scores["multipitch_f1"])
                )

    print("Ceiling results -> %s" % args.out_csv)


if __name__ == "__main__":
    main()
