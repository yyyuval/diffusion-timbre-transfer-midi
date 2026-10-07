"""Score generated outputs with mir_eval transcription + multipitch.

Reads the existing inference_metrics.csv for (experiment, sample_id, output_audio)
pairs, re-derives the input track's ground-truth MIDI from the source ensemble,
and writes a SEPARATE CSV (never extends inference_metrics.csv).

  python evaluation/compute_content_metrics.py
  python evaluation/compute_content_metrics.py --experiments no_midi original_midi
  python evaluation/compute_content_metrics.py --note-threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.content_metrics import (  # noqa: E402
    midi_paths_from_audio_paths,
    resolve_stem_paths,
    score_audio_against_midi,
)

DEFAULT_DATASET_ROOT = Path(
    "/dsi/gannot-lab/gannot-lab1/datasets/Yuval_Shlomi_2026_Music_Proj/"
    "cocochorales_tiny_v1_zipped/main_dataset"
)
DEFAULT_METRICS_CSV = Path("Outputs/WAV_files/metrics/inference_metrics.csv")
DEFAULT_OUT_CSV = Path("Outputs/WAV_files/metrics/content_metrics.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    p.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="If set, only these experiment names (exact match on CSV column)",
    )
    p.add_argument("--source", default="flute_bassoon", help="Source ensemble name")
    p.add_argument("--note-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=None, help="Max rows to score")
    p.add_argument("--track-list", type=Path, default=None,
                   help="score only these sample_ids, one per line. The bridge CSV "
                        "holds ~1000 clips per experiment because FAD needs them, "
                        "but only the clips EVERY architecture held out can be "
                        "compared clip by clip. Point this at the paired prefix "
                        "(the first --n-content lines of any track_list_*.txt, or "
                        "tools/pools/shared_<case>.txt) so the paired test in "
                        "paired_stats.py is valid. --limit truncates the table; "
                        "this selects from it, which is not the same thing.")
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps — model runs on CPU by default")
    return p.parse_args()


def already_scored(out_csv: Path) -> set:
    """(experiment, sample_id) pairs already present — allows resume."""
    done = set()
    if not out_csv.is_file():
        return done
    with out_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["experiment"], row["sample_id"]))
    return done


def main():
    args = parse_args()
    if not args.metrics_csv.is_file():
        raise SystemExit("Missing metrics CSV: %s" % args.metrics_csv)

    import pandas as pd
    from audio_diffusion_pytorch import PitchTracker

    df = pd.read_csv(args.metrics_csv)
    if args.experiments:
        df = df[df["experiment"].isin(args.experiments)].copy()
    if args.track_list is not None:
        wanted = [l.strip() for l in args.track_list.read_text().splitlines()
                  if l.strip() and not l.strip().startswith("#")]
        before = len(df)
        df = df[df["sample_id"].astype(str).isin(set(wanted))].copy()
        print("track list %s: %d ids -> %d of %d rows kept"
              % (args.track_list, len(wanted), len(df), before))
        per_exp = df.groupby("experiment")["sample_id"].nunique()
        if per_exp.nunique() > 1:
            print("WARNING: the experiments do not cover the same clips -- %s.\n"
                  "         A paired test needs identical inputs; score only the "
                  "clips common to all of them." % per_exp.to_dict())
    if df.empty:
        raise SystemExit("No rows to score after filtering.")

    if args.limit is not None:
        df = df.head(args.limit)

    tracker = PitchTracker()
    device = torch.device(args.device)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = already_scored(args.out_csv)
    write_header = not args.out_csv.exists()

    fieldnames = [
        "timestamp",
        "experiment",
        "sample_id",
        "source",
        "target",
        "output_audio",
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

    n_ok = 0
    n_skip = 0
    with args.out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, row in df.iterrows():
            experiment = str(row["experiment"])
            sample_id = str(row["sample_id"])
            key = (experiment, sample_id)
            if key in done:
                n_skip += 1
                continue

            output_audio = Path(str(row["output_audio"]))
            if not output_audio.is_file():
                # relative paths in CSV assume repo root cwd
                alt = ROOT / output_audio
                if alt.is_file():
                    output_audio = alt
                else:
                    print("MISSING output: %s — skip" % row["output_audio"])
                    continue

            # Prefer the per-row source from inference_metrics.csv (poly vs mono
            # bridges differ). --source is only the fallback when the column is absent.
            source = str(row["source"]) if "source" in row and str(row["source"]) not in ("", "nan") else args.source
            target = str(row["target"]) if "target" in row and str(row["target"]) not in ("", "nan") else ""

            try:
                stem_paths = resolve_stem_paths(args.dataset_root, sample_id, source)
                midi_paths = midi_paths_from_audio_paths(stem_paths)
            except (FileNotFoundError, KeyError) as e:
                print("REF ERROR %s: %s — skip" % (sample_id, e))
                continue

            wav, sr = torchaudio.load(str(output_audio))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)

            scores = score_audio_against_midi(
                wav,
                midi_paths,
                tracker,
                note_threshold=args.note_threshold,
                device=device,
            )

            out_row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": experiment,
                "sample_id": sample_id,
                "source": source,
                "target": target,
                "output_audio": str(output_audio),
                "note_threshold": args.note_threshold,
                **{k: scores[k] for k in fieldnames if k in scores},
            }
            writer.writerow(out_row)
            f.flush()
            n_ok += 1
            print(
                "[%d] %s / %s  F1_tr=%.3f  F1_mp=%.3f  (P=%.3f R=%.3f)"
                % (
                    n_ok,
                    experiment,
                    sample_id,
                    scores["transcription_f1"],
                    scores["multipitch_f1"],
                    scores["transcription_precision"],
                    scores["transcription_recall"],
                )
            )

    print("Wrote %d new rows (%d skipped already present) -> %s" % (n_ok, n_skip, args.out_csv))


if __name__ == "__main__":
    main()
