"""Evaluate onset-tolerance sensitivity without regenerating bridge outputs."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("Outputs/WAV_files/pitch_validation_200"))
    p.add_argument("--tolerances", nargs="+", type=float, default=[0.05, 0.1, 0.15])
    p.add_argument("--note-threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, help="Limit generated clips for a timing pilot")
    args = p.parse_args()
    if any(not math.isfinite(t) or t <= 0 for t in args.tolerances):
        p.error("Tolerances must be finite and positive")
    if not 0 < args.note_threshold <= 1:
        p.error("Note threshold must be in (0, 1]")
    import numpy as np
    import torchaudio
    from evaluation.content_metrics import transcribe_wav, frames_to_note_events, notes_from_midi_files, score_transcription, midi_paths_from_audio_paths
    from audio_diffusion_pytorch import PitchTracker
    with (args.root / "bridge_metrics_all.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    cache = args.root / "onset_sensitivity" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache.parent / ("pilot.csv" if args.limit else "scores.csv")
    fields = ["kind", "experiment", "sample_id", "case", "onset_tolerance", "pitch_tolerance_cents", "note_threshold", "precision", "recall", "f1", "n_ref_notes", "n_est_notes"]
    tracker = None
    def events(path):
        nonlocal tracker
        path = Path(path)
        stat = path.stat()
        identity = [str(path.resolve()), stat.st_size, stat.st_mtime_ns, args.note_threshold, "events-v1-22050"]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        dest = cache / (key + ".npz")
        if dest.exists():
            with np.load(dest, allow_pickle=False) as a:
                return a["intervals"], a["pitches"], float(a["duration"])
        if tracker is None:
            tracker = PitchTracker()
        wav, sr = torchaudio.load(str(path))
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        duration = wav.shape[-1] / 24000
        intervals, pitches = frames_to_note_events(transcribe_wav(wav, tracker), threshold=args.note_threshold)
        np.savez_compressed(dest, intervals=intervals, pitches=pitches, duration=duration)
        return intervals, pitches, duration
    seen_inputs = set()
    start = time.monotonic()
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            stems = row["input_stems"].split(";")
            midi = midi_paths_from_audio_paths(stems)
            jobs = [("generated", row["experiment"], row["output_audio"])]
            input_key = (row["case"], row["sample_id"])
            if input_key not in seen_inputs:
                jobs.append(("real_input", "input_" + row["case"], row["input_audio"]))
                seen_inputs.add(input_key)
            for kind, exp, path in jobs:
                intervals, pitches, duration = events(path)
                ref_intervals, ref_pitches = notes_from_midi_files(midi, max_time=duration)
                for tol in args.tolerances:
                    scores = score_transcription(ref_intervals, ref_pitches, intervals, pitches, onset_tolerance=tol, pitch_tolerance=50)
                    writer.writerow(dict(kind=kind, experiment=exp, sample_id=row["sample_id"], case=row["case"], onset_tolerance=tol,
                                         pitch_tolerance_cents=50, note_threshold=args.note_threshold,
                                         precision=scores["precision"], recall=scores["recall"], f1=scores["f1"],
                                         n_ref_notes=len(ref_intervals), n_est_notes=len(intervals)))
            f.flush()
            print("[%d/%d] %s / %s; elapsed %.1f min" % (i, len(rows), row["experiment"], row["sample_id"], (time.monotonic()-start)/60), flush=True)
    print("Completed ->", out, flush=True)

if __name__ == "__main__":
    main()
