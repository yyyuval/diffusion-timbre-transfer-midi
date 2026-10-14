"""Generated merged MIDI: basic-pitch transcribing the two-instrument mixture.

The polyphonic counterpart of create_bassoon_midi_cache.py. That one transcribes
one stem at a time; this one sums the stems first and transcribes the mixture,
which is what you would have with a real recording -- one audio file, one MIDI,
no record of which instrument played which note. The result is a merged roll
by construction, read by training with midi_merged=true.

For every track that has all the named stems:

  1. sum the stems the way training does (load_mixture): mono, summed, and
     rescaled to 0.95 only if the sum would clip
  2. transcribe the mixture with the official basic-pitch tool
  3. KEEP the .mid at <midi_out_root>/<track>/merged_<instruments>.mid -- it is
     the generated MIDI itself, and it makes re-rendering at another fps free
  4. render <cache_root>/<track>/merged_<instruments>_pianoroll.npy

Transcription is by far the slow step, so many files go to one call of the tool
(the model loads once), tracks whose .mid already exists are not transcribed
again, and --shard splits the list between processes:

    for i in 0 1 2 3 4 5 6 7; do
      python midi_preprocessing/create_mixture_bp_midi_cache.py --dataset_root $D \\
        --instrument flute bassoon --shard $i/8 \\
        --cache_root midi_cache_wood_fps75_bp_merged --midi_out_root midi_bp_merged/wood &
    done

The tool runs on CPU unless --use_gpu: TensorFlow otherwise takes the whole
memory of every visible GPU, which is not something to do beside training runs.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from basic_pitch_cli import expected_output, find_basic_pitch, output_tail, run_basic_pitch  # noqa: E402


def build_mixture(stem_paths: List[Path]):
    """Whole track, summed exactly as load_mixture sums a training crop."""
    import torch
    import torchaudio

    parts, rate = [], None
    for path in stem_paths:
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if rate is None:
            rate = sr
        elif sr != rate:
            wav = torchaudio.functional.resample(wav, sr, rate)
        parts.append(wav)
    length = max(w.shape[-1] for w in parts)
    mixture = torch.zeros(1, length)
    for w in parts:
        mixture[:, : w.shape[-1]] += w
    peak = float(mixture.abs().max())
    if peak > 0.95:
        mixture = 0.95 * mixture / peak
    return mixture, rate


def render_roll(midi_path: Path, fps: int, velocity_threshold: float) -> np.ndarray:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return (pm.get_piano_roll(fs=fps) > velocity_threshold).astype(np.uint8)


def parse_shard(text: str):
    try:
        index, count = (int(x) for x in text.split("/"))
    except ValueError:
        raise SystemExit("--shard must look like 3/8")
    if not 0 <= index < count:
        raise SystemExit("--shard %s: the index must be in 0..%d" % (text, count - 1))
    return index, count


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--instrument", nargs="+", required=True, help="e.g. flute bassoon")
    p.add_argument("--cache_root", required=True, help="where the merged rolls go")
    p.add_argument("--midi_out_root", required=True,
                   help="where the generated .mid files are kept, one folder per track")
    p.add_argument("--work_dir", default=None,
                   help="temporary mixtures and raw tool output; default <midi_out_root>/_work")
    p.add_argument("--fps", type=int, default=75,
                   help="must match midi_fps in exp/datamodule/base.yaml")
    p.add_argument("--velocity_threshold", type=float, default=40.0,
                   help="same as the existing generated caches; basic-pitch writes real velocities")
    p.add_argument("--batch_size", type=int, default=32, help="files per call of the tool")
    p.add_argument("--shard", default="0/1", help="i/n: process every n-th track starting at i")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="re-render rolls that exist; the .mid is reused unless --retranscribe")
    p.add_argument("--retranscribe", action="store_true",
                   help="run basic-pitch again even where a .mid is kept")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--basic_pitch_bin", default=None)
    args = p.parse_args()

    if len(args.instrument) < 2:
        p.error("--instrument needs two or more names; for one stem use the single-stem scripts")
    shard_index, shard_count = parse_shard(args.shard)
    binary = find_basic_pitch(args.basic_pitch_bin)

    from audio_data_pytorch.datasets.wav_dataset import (
        get_paired_stem_filenames,
        merged_midi_name,
    )

    groups = get_paired_stem_filenames([args.dataset_root], args.instrument)
    groups = groups[shard_index::shard_count]
    if args.limit is not None:
        groups = groups[: args.limit]
    if not groups:
        sys.exit("No track has all of %s under %s." % (args.instrument, args.dataset_root))

    name = merged_midi_name(args.instrument)
    cache_root = Path(args.cache_root)
    midi_out_root = Path(args.midi_out_root)
    work_dir = Path(args.work_dir) if args.work_dir else midi_out_root / "_work"
    print("%d tracks (shard %d/%d) with all of %s" % (len(groups), shard_index, shard_count, args.instrument))
    print("basic-pitch: %s  (%s)" % (binary, "GPU allowed" if args.use_gpu else "CPU only"))
    print("-> %s/<track>/%s.mid  and  %s/<track>/%s_pianoroll.npy  (fps %d, threshold %g)"
          % (midi_out_root, name, cache_root, name, args.fps, args.velocity_threshold))

    to_render, to_transcribe = [], []
    skipped = 0
    for group in groups:
        stems = [Path(s) for s in group]
        track = stems[0].parent.parent.name
        roll_path = cache_root / track / (name + "_pianoroll.npy")
        mid_path = midi_out_root / track / (name + ".mid")
        if roll_path.exists() and not args.overwrite:
            skipped += 1
        elif mid_path.exists() and not args.retranscribe:
            to_render.append((track, mid_path, roll_path))
        else:
            to_transcribe.append((track, stems, mid_path, roll_path))
    print("already cached %d, to render from a kept .mid %d, to transcribe %d"
          % (skipped, len(to_render), len(to_transcribe)))

    rendered = failed = 0

    def render(track, mid_path, roll_path):
        nonlocal rendered, failed
        try:
            roll = render_roll(mid_path, args.fps, args.velocity_threshold)
            roll_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(roll_path, roll)
            rendered += 1
        except Exception as e:
            failed += 1
            print("  FAILED render %s: %s" % (track, e))

    for track, mid_path, roll_path in to_render:
        render(track, mid_path, roll_path)

    start = time.monotonic()
    done = 0
    batch_tag = "shard%d" % shard_index
    for b in range(0, len(to_transcribe), args.batch_size):
        batch = to_transcribe[b : b + args.batch_size]
        batch_dir = work_dir / ("%s_batch%05d" % (batch_tag, b // args.batch_size))
        audio_dir, out_dir = batch_dir / "audio", batch_dir / "midi"
        if batch_dir.exists():
            shutil.rmtree(batch_dir)
        audio_dir.mkdir(parents=True)

        import torchaudio

        inputs = []
        for track, stems, mid_path, roll_path in batch:
            try:
                mixture, rate = build_mixture(stems)
                wav_path = audio_dir / ("%s__%s.wav" % (track, name))
                torchaudio.save(str(wav_path), mixture, rate)
                inputs.append((track, wav_path, mid_path, roll_path))
            except Exception as e:
                failed += 1
                print("  FAILED mixing %s: %s" % (track, e))

        if not inputs:
            shutil.rmtree(batch_dir, ignore_errors=True)
            done += len(batch)
            continue

        if args.retranscribe:
            for _, _, mid_path, _ in inputs:
                if mid_path.exists():
                    mid_path.unlink()

        proc = run_basic_pitch(out_dir, [w for _, w, _, _ in inputs],
                               binary=binary, cpu_only=not args.use_gpu)

        lost = []
        for track, wav_path, mid_path, roll_path in inputs:
            produced = expected_output(out_dir, wav_path)
            if not produced.exists():
                lost.append(track)
                continue
            mid_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), str(mid_path))
            render(track, mid_path, roll_path)
        if lost:
            failed += len(lost)
            print("  basic-pitch produced no MIDI for %d of %d (exit %d), e.g. %s. Its output:"
                  % (len(lost), len(inputs), proc.returncode, lost[0]))
            print(output_tail(proc))
        else:
            # Kept on failure, so the mixtures can be looked at.
            shutil.rmtree(batch_dir, ignore_errors=True)

        done += len(batch)
        minutes = (time.monotonic() - start) / 60
        per_minute = done / minutes if minutes > 0 else 0.0
        left = (len(to_transcribe) - done) / per_minute if per_minute > 0 else float("nan")
        print("[%d/%d transcribed] %.1f min, %.1f tracks/min, ~%.0f min left"
              % (done, len(to_transcribe), minutes, per_minute, left), flush=True)

    total = len(groups)
    covered = skipped + rendered
    print("\nDone")
    print("  rendered : %d" % rendered)
    print("  skipped  : %d  (already cached)" % skipped)
    print("  failed   : %d" % failed)
    print("\ncoverage: %d/%d tracks in this shard (%.1f%%)" % (covered, total, 100.0 * covered / total))
    if covered < total:
        print("Every track without a roll trains on an all-zero roll. Re-run the same "
              "command -- finished tracks are skipped -- before launching training.")


if __name__ == "__main__":
    main()
