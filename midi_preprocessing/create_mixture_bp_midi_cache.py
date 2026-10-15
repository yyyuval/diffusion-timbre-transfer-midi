"""Generated merged MIDI: basic-pitch transcribing the two-instrument mixture.

The polyphonic counterpart of create_bassoon_midi_cache.py. That one transcribes
one stem at a time; this one sums the stems first and transcribes the mixture,
which is what you would have with a real recording -- one audio file, one MIDI,
no record of which instrument played which note. The result is a merged roll
by construction, read by training with midi_merged=true.

For every track that has all the named stems:

  1. sum the stems the way training does (load_mixture): mono, summed, and
     rescaled to 0.95 only if the sum would clip
  2. transcribe the mixture with basic-pitch
  3. KEEP the .mid at <midi_out_root>/<track>/merged_<instruments>.mid -- it is
     the generated MIDI itself, and it makes re-rendering at another fps free
  4. render <cache_root>/<track>/merged_<instruments>_pianoroll.npy

basic-pitch here is the PyTorch port already in the repo
(audio_diffusion_pytorch/basic-pitch-torch): the official ICASSP 2022 weights,
and its note_creation.py is the official note-building code. The official tool
is not installed on the server, and the port needs nothing new in venv_yuval.
The audio goes in by file path, so the port's own loader resamples it to
22.05 kHz -- the sample-rate trap in pitch_tracking_utils comes from handing
the model 24 kHz tensors directly, and does not apply here.

The port's predict() reloads the model from a relative path on every call, so
it is not used: the model is loaded once and predict()'s own body is run with
its own defaults. Resumable -- a track with a kept .mid is never transcribed
again -- and --shard splits the list between processes:

    for i in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=4 python midi_preprocessing/create_mixture_bp_midi_cache.py \\
        --dataset_root $D --instrument flute bassoon --shard $i/4 \\
        --cache_root midi_cache_wood_fps75_bp_merged --midi_out_root midi_bp_merged/wood &
    done

It uses whichever GPU CUDA_VISIBLE_DEVICES shows it, or the CPU if none.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PORT_DIR = ROOT / "audio_diffusion_pytorch" / "basic-pitch-torch"
WEIGHTS = PORT_DIR / "assets" / "basic_pitch_pytorch_icassp_2022.pth"
for p in (str(ROOT), str(PORT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# predict()'s defaults, which are the official tool's defaults.
ONSET_THRESHOLD = 0.5
FRAME_THRESHOLD = 0.3
MINIMUM_NOTE_LENGTH_MS = 127.70


class Transcriber:
    """The port's predict(), with the model loaded once instead of per call."""

    def __init__(self):
        import scipy.signal
        import torch
        from basic_pitch_torch.model import BasicPitchTorch

        # The port's note_creation.py comes from an older basic-pitch and calls
        # scipy.signal.gaussian, which SciPy 1.13 removed. The same function
        # lives on as scipy.signal.windows.gaussian. Restored here rather than
        # by editing the port, which is not under version control.
        if not hasattr(scipy.signal, "gaussian"):
            scipy.signal.gaussian = scipy.signal.windows.gaussian

        if not WEIGHTS.exists():
            sys.exit("basic-pitch weights not found at %s" % WEIGHTS)
        self.torch = torch
        self.model = BasicPitchTorch()
        self.model.load_state_dict(
            torch.load(str(WEIGHTS), map_location="cpu", weights_only=True))
        self.model.eval()
        # run_inference moves the audio to cuda whenever cuda is available, so
        # the model has to follow the same rule or the two end up apart.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            self.model.cuda()

    def __call__(self, wav_path: Path):
        from basic_pitch_torch import note_creation
        from basic_pitch_torch.constants import AUDIO_SAMPLE_RATE, FFT_HOP
        from basic_pitch_torch.inference import run_inference

        # run_inference does not turn autograd off; without this every call
        # keeps the whole forward graph alive for nothing.
        with self.torch.inference_mode():
            model_output = run_inference(str(wav_path), self.model)
        min_note_len = int(np.round(MINIMUM_NOTE_LENGTH_MS / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
        midi_data, _ = note_creation.model_output_to_notes(
            model_output,
            onset_thresh=ONSET_THRESHOLD,
            frame_thresh=FRAME_THRESHOLD,
            min_note_len=min_note_len,
            min_freq=None,
            max_freq=None,
            multiple_pitch_bends=False,
            melodia_trick=True,
            midi_tempo=120,
        )
        return midi_data


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


def render_roll(pm, fps: int, velocity_threshold: float) -> np.ndarray:
    """The same two lines every cache builder here uses."""
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
    p.add_argument("--fps", type=int, default=75,
                   help="must match midi_fps in exp/datamodule/base.yaml")
    p.add_argument("--velocity_threshold", type=float, default=40.0,
                   help="same as the existing generated caches; basic-pitch writes real velocities")
    p.add_argument("--shard", default="0/1", help="i/n: process every n-th track starting at i")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="re-render rolls that exist; the .mid is reused unless --retranscribe")
    p.add_argument("--retranscribe", action="store_true",
                   help="run basic-pitch again even where a .mid is kept")
    args = p.parse_args()

    if len(args.instrument) < 2:
        p.error("--instrument needs two or more names; for one stem use the single-stem scripts")
    shard_index, shard_count = parse_shard(args.shard)

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

    print("%d tracks (shard %d/%d) with all of %s" % (len(groups), shard_index, shard_count, args.instrument))
    print("-> %s/<track>/%s.mid  and  %s/<track>/%s_pianoroll.npy  (fps %d, threshold %g)"
          % (midi_out_root, name, cache_root, name, args.fps, args.velocity_threshold))
    print("already cached %d, to render from a kept .mid %d, to transcribe %d"
          % (skipped, len(to_render), len(to_transcribe)), flush=True)

    import pretty_midi
    import torchaudio

    rendered = failed = 0

    def save_roll(pm, track, roll_path):
        nonlocal rendered
        roll = render_roll(pm, args.fps, args.velocity_threshold)
        roll_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(roll_path, roll)
        rendered += 1
        return roll

    for track, mid_path, roll_path in to_render:
        try:
            save_roll(pretty_midi.PrettyMIDI(str(mid_path)), track, roll_path)
        except Exception as e:
            failed += 1
            print("  FAILED render %s: %s" % (track, e))

    if to_transcribe:
        transcribe = Transcriber()
        print("basic-pitch (PyTorch port) on %s" % transcribe.device, flush=True)

    start = time.monotonic()
    transcription_failures = 0
    with tempfile.TemporaryDirectory(prefix="bp_mixture_") as tmp:
        wav_path = Path(tmp) / "mixture.wav"
        for n, (track, stems, mid_path, roll_path) in enumerate(to_transcribe, start=1):
            try:
                mixture, rate = build_mixture(stems)
                torchaudio.save(str(wav_path), mixture, rate)
                midi = transcribe(wav_path)
                mid_path.parent.mkdir(parents=True, exist_ok=True)
                midi.write(str(mid_path))
                roll = save_roll(midi, track, roll_path)
                if n <= 3:
                    notes = sum(len(i.notes) for i in midi.instruments)
                    active = int((roll.sum(axis=0) > 0).sum())
                    print("  %s: %d notes, %d/%d frames sounding"
                          % (track, notes, active, roll.shape[1]), flush=True)
            except Exception as e:
                failed += 1
                transcription_failures += 1
                print("  FAILED %s: %s: %s" % (track, type(e).__name__, e), flush=True)
                # Every track failing from the start is a bug, not bad audio.
                # Each attempt still pays for inference, so without this a
                # systematic error grinds through thousands of tracks for hours.
                if transcription_failures == n and n >= 5:
                    sys.exit("The first %d tracks all failed -- stopping. Fix the error "
                             "above and run the same command again." % n)

            if n % 100 == 0 or n == len(to_transcribe):
                minutes = (time.monotonic() - start) / 60
                per_minute = n / minutes if minutes > 0 else 0.0
                left = (len(to_transcribe) - n) / per_minute if per_minute > 0 else float("nan")
                print("[%d/%d transcribed] %.1f min, %.1f tracks/min, ~%.0f min left"
                      % (n, len(to_transcribe), minutes, per_minute, left), flush=True)

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
