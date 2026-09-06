"""
Build a piano-roll cache from the dataset's OWN MIDI (stems_midi/), not from
basic-pitch.

CocoChorales is synthesised -- the audio was rendered from this MIDI -- so these
notes are exact rather than transcribed. There is no basic-pitch step here, which
makes this minutes rather than hours, and nothing intermediate is written.

Measured on string_track000534, transcription differs from ground truth like this:
    ground truth   30 notes, D3-E4,  strictly monophonic (max 1 note/frame)
    basic-pitch    62 notes, E2-B4,  up to 3 notes/frame
    overlap        IoU 0.830  (69 cells missed, 281 invented)

Velocity note: CocoChorales writes a flat velocity of 64 for every note -- the
field carries no dynamics. So the threshold is a no-op as long as it is under 64;
it defaults to 0 here. Do NOT reuse 40 out of habit thinking it filters anything.

    python midi_preprocessing/create_gt_midi_cache.py \
      --dataset_root /dsi/.../main_dataset \
      --instrument cello \
      --cache_root /home/shared_workspace/diffusion-timbre-transfer/midi_cache_cello_fps75_gt
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pretty_midi


def render(midi_path: Path, out_path: Path, fps: int, velocity_threshold: float):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    velocity_roll = pm.get_piano_roll(fs=fps)
    piano_roll = (velocity_roll > velocity_threshold).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, piano_roll)
    return piano_roll.shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--instrument", required=True)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--fps", type=int, default=75,
                   help="must match midi_fps in exp/datamodule/base.yaml")
    p.add_argument("--velocity_threshold", type=float, default=0.0,
                   help="0 keeps every sounding note; CocoChorales velocities "
                        "are a flat 64, so anything under 64 is equivalent")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    inst = args.instrument

    # Drive off the AUDIO stems, not the MIDI ones. Training asks for a roll per
    # wav, so enumerating wavs is the only way to guarantee the cache covers
    # exactly what will be requested -- and to notice when one is missing.
    wav_files = sorted(dataset_root.glob(f"*/stems_audio/*{inst}*.wav"))

    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    print(f"Found {len(wav_files)} {inst} wav files")
    print(f"fps={args.fps}  velocity_threshold={args.velocity_threshold}")
    print(f"cache_root={cache_root}")
    print()

    if not wav_files:
        sys.exit("No wav files matched -- check --dataset_root and --instrument.")

    success = skipped = missing = failed = 0
    missing_examples = []

    for i, wav_path in enumerate(wav_files, start=1):
        track_name = wav_path.parent.parent.name
        out_path = cache_root / track_name / f"{wav_path.stem}_pianoroll.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        # stems_midi/<same stem>.mid sits beside stems_audio/<stem>.wav
        midi_path = wav_path.parent.parent / "stems_midi" / f"{wav_path.stem}.mid"

        if not midi_path.exists():
            missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(str(midi_path))
            continue

        try:
            shape = render(midi_path, out_path, args.fps, args.velocity_threshold)
            success += 1
            if i % 500 == 0 or i == len(wav_files):
                print(f"[{i}/{len(wav_files)}] {track_name} shape={shape}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(wav_files)}] FAILED {track_name}: {e}")

    print()
    print("Done")
    print(f"  written : {success}")
    print(f"  skipped : {skipped}  (already cached)")
    print(f"  missing : {missing}  (wav had no matching .mid)")
    print(f"  failed  : {failed}")

    if missing:
        print()
        print("WARNING: some stems have audio but no ground-truth MIDI. Training")
        print("will fall back to an all-zero roll for those, silently weakening")
        print("the conditioning. Examples:")
        for m in missing_examples:
            print(f"  {m}")

    covered = success + skipped
    if covered:
        pct = 100.0 * covered / len(wav_files)
        print()
        print(f"coverage: {covered}/{len(wav_files)} stems ({pct:.1f}%)")


if __name__ == "__main__":
    main()
