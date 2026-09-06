from pathlib import Path
import argparse
import shutil
import subprocess
import numpy as np
import pretty_midi


def midi_to_pianoroll(midi_path: Path, out_path: Path, fps: int, velocity_threshold: float):
    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    velocity_roll = midi_data.get_piano_roll(fs=fps)
    piano_roll = (velocity_roll > velocity_threshold).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, piano_roll)

    return piano_roll.shape


def find_existing_midi(wav_path: Path, temp_midi_dir: Path):
    """The .mid basic-pitch would produce for this wav, if it is already here.

    Matched on the wav's own stem only. A track folder can hold more than one
    stem of the same instrument, so a bare *.mid match could return a different
    stem's transcription -- silently pairing the wrong notes with the audio.
    """
    matches = sorted(temp_midi_dir.glob(f"{wav_path.stem}*_basic_pitch.mid"))
    return matches[0] if matches else None


def run_basic_pitch(wav_path: Path, temp_midi_dir: Path) -> Path:
    temp_midi_dir.mkdir(parents=True, exist_ok=True)

    # Reuse a previous transcription if one is sitting here. basic-pitch is by
    # far the slowest step, and what it produces depends only on the wav --
    # neither --fps nor --velocity_threshold affects it, because both are
    # applied later when the roll is rendered. So rebuilding the cache at a
    # different fps or threshold must never re-transcribe.
    existing = find_existing_midi(wav_path, temp_midi_dir)
    if existing is not None:
        return existing

    cmd = [
        "basic-pitch",
        str(temp_midi_dir),
        str(wav_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    midi_path = find_existing_midi(wav_path, temp_midi_dir)

    if midi_path is None:
        raise FileNotFoundError(f"No MIDI file created for {wav_path}")

    return midi_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="midi_cache_cello")
    parser.add_argument("--temp_midi_root", type=str, default="midi_outputs/cache_temp_midis")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--velocity_threshold", type=float, default=40.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep_midi",
        action="store_true",
        help=(
            "Keep the intermediate .mid files instead of deleting them once the "
            ".npy is written. Worth doing if you may rebuild the cache at a "
            "different --fps or --velocity_threshold: both are applied when the "
            "roll is rendered, so kept MIDI is reused and the slow transcription "
            "step is skipped entirely. Costs one directory per track."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    cache_root = Path(args.cache_root)
    temp_midi_root = Path(args.temp_midi_root)

    wav_files = sorted(dataset_root.glob("*/stems_audio/*_cello.wav"))

    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    print(f"Found {len(wav_files)} cello wav files")

    success = 0
    skipped = 0
    failed = 0

    for i, wav_path in enumerate(wav_files, start=1):
        track_name = wav_path.parent.parent.name
        out_path = cache_root / track_name / f"{wav_path.stem}_pianoroll.npy"

        if out_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{i}/{len(wav_files)}] SKIP exists: {out_path}")
            continue

        temp_midi_dir = temp_midi_root / track_name

        try:
            midi_path = run_basic_pitch(wav_path, temp_midi_dir)
            shape = midi_to_pianoroll(
                midi_path=midi_path,
                out_path=out_path,
                fps=args.fps,
                velocity_threshold=args.velocity_threshold,
            )

            success += 1
            print(f"[{i}/{len(wav_files)}] OK {track_name} shape={shape}")

            # The .npy is written, so the .mid has served its purpose. Drop it
            # unless it was asked for -- one leftover directory per track adds
            # up to tens of thousands across a full dataset.
            # Only on success: a failed render is exactly when you want the
            # expensive transcription kept around to inspect.
            if not args.keep_midi:
                shutil.rmtree(temp_midi_dir, ignore_errors=True)

        except Exception as e:
            failed += 1
            print(f"[{i}/{len(wav_files)}] FAIL {track_name}: {e}")

    print("\nDone")
    print(f"success: {success}")
    print(f"skipped: {skipped}")
    print(f"failed: {failed}")

    if args.keep_midi:
        print(f"intermediate MIDI kept under: {temp_midi_root}")
    else:
        print("intermediate MIDI removed after each successful render "
              "(pass --keep_midi to keep it for re-rendering)")


if __name__ == "__main__":
    main()